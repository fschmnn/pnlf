import logging              # use instead of print for more control
from pathlib import Path    # filesystem related stuff
import numpy as np          # numerical computations
from matplotlib.pyplot import subplots, figure, savefig
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats  # calcualte statistics of images
from astropy.table import Table

from scipy.optimize import minimize
from scipy.integrate import quad
from inspect import signature

from .constants import single_column,two_column,tab10
from .old import MaximumLikelihood

logger = logging.getLogger(__name__)


def emission_line_diagnostics(table,distance_modulus,completeness_limit,SNR=True):
    '''Classify objects based on their emission lines 
    
    we use three criteria to distinguish between PN, HII regions and SNR:
    
    criteria1 -> emperical upper limit
    4 > log10 [OIII] / (Ha +[NII])
    
    criteria2 -> HII regions
    log10 [OIII] / (Ha +[NII]) > -0.37 M[OIII] - 1.16
    
    criteria3 -> SNR
    Ha / [SII] < 2.5
    
    The second criteria requires the absolute magnitude of the objects. 
    Therefor the distance_modulus
    
    
    Parameters
    ----------
    table : Astropy Table
        Table with measured fluxes 
    
    completeness_limit : float
        Sources fainter than this magnitude are ignored
        
    distance_modulus : float 
       A first guess of the distance modulus (used for diagnostics)
       distance_modulus = m - M
       
    SNR : bool
        remove supernovae remnants based on SII to HA ratio

    Returns
    -------
    table : Astropy Table
        The input table with an additional column, indicating the type of the object
    '''
    
    # next we check if all columns exist
    required = ['OIII5006','HA6562','NII6583','SII6716','mOIII']
    missing = set(required) - set(table.columns)
    if missing:
        raise KeyError(f'input table is missing {", ".join(missing)}')
    del missing
       
    # we don't want to modift the input table
    table = table.copy()

    logger.info(f'{len(table)} entries in initial catalogue')
    logger.info(f'using mu={distance_modulus}')

    # make sure that the new column can save strings with 3 characters
    table['type'] = np.empty(len(table),dtype='U3')
    table['type'][:] = 'PN'

    # if the flux is smaller than the error we set it to the error
    for col in ['OIII5006','HA6562','NII6583','SII6716']:
        # median of error maps is a factor of 3 smaller than std of maps
        detection = (table[col]>0) & (table[col]>3*table[f'{col}_err'])
        logger.info(f'{np.sum(~detection)} not detected in {col}')
        table[col][np.where(table[col]<0)] = table[f'{col}_err'][np.where(table[col]<0)] 
        #table[col][np.where(~detection)] = table[f'{col}_err'][np.where(~detection)] 
        table[f'{col}_detection'] = detection

    # calculate the absolute magnitude based on a first estimate of the distance modulus 
    table['MOIII'] = table['mOIII'] - distance_modulus

    # calculate velocity dispersion (use line with best signal to noise)
    table['OIII5006_S/N'] =  table['OIII5006']/table['OIII5006_err']
    table['HA6562_S/N']   =  table['HA6562']/table['HA6562_err']
    table['SII6716_S/N']  =  table['SII6716']/table['SII6716_err'] 

    table['v_SIGMA']     = table['OIII5006_SIGMA']
    table['v_SIGMA_S/N'] = table['OIII5006_S/N'] 

    table['v_SIGMA'][np.where(table['HA6562_S/N']>table['v_SIGMA_S/N'] )] = table['HA6562_SIGMA'][np.where(table['HA6562_S/N']>table['v_SIGMA_S/N'] )]
    table['v_SIGMA_S/N'][np.where(table['HA6562_S/N']>table['v_SIGMA_S/N'] )] = table['HA6562_S/N'][np.where(table['HA6562_S/N']>table['v_SIGMA_S/N'] )] 
    
    table['v_SIGMA'][np.where(table['SII6716_S/N']>table['v_SIGMA_S/N'])] = table['SII6716_SIGMA'][np.where(table['SII6716_S/N']>table['v_SIGMA_S/N'] )]
    table['v_SIGMA_S/N'][np.where(table['SII6716_S/N']>table['v_SIGMA_S/N'] )] = table['SII6716_S/N'][np.where(table['SII6716_S/N']>table['v_SIGMA_S/N'] )] 

    logger.info('v_sigma: median={:.2f}, median={:.2f}, sig={:.2f}'.format(*sigma_clipped_stats(table['v_SIGMA'][~np.isnan(table['v_SIGMA'])])))
    
    table['R']  =  np.log10(table['OIII5006'] / (table['HA6562']+table['NII6583']))
    table['dR'] = np.sqrt((table['OIII5006_err'] / table['OIII5006'])**2 + (table['HA6562_err'] / (table['HA6562']+table['NII6583']))**2 + (table['NII6583_err'] / (table['HA6562']+table['NII6583']))**2) /np.log(10) 

    # define criterias to exclude non PN objects
    criteria = {}
    criteria[''] = (4 <table['R']-table['dR']) #& (table['HA6562_detection'])
    
    #criteria['HII'] = (10**(table['R']+table['dR']) < 1.6)
    criteria['HII'] = (table['R'] + table['dR'] < -0.37*table['MOIII'] - 1.16) #& (table['HA6562_detection'] | table['NII6583_detection'])
    criteria['SNR'] = ((table['HA6562']) / (table['SII6716']) < 2.5)  & (table['SII6716_detection']) 
    # only apply this criteria if signal to noise is > 3
    criteria['SNR'] |= ((table['v_SIGMA']>100) & (table['v_SIGMA_S/N']>3))

    # objects that would be classified as PN by narrowband observations
    table['SNRorPN'] = criteria['SNR'] & ~criteria['HII']

    for k in criteria.keys():
        table['type'][np.where(criteria[k])] = k

    # remove rows with NaN values in some columns
    mask =  np.ones(len(table), dtype=bool)
    for col in required:
        mask &=  ~np.isnan(table[col])
    table['type'][np.where(~mask)] = 'NaN'
    #table = table[mask]
    logger.info(f'{np.sum(~mask)} rows contain NaN values')

    # purely for information
    mask = table['mOIII']< completeness_limit
    logger.info(f'{np.sum(~mask)} objects below the completness limit of {completeness_limit}')    

    logger.info(f'{len(table[table["type"]==""])} objects classified as 4<log [OIII]/Ha')
    logger.info(f'{len(table[table["type"]=="HII"])} ({len(table[(table["type"]=="HII") & (table["mOIII"]<completeness_limit)])}) objects classified as HII')
    logger.info(f'{len(table[table["type"]=="SNR"])} ({len(table[(table["type"]=="SNR") & (table["mOIII"]<completeness_limit)])}) objects classified as SNR')
    logger.info(f'{len(table[table["type"]=="PN"])} ({len(table[(table["type"]=="PN") & (table["mOIII"]<completeness_limit)])}) objects classified as PN')
    
    return table

def gaussian(x,mu,sig):
    return 1/np.sqrt(2*np.pi*sig**2) * np.exp(-(x-mu)**2/(2*sig**2))



class MaximumLikelihood1D:
    '''

    for uncertainties 
    https://erikbern.com/2018/10/08/the-hackers-guide-to-uncertainty-estimates.html
    
    Parameters
    ----------
    func : function
        PDF of the form `func(data,params)`. `func` must accept a
        ndarray for `data` and can have any number of additional
        parameters (at least one).
        
    data : ndarray
        Measured data that are feed into `func`.

    err : ndarray
        Error associated with data.

    prior : function
        Prior probabilities for the parameters of func.

    method : 
        algorithm that is used for the minimization.

    **kwargs
       additional fixed key word arguments that are passed to func.
    '''
    
    def __init__(self,func,data,err=None,prior=None,method='Nelder-Mead',**kwargs):
        
        #if len(signature(func).parameters)-len(kwargs)!=2:
        #    raise ValueError(f'`func` must have at least one free argument')
        self.func = func

        logger.info(f'initialize fitter with {len(data)} data points')
        self.data   = data
        self.err    = err
        if prior:
            self.prior = prior
        self.method = method
        self.kwargs = kwargs

        width = 5
        size = 1000
        
        idx_low = np.argmin(self.data)
        idx_high = np.argmax(self.data)
        if np.any(err):
            self.grid = np.linspace(self.data[idx_low]-width*self.err[idx_low],self.data[idx_high]+width*self.err[idx_high],size)
        
    def prior(self,*args):
        '''uniform prior'''
        return 1/len(self.data)

    def evidence(self,param):
        '''the evidence is the likelihood of observing the data given the parameter'''
        
        # real integration takes way too long
        #return -np.sum(np.log([quad(lambda x: self.func(x,param,**self.kwargs)*gaussian(x,d,e),d-5*e,d+5*e)[0] for d,e in zip(self.data,self.err)]))
        
        if np.any(self.err):
            ev = [np.trapz(self.func(self.grid,param,**self.kwargs)*gaussian(self.grid,d,e),self.grid) for d,e in zip(self.data,self.err)]                
            return np.sum(np.log(ev))
        else:
            ev = self.func(self.data,param,**self.kwargs)
            return np.sum(np.log(ev))
        
    def likelihood(self,param):
        '''the evidence multiplied with some prior'''
        
        return -self.evidence(param) - np.log(self.prior(param)) 
        
    def fit(self,guess):
        '''use scipy minimize to find the best parameters'''
        
        #logger.info(f'searching for best parameters with {len(self.data)} data points')

        self.result = minimize(self.likelihood,guess,method=self.method)
        self.x = self.result.x[0]
        if not self.result.success:
            raise RuntimeError('fit was not successful')

        #for name,_x,_dx in zip(list(signature(self.func).parameters)[1:],self.x,self.dx):
        #    print(f'{name} = {_x:.3f} + {_dx[1]:.3f} - {_dx[0]:.3f} ')
        size = 0.5
        self.x_arr = np.linspace(self.x-size,self.x+size,1000)
        self.evidence_arr   = np.exp([self.evidence(_) for _ in self.x_arr])
        self.prior_arr      = np.array([self.prior(_) for _ in self.x_arr])
        self.likelihood_arr = np.exp([-self.likelihood(_) for _ in self.x_arr])
 
        valid = ~np.isnan(self.evidence_arr) &  ~np.isnan(self.likelihood_arr) 
        self.evidence_arr   /= np.abs(np.trapz(self.evidence_arr[valid],self.x_arr [valid]))
        self.prior_arr      /= np.abs(np.trapz(self.prior_arr[valid],self.x_arr [valid]))
        self.likelihood_arr /= np.abs(np.trapz(self.likelihood_arr[valid],self.x_arr [valid]))

        normalization = np.trapz(self.likelihood_arr,self.x_arr )
        self.integral = np.array([np.trapz(self.likelihood_arr[self.x_arr<=xp],self.x_arr[self.x_arr<=xp])/normalization for xp in self.x_arr[1:]])
       
        # 1 sigma interval for cumulative likelihood
        self.mid = np.argmin(np.abs(self.integral-0.5))
        self.high = np.argmin(np.abs(self.integral-0.8415))
        self.low = np.argmin(np.abs(self.integral-0.1585))

        self.plus  = self.x_arr[self.high]-self.x
        self.minus = self.x-self.x_arr[self.low]

        logger.info(f'{self.x:.3f}+{self.plus:.3f}-{self.minus:.3f}')

        return self.x,self.plus,self.minus

    def plot(self,limits=[]):
        '''plot the likelihood
        
        plot the evidence, prior and likelihood for the given data over
        some parameters space.
        '''

        if not hasattr(self,'x'):
            logger.warning('run fit function first. I do it for you this time.')
            x,dp,dm=self.fit(24)
        else:
            x,dp,dm = self.x,self.plus,self.minus
        
        fig, (ax1,ax2) = subplots(nrows=2,ncols=1,figsize=(single_column,single_column),sharex=True)
        #fig = figure(figsize=(single_column,single_column))
        #ax1 = fig.add_subplot(2,1,1)
        #ax2 = fig.add_subplot(2,1,2,sharex=ax1)
        ax1.tick_params(labelbottom=False)

        ax1.plot(self.x_arr,self.prior_arr,label='prior',color=tab10[1])
        ax1.plot(self.x_arr,self.evidence_arr,label='evidence',color=tab10[0])
        l = ax1.plot(self.x_arr,self.likelihood_arr,label='likelihood',color=tab10[2])
        
        ax1.axvline(self.x,ls='--',c='k',lw=0.5)
        ax1.axvline(self.x_arr[self.low],ls='--',c='k',lw=0.5)
        ax1.axvline(self.x_arr[self.high],ls='--',c='k',lw=0.5)
        

        ax1.set_ylabel('likelihood')
        
        ax2.plot(self.x_arr[1:],self.integral,label='cumulative likelihood',color=tab10[2])
        ax2.axvline(self.x,ls='--',c='k',lw=0.5)
        ax2.axhline(0.5,ls='--',c='k',lw=0.5)

        ax2.axhline(0.5+0.683/2,ls='--',c='k',lw=0.5)
        ax2.axhline(0.5-0.683/2,ls='--',c='k',lw=0.5)
        ax2.axvline(self.x_arr[self.low],ls='--',c='k',lw=0.5)
        ax2.axvline(self.x_arr[self.high],ls='--',c='k',lw=0.5)

        ax1.legend()

        ax2.set_xlabel(r'$(m-M)$ / mag')
        ax2.set_ylabel('cumulative likelihood')
        ax1.set_title(f'{self.x:.3f}+{dp:.3f}-{dm:.3f}')
        ax1.annotate(f'{len(self.data)} data points',(0.02,0.87),xycoords='axes fraction',fontsize=8)
        plt.subplots_adjust(hspace = .001)      

        return fig 

    def __call__(self,guess):
        '''use scipy minimize to find the best parameters'''

        return self.fit(guess)


def f(m,mu,Mmax=-4.47):
    '''luminosity function (=density)'''
    
    return np.exp(0.307*(m-mu)) * (1-np.exp(3*(Mmax-m+mu)))


def F(m,mu,Mmax=-4.47):
    '''indefinite integral of the luminosity function'''

    #return np.exp(-0.307*mu) * (np.exp(0.307*m)/0.307 + np.exp(3*(Mmax-mu)-2.693*m) / 2.693)
    return np.exp(0.307*(m-mu))/0.307 + np.exp(2.693*(mu-m)+3*Mmax)/2.693


def pnlf(m,mu,mhigh,Mmax=-4.47):
    '''Planetary Nebula Luminosity Function (PNLF)
    
    N(m) ~ e^0.307(m-mu) * (1-e^3(Mmax-m+mu))
        
    The normalization is calculated by integrating from Mmax+mu
    (the root of the function) to the specified completeness. 
    Objects that lie outside this intervall are ignored.

    Parameters
    ----------
    m : ndarray
        apparent magnitudes of the PNs
        
    mu : float
        distance modulus

    mhigh : float
        completeness level (magnitude of the faintest sources that
        are consistently detected). Required for normalization.
    '''

    m = np.atleast_1d(m)
    mlow = Mmax+mu
    
    normalization = 1/(F(mhigh,mu) - F(mlow,mu))    
    out = normalization * np.exp(0.307*(m-mu)) * (1-np.exp(3*(Mmax-m+mu)))
    out[(m>mhigh) | (m<mlow)] = 0
    
    return out

def PNLF(bins,mu,mhigh,Mmax=-4.47):
    '''integrated Planetary Nebula Luminosity Function
    
    Parameters
    ----------
    
    bins : ndarray
        Defines a monotonically increasing array of bin edges.
    
    mu : float
        Distance modulus
        
    mhigh : float
        completness level (magnitude of the faintest sources that
        are consistently detected). Required for normalization.
    
    Mmax : float
        Magnitude of the brightest PN.
    '''
    
    mlow = mu+Mmax

    lower = bins[:-1]
    upper = bins[1:]
    
    normalization = 1/(F(mhigh,mu,Mmax=Mmax)-F(mlow,mu,Mmax=Mmax))
    #out = normalization * (np.exp(2.693*mu + 3.*Mmax) * (-0.371333/np.exp(2.693*lower)) + 0.371333/np.exp(2.693*upper)) + (-3.25733*np.exp(0.307*lower) + 3.25733*np.exp(0.307*upper))/np.exp(0.307*mu)
    out = normalization * (F(upper,mu,Mmax=Mmax) - F(lower,mu,Mmax=Mmax))
    out[out<0] = 0

    return out
