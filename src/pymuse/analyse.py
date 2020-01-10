import logging              # use instead of print for more control
from pathlib import Path    # filesystem related stuff
import numpy as np          # numerical computations

from astropy.table import Table

from scipy.optimize import minimize, curve_fit
from inspect import signature

logger = logging.getLogger(__name__)


def emission_line_diagnostics(table,distance_modulus,completeness_limit):
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
       
    # otherwise we modify the input table
    table = table.copy()
                       
    # calculate the absolute magnitude based on a first estimate of the distance modulus 
    table['MOIII'] = table['mOIII'] - distance_modulus
    # make sure that the new column can save strings with 3 characters
    table['type'] = np.empty(len(table),dtype='U3')
    table['type'][:] = 'PN'
                       
    # if the flux is smaller than the error we set it to the error
    for line in ['OIII5006','HA6562','NII6583','SII6716']:
        mask = table[line]<table[f'{line}_err']
        #print(f'{line}: {np.sum(mask)} values replaced')
        table[line][np.where(mask)] = table[f'{line}_err'][np.where(mask)]
                       
    logger.info(f'{len(table)} entries in initial catalogue')
                       
    table['type'][np.where(4 < np.log10(table['OIII5006'] / (table['HA6562']+table['NII6583'])))] = ''
    table['type'][np.where(np.log10(table['OIII5006'] / (table['HA6562']+table['NII6583'])) < -0.37*table['MOIII'] - 1.16)] = 'HII'
    table['type'][np.where(table['HA6562'] / table['SII6716'] < 2.5)] = 'SNR'

    # remove rows with NaN values in some columns
    mask =  np.ones(len(table), dtype=bool)
    for col in required:
        mask &=  ~np.isnan(table[col])
    table['type'][np.where(mask==False)] = 'NaN'
    #table = table[mask]
    logger.info(f'{len(mask[mask==False])} rows contain NaN values')

    mask = table['mOIII']< completeness_limit
    #table['type'][np.where(mask==False)] = 'cl'
    #table = table[mask]
    logger.info(f'{len(mask[mask==False])} objects below the completness limit')    


    logger.info(f'{len(table[table["type"]==""])} objects classified as 4<log [OIII]/Ha')
    logger.info(f'{len(table[table["type"]=="HII"])} objects classified as HII')
    logger.info(f'{len(table[table["type"]=="SNR"])} objects classified as SNR')
    logger.info(f'{len(table[table["type"]=="PN"])} possible planetary nebula found')
    
    return table

class MaximumLikelihood:
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

    method : 
        algorithm that is used for the minimization.

    **kwargs
       additional fixed key word arguments that are passed to func.
    '''
    
    def __init__(self,func,data,err=None,method='Nelder-Mead',**kwargs):
        
        if len(signature(func).parameters)-len(kwargs)<2:
            raise ValueError(f'`func` must have at least one free argument')
        self.func = func

        self.data   = data
        self.err    = err
        self.method = method
        self.kwargs = kwargs

    def _loglike(self,params,data):
        '''calculate the log liklihood of the given parameters
        
        This function takes the previously specified PDF and calculates
        the sum of the logarithmic probabilities. If key word arguments
        were initially passed to the class, they are also passed to the
        function
        '''
        return -np.sum(np.log(self.func(data,*params,**self.kwargs)))

    def fit(self,guess):
        '''use scipy minimize to find the best parameters'''
        
        logger.info(f'searching for best parameters with {len(self.data)} data points')

        self.result = minimize(self._loglike,guess,args=(self.data),method=self.method)
        self.x = self.result.x
        if not self.result.success:
            raise RuntimeError('fit was not successful')

        self.dx = np.zeros((len(self.x),2))
        if np.any(self.err):
            
            B = 500
            #bootstrapping
            result_bootstrap = np.zeros((B,len(self.x)))
            for i in range(B):
                sample = np.random.normal(self.data,self.err)
                result_bootstrap[i,:] = minimize(self._loglike,guess,args=(sample),method=self.method).x
            err_boot = np.sqrt(np.sum((result_bootstrap-self.x)**2,axis=0)/B)
            self.dx[:,0] = err_boot 
            self.dx[:,1] = err_boot  
        
            '''
            self.result_m = minimize(self._loglike,guess,args=(self.data-self.err),method=self.method)
            self.result_p = minimize(self._loglike,guess,args=(self.data+self.err),method=self.method)

            if not self.result_m.success or not self.result_p.success:
                raise RuntimeError('fit for error was not successful')
            
            self.dx[:,0] = self.x - self.result_m.x
            self.dx[:,1] = self.result_p.x - self.x
            '''

        else:
            B = 500
            #bootstrapping
            result_bootstrap = np.zeros((B,len(self.x)))
            for i in range(B):
                sample = np.random.choice(self.data,len(self.data))
                result_bootstrap[i,:] = minimize(self._loglike,guess,args=(sample),method=self.method).x
            err_boot = np.sqrt(np.sum((result_bootstrap-self.x)**2,axis=0)/B)
            self.dx[:,0] = err_boot 
            self.dx[:,1] = err_boot  

        for name,_x,_dx in zip(list(signature(self.func).parameters)[1:],self.x,self.dx):
            print(f'{name} = {_x:.3f} + {_dx[1]:.3f} - {_dx[0]:.3f} ')

        return self.x

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
    out[(m>mhigh) & (m<mlow)] = 0
    
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