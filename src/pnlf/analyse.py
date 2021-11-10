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


def emission_line_diagnostics(table,distance_modulus,completeness_limit,distance_modulus_err=0.1,detection_limit=5):
    '''Classify objects based on their emission lines 
    
    we use four criteria to distinguish between PN, HII regions and SNR:
    
    criteria1 -> emperical upper limit (currently not used)
    4 > log10 [OIII] / (Ha +[NII])
    
    criteria2 -> HII regions
    log10 [OIII] / (Ha +[NII]) > -0.37 M[OIII] - 1.16
    
    criteria3 -> SNR
    Ha / [SII] < 2.5
    
    criteria4 -> velocity dispersion

    The second criteria requires the absolute magnitude of the objects. 
    Therefor the distance_modulus is needed as an input to this function
    
    
    Parameters
    ----------
    table : Astropy Table
        Table with measured fluxes 
    
    completeness_limit : float
        Sources fainter than this magnitude are ignored
        
    distance_modulus : float 
       A first guess of the distance modulus (used for diagnostics)
       distance_modulus = m - M
       
    detection_limit : float
        how many sigma are needed for a detection (default=3). Non 
        detections are set to detection_limit*uncertainty. 

    Returns
    -------
    table : Astropy Table
        The input table with an additional column, indicating the type of the object
    '''

    # we don't want to modift the original input table
    table = table.copy()

    logger.info(f'{len(table)} entries in initial catalogue')
    logger.info(f'using mu={distance_modulus:.2f}, cl={completeness_limit}')

    # make sure that the new column can save strings with 3 characters
    # we start with the assumption that all sources are PN and remove contaminants later
    table['type'] = np.empty(len(table),dtype='U3')
    table['type'][:] = 'PN'

    # calculate the absolute magnitude based on a first estimate of the distance modulus 
    table['MOIII'] = table['mOIII'] - distance_modulus
    table['dMOIII'] = np.sqrt(table['dmOIII']**2 + distance_modulus_err**2)

    # we need the the sum of the [SII] lines for the SNR criteria
    table['SII_flux'] = table['SII6716_flux']+table['SII6730_flux']
    table['SII_flux_err'] = np.sqrt(table['SII6716_flux_err']**2+table['SII6730_flux_err']**2)

    # we set negative fluxes and none detections to th error (0 would cause errors because we work with ratios)
    for col in ['HB4861','OIII5006','HA6562','NII6583','SII']:
        detection = (table[f'{col}_flux']>0) & (table[f'{col}_flux']>detection_limit*table[f'{col}_flux_err'])
        table[f'{col}_flux'][np.where(~detection)] = table[f'{col}_flux_err'][np.where(~detection)] 
        table[f'{col}_detection'] = detection

    # calculate velocity dispersion (use line with best signal to noise)
    table['OIII5006_S/N'] =  table['OIII5006_flux']/table['OIII5006_flux_err']
    table['HA6562_S/N']   =  table['HA6562_flux']/table['HA6562_flux_err']
    table['SII_S/N']  =  table['SII_flux']/table['SII_flux_err'] 

    # we use the Halpha velocity dispersion. the others can behave funny
    table['v_SIGMA']     = table['HA6562_SIGMA']
    table['v_SIGMA_S/N'] = table['HA6562_S/N']
    #table['v_SIGMA'][np.where(table['HA6562_SIGMA']/table['HA6562_SIGMA_ERR']>table['v_SIGMA_S/N'] )] = table['HA6562_SIGMA'][np.where(table['HA6562_SIGMA']/table['HA6562_SIGMA_ERR']>table['v_SIGMA_S/N'] )]
    #table['v_SIGMA_S/N'][np.where(table['HA6562_SIGMA']/table['HA6562_SIGMA_ERR']>table['v_SIGMA_S/N'] )] = table['HA6562_S/N'][np.where(table['HA6562_SIGMA']/table['HA6562_SIGMA_ERR']>table['v_SIGMA_S/N'] )] 
    #table['v_SIGMA'][np.where(table['SII6716_SIGMA']/table['SII6716_SIGMA_ERR']>table['v_SIGMA_S/N'])] = table['SII6716_SIGMA'][np.where(table['SII6716_SIGMA']/table['SII6716_SIGMA_ERR']>table['v_SIGMA_S/N'] )]
    #table['v_SIGMA_S/N'][table['SII6716_SIGMA']/table['SII6716_SIGMA_ERR']>table['v_SIGMA_S/N'] )] = table['SII_S/N'][np.where(table['SII6716_SIGMA']/table['SII6716_SIGMA_ERR']>table['v_SIGMA_S/N'] )] 
    #logger.info('v_sigma: median={:.2f}, median={:.2f}, sig={:.2f}'.format(*sigma_clipped_stats(table['v_SIGMA'][~np.isnan(table['v_SIGMA'])])))
    
    # define ratio of OIII to Halpha and NII for the first criteria (with error). If NII is not detected we assume NII=0.5Halpha
    table['logR']  =  np.log10(table['OIII5006_flux'] / (table['HA6562_flux']+table['NII6583_flux']))
    table['logR'][~table['NII6583_detection']] = np.log10(table['OIII5006_flux'][~table['NII6583_detection']] / (1.5*table['HA6562_flux'][~table['NII6583_detection']])) 
    table['dlogR'] = np.sqrt((table['OIII5006_flux_err'] / table['OIII5006_flux'])**2 + (table['HA6562_flux_err'] / (table['HA6562_flux']+table['NII6583_flux']))**2 + (table['NII6583_flux_err'] / (table['HA6562_flux']+table['NII6583_flux']))**2) /np.log(10) 

    # define criterias to exclude non PN objects
    criteria = {}

    if True:
        # this criteria should be log(4) which would remove a lot of objects. no idea why so many objects have such high line ratios
        #criteria[''] = (np.log10(4) < (table['R'])) #& (table['HA6562_detection'])
        criteria['HII'] = (table['logR'] < -0.37*(table['MOIII']+table['dMOIII']) - 1.16) & (table['HA6562_detection'])
        criteria['SNR'] = (table['HA6562_flux']/table['SII_flux'] < 2.5) & (table['SII_detection']) 
    elif True:
        # here we retain things in the sample if they are within 1 sigma
        #criteria[''] = (4 < (table['R']- 3*table['dR'])) #& (table['HA6562_detection'])
        criteria['HII'] = (table['logR'] + table['dlogR'] < -0.37*(table['MOIII']+table['dmOIII']) - 1.16) #& (table['HA6562_detection'] | table['NII6583_detection'])
        HAoverSII = table['HA6562_flux']/table['SII_flux']
        HAoverSII_err = HAoverSII * np.sqrt((table['HA6562_flux_err']/table['HA6562_flux'])**2 + (table['SII_flux_err']/table['SII_flux'])**2)
        criteria['SNR'] = ((HAoverSII + HAoverSII_err) < 2.5)

    else:
        # use HB as a criteria (because this line is close to OIII, extinction should not be an issue)
        criteria['HII'] = (np.log10(table['OIII5006'] / table['HB4861']) < -0.37*table['MOIII'] - 0.71) & table['HB4861_detection'] 

    #criteria['SNR'] |= (table['v_SIGMA']>100) & table['SII_detection']
    # objects that would be classified as PN by narrowband observations
    table['SNRorPN'] = criteria['SNR'] & ~criteria['HII']

    for k in criteria.keys():
        table['type'][np.where(criteria[k])] = k

    # remove rows with NaN values in some columns
    mask =  np.ones(len(table), dtype=bool)
    for col in ['HB4861_flux','OIII5006_flux','HA6562_flux','NII6583_flux','SII_flux']:
        mask &=  ~np.isnan(table[col])
    table['type'][np.where(~mask)] = 'NaN'

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
        # evidence and likelihood are in log 
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

        #logger.info(f'{self.x:.3f}+{self.plus:.3f}-{self.minus:.3f}')

        return self.x,self.plus,self.minus

    def bootstrap(self,guess,N_boot=100):
        '''use bootstraping to estinate the uncertainties'''

        loglike = lambda param,data: -np.sum(np.log(self.func(data,param,**self.kwargs)))
        result_bootstrap = np.zeros((N_boot))

        for i in range(N_boot):
            sample = np.random.choice(self.data,len(self.data))
            result_bootstrap[i] = minimize(loglike,guess,args=(sample),method=self.method).x[0]

        return np.mean(result_bootstrap),np.std(result_bootstrap)

    def plot(self,limits=[]):
        '''plot the likelihood
        
        plot the evidence, prior and likelihood for the given data over
        some parameters space.
        '''

        if not hasattr(self,'x'):
            logger.warning('run fit function first. I do it for you this time.')
            x,dp,dm=self.fit(28        )
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

def cdf(x,mu,mhigh,Mmax = -4.47):
    '''Cumulative distribution function for PNe'''

    mlow = mu+Mmax
    
    normalization = 1/(F(mhigh,mu,Mmax=Mmax)-F(mlow,mu,Mmax=Mmax))
    out = normalization * (F(x,mu,Mmax=Mmax) - F(mlow,mu,Mmax=Mmax))
    
    out[x<mlow]  = 0
    out[x>mhigh] = 1
    
    return out

def sample_pnlf(size,mu,cl):
    
    Nbins = 1000
    x =np.linspace(mu-4.47,cl,Nbins)
    cdf = np.cumsum(pnlf(x,mu,cl))*(cl-mu+4.47)/Nbins
    u = np.random.uniform(size=size)
    
    return np.interp(u,cdf,x)

'''
from scipy.stats import ks_2samp
from pnlf.analyse import sample_pnlf

sampled_data = sample_pnlf(10000,galaxy.mu,galaxy.completeness_limit)
ks,pv = ks_2samp(data,sampled_data)
print(f'statistic={ks:.3f}, pvalue={pv:.3f}')
'''


def prior(mu):
    mu0 = 29.91
    std = 0.13
    
    return 1 / (std*np.sqrt(2*np.pi)) * np.exp(-(mu-mu0)**2 / (2*std**2))


def N25(mu,completeness,data,deltaM):
    '''calculate the number of PN within deltaM
    
    the cutoff of the luminosity function is at mu-4.47.
    
    Step1: number of PN in data between cutoff and completeness
    Step2: calculate same number from theoretical function
    Step3: calculate theoreticla number betwenn cutoff and deltaM
    Step4: Scale number from Step1 with results from Step 2 and 3
    
    Parameters
    ----------
    mu : float
        distance modulus 
    completeness : float
        completeness limit (upper limit for PNLF). Used for normalization
    data : ndarray
        array of magnitudes
    deltaM : float
        Interval above the cutoff
    '''
    
    cutoff = mu - 4.47
    
    N_total  = len(data[data<completeness])
    p_deltaM = (F(cutoff+deltaM,mu) - F(cutoff,mu)) / (F(completeness,mu) - F(cutoff,mu))
    
    return N_total * p_deltaM



from scipy.optimize import minimize


def estimate_uncertainties_from_SII(tbl,plot=False):
    '''
    The uncertainties in the PHANGS-MUSE DAP products are somewhat underestimated. 
    To get a better handle on the errors, we use that the SII6716/SII6730 ratio
    should theoretically be 1.4484. Any diviation from this value can be attributed
    to the errors in the measurements. We divide the diviation by the error of the
    ratio. This should follow a gaussian with width 1. From the actuall width of 
    the distribution we can estimate the real uncertainty of the data.
    
    to use Francescos catalogue instead:

    ```
    with fits.open(data_ext/'Products'/'Nebulae catalogue' / 'Nebulae_catalogue_v2.fits') as hdul:
        nebulae = Table(hdul[1].data)
    nebulae['gal_name'][nebulae['gal_name']=='NGC628'] = 'NGC0628'
    nebulae = nebulae[(nebulae["flag_edge"] == 0) & (nebulae["flag_star"] == 0) & (nebulae["BPT_NII"] == 0) & (nebulae["BPT_SII"] == 0) & (nebulae["BPT_OI"] == 0) & (nebulae['HA6562_SIGMA'] < 100)]
    nebulae.rename_columns(['SII6730_FLUX','SII6730_FLUX_ERR','SII6716_FLUX','SII6716_FLUX_ERR'],['SII6730','SII6730_err','SII6716','SII6716_err'])
    nebulae['type'] = 'HII'
    ```

    '''

    if tbl is None:
        from astropy.io import fits 

        with fits.open(Path('a:') /'MUSE_DR2' /'Nebulae catalogue' / 'Nebulae_Catalogue_DR2_native.fits') as hdul:
            tbl = Table(hdul[1].data)
        tbl['gal_name'][tbl['gal_name']=='NGC628'] = 'NGC0628'
        tbl = tbl[(tbl["flag_edge"] == 0) & (tbl["flag_star"] == 0) & (tbl["BPT_NII"] == 0) & (tbl["BPT_SII"] == 0) & (tbl["BPT_OI"] == 0) & (tbl['HA6562_SIGMA'] < 100)]
        tbl.rename_columns(['SII6730_FLUX','SII6730_FLUX_ERR','SII6716_FLUX','SII6716_FLUX_ERR'],['SII6730','SII6730_err','SII6716','SII6716_err'])
        tbl['type'] = 'HII'

    tmp = tbl[(tbl['type']=='HII') & (tbl['SII6730']>0) &  (tbl['SII6716']>0)]
    logger.info(f'sample contains {len(tmp)} nebulae')
    
    ratio =  tmp['SII6716'] / tmp['SII6730']
    ratio_err =  ratio * np.sqrt((tmp['SII6716_err']/tmp['SII6716'])**2+(tmp['SII6730_err']/tmp['SII6730'])**2)
    diff_norm = (ratio-1.4484) / ratio_err
    diff_norm = diff_norm[diff_norm>0] 
    
    gauss = lambda x,mu,std: 1/np.sqrt(2* np.pi*std**2) * np.exp(-0.5 * ((x - mu) / std)**2)
    log_likelihood = lambda std,x: - np.sum(np.log(gauss(x,0,std)))
    std = minimize(log_likelihood,[1],args=(diff_norm,)).x[0]

    if plot:
        fig, ax = plt.subplots(figsize=(4,3))

        hist, bins, patches = ax.hist((ratio-1.4484) / ratio_err,bins=20,range=(0,6),color='silver')
        
        y2 = gauss(bins,0,1)
        ax.plot(bins,hist[0]/y2[0]*y2, '--',label='std=1',color='tab:blue')
        y = gauss(bins,0,std)
        ax.plot(bins,hist[0]/y[0]*y, '--',label=f'std={std:.2f}',color='tab:red')

        ax.legend()
        
        ax.set(xlabel="Deviation / Error",ylabel="Number of regions",yscale='log',ylim=[1,1.5*hist[0]])
        plt.show()
    
    logger.info(f'std={std:.3f}')
    return std

