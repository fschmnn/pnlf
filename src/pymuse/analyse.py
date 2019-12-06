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
        table[table[line]<table[f'{line}_err']][line] = table[table[line]<table[f'{line}_err']][f'{line}_err']
                       
    logger.info(f'{len(table)} entries in initial catalogue')
             
    # remove rows with NaN values in some columns
    mask =  np.ones(len(table), dtype=bool)
    for col in required:
        mask &=  ~np.isnan(table[col])
    table['type'][np.where(mask==False)] = 'NaN'
    #table = table[mask]
    logger.info(f'{len(mask[mask==False])} rows contain NaN values')

    mask = table['mOIII']< completeness_limit
    table['type'][np.where(mask==False)] = 'cl'
    #table = table[mask]
    logger.info(f'{len(mask[mask==False])} objects below the completness limit')    
                       
    table['type'][np.where(4 < np.log10(table['OIII5006'] / (table['HA6562']+table['NII6583'])))] = ''
    table['type'][np.where(np.log10(table['OIII5006'] / (table['HA6562']+table['NII6583'])) < -0.37*table['MOIII'] - 1.16)] = 'HII'
    table['type'][np.where(table['HA6562'] / table['SII6716'] < 2.5)] = 'SNR'

    logger.info(f'{len(table[table["type"]==""])} objects classified as 4<log [OIII]/Ha')
    logger.info(f'{len(table[table["type"]=="HII"])} objects classified as HII')
    logger.info(f'{len(table[table["type"]=="SNR"])} objects classified as SNR')
    logger.info(f'{len(table[table["type"]=="PN"])} possible planetary nebula found')
    
    return table

def PNLF(m,mu,completeness,truncate=False):
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

    completeness : float
        completeness level (magnitude of the faintest sources that
        are consistently detected).

    truncate : bool
        mask the output based on completeness (required for MLE)
    '''
    
    if not completeness:
        raise ValueError('specify completeness')

    Mmax = -4.47

    m = np.atleast_1d(m)

    if truncate:
        m = m[(m<completeness)]

    normalization = -3.62866*np.exp(0.307*Mmax) + 3.25733*np.exp(0.307*completeness-0.307*mu) + 0.371333 * np.exp(3*Mmax - 2.693 * completeness + 2.693 * mu)
    
    out = np.exp(0.307*(m-mu)) * (1-np.exp(3*(Mmax-m+mu))) / normalization
    
    out[(m>completeness) & (m<Mmax+mu)] = 0
    
    return out


class MaximumLikelihood:
    '''
    
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
        
        self.result = minimize(self._loglike,guess,args=(self.data),method=self.method)
        self.x = self.result.x
        if not self.result.success:
            raise RuntimeError('fit was not successful')

        self.dx = np.zeros((len(self.x),2))
        if np.any(self.err):
            
            self.result_m = minimize(self._loglike,guess,args=(self.data-self.err),method=self.method)
            self.result_p = minimize(self._loglike,guess,args=(self.data+self.err),method=self.method)

            if not self.result_m.success or not self.result_p.success:
                raise RuntimeError('fit for error was not successful')
            
            self.dx[:,0] = self.x - self.result_m.x
            self.dx[:,1] = self.result_p.x - self.x

        for name,_x,_dx in zip(list(signature(self.func).parameters)[1:],self.x,self.dx):
            print(f'{name} = {_x:.3f} + {_dx[1]:.3f} - {_dx[0]:.3f} ')

        return self.x

    def __call__(self,guess):
        '''use scipy minimize to find the best parameters'''

        return self.fit(guess)
        

