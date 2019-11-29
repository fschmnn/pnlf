import logging              # use instead of print for more control
from pathlib import Path    # filesystem related stuff
import numpy as np          # numerical computations

from astropy.table import Table

from scipy.optimize import minimize, curve_fit
from inspect import signature

logger = logging.getLogger(__name__)

def emission_line_diagnostics(table,distance_modulus,completeness_limit):
    '''Classify sources based on emission lines 
    
    criteria1:
    4 > log10 [OIII] / (Ha +[NII])
    
    criteria2:
    log10 [OIII] / (Ha +[NII]) > -0.37 M[OIII] - 1.16
    
    
    Parameters
    ----------
    table : Astropy Table
        Table with measured fluxes 
    
    completeness_limit : float
        Sources fainter than this magnitude are ignored
        
    distance_modulus : float 
       A first guess of the distance modulus (used for diagnostics)
       distance_modulus = m - M
    '''
    
        
    # make sure the input is of the correct type
    if not isinstance(table,Table):
        raise TypeError('wrong input')
    
    # next we check if all columns exist
    required = ['OIII5006','HA6562','NII6583','SII6716','mOIII']
    missing = set(required) - set(table.columns)
    if missing:
        raise KeyError(f'input table is missing {", ".join(missing)}')
    del missing
       
    #
    # Preparation
                       
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
    table = table[mask]
    logger.info(f'{len(mask[mask==False])} rows were removed because they contain NaN values')

    mask = table['mOIII']< completeness_limit
    table = table[mask]
    logger.info(f'{len(mask[mask==False])} objects below the completness limit removed')    
                       
    table['type'][np.where(4 < np.log10(table['OIII5006'] / (table['HA6562']+table['NII6583'])))] = ''
    table['type'][np.where(np.log10(table['OIII5006'] / (table['HA6562']+table['NII6583'])) < -0.37*table['MOIII'] - 1.16)] = 'HII'
    table['type'][np.where(table['HA6562'] / table['SII6716'] < 2.5)] = 'SNR'

    logger.info(f'{len(table[table["type"]=="HII"])} objects classified as HII')
    logger.info(f'{len(table[table["type"]==""])} objects classified as ...')
    logger.info(f'{len(table[table["type"]=="SNR"])} objects classified as SNR')
    logger.info(f'{len(table[table["type"]=="PN"])} possible planetary nebula found')
    
    return table

def pnlf(m,mu,N0):
    '''planetary nebula luminosity function
    
    Parameters
    ----------
        
    m : ndarray
        apparent magnitudes of the PNs
        
    mu : float
        distance modulus
    '''
    
    Mmax = -4.47
    delta = 2.5
    norm = np.exp(0.307*Mmax - 2.693*delta)*(0.371333 - 3.62866*np.exp(2.693*delta) + 3.25733 * np.exp(3*delta))
    
    return N0*np.exp(0.307*(m-mu)) * (1-np.exp(3*(Mmax-m+mu)))


class MaximumLikelihood:
    '''
    
    Parameters
    ----------
    func : function
        PDF of the form `func(data,params)`. `func` must accept a
        ndarray for `data` and can have any number of additional
        parameters.
        
    data : ndarray
        Measured data that are feed into `func`.
    '''
    
    def __init__(self,func,data):
        
        self.data = data
        if len(signature(func).parameters)<2:
            raise ValueError(f'`func` must accept at least two arguments')
        self.func = func

    def loglik(self,params):
        '''calculate the log liklihood of the given parameters
        
        This function takes the previously specified PDF and calculates
        the sum of the logarithmic probabilities.
        '''
        
        return np.sum(np.log(self.func(self.data,*params)))
    
    def fit(self,guess):
        '''use scipy minimize to find the best parameters'''
        
        self.result = minimize(self.loglik,guess,method ='Nelder-Mead')
        #for name,var in zip(list(signature(self.func).parameters)[1:],self.result.x):
        #    print(f'{name}={var:.3g}')
        return self.result.x

    def __call__(self,guess):
        return self.fit(guess)
        

