import logging              # use instead of print for more control
from pathlib import Path    # filesystem related stuff
import numpy as np          # numerical computations

from astropy.io import ascii
from astropy.table import Table                # useful data structure
from astropy.table import vstack               # combine multiple tables

from astropy.stats import sigma_clipped_stats  # calcualte statistics of images
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
from astropy.coordinates import SkyCoord

from photutils import DAOStarFinder            # DAOFIND routine to detect sources
from photutils import IRAFStarFinder           # IRAF starfind routine to detect stars

from collections import OrderedDict                           # make random table reproducable
from photutils.datasets import make_random_gaussians_table    # create table with mock sources

#from astropy.convolution import convolve, Gaussian2DKernel

from .io import ReadLineMaps

basedir = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)

# IRAFStarFinder complains about NaN values in lt and gt
np.warnings.filterwarnings('ignore')

def detect_unresolved_sources(
    self : ReadLineMaps,
    line : list,
    StarFinder,
    data = None,
    threshold : float=4.,
    PSF_size : float=1.,
    save=False
    ) -> Table:
    '''detect unresolved sources in a ReadLineMaps object
    
    Parameters
    ----------
    self : obj. ReadLineMaps
        Object in which to search for unresolved sources.
        
    line : string
        name of a line map in self that is used for the detection.
        
    StarFinder :
        astropy StarFinder (DAO or IRAF)

    threshold : 
        detection threshold in terms of background median

    save : bool
        save the result is to a file in `reports/catalogues/`
    '''
    
    if not isinstance(self,ReadLineMaps):
        raise TypeError('input must be of type ReadLineMaps')
    
    # for convenience only, to make accessing the data easier
    data = getattr(self,line)
    err  = getattr(self,f'{line}_err')
    PSF  = getattr(self,'PSF') 

    #sigma_max = np.nanmax(PSF) * gaussian_fwhm_to_sigma
    #kernel = Gaussian2DKernel(x_stddev=sigma_max)
    #data = convolve(data, kernel)
    #PSF[:,:] = sigma_max

    logger.info(f'searching for sources in {self.name} with [{line}] line map (using ' + \
          str(StarFinder).split('.')[-1][:-2] + ')\n' )
    
    # header for the print information
    logger.info(f'{"fwhm":>9}{"#N":>5}{"mean":>8}{"median":>8}{"std":>8}')
    
    # loop over all pointings with different PSFs
    for fwhm in np.unique(PSF[~np.isnan(PSF)]):
                
        # we create a mask for the current pointing (must be inverted)
        mask = ~(PSF == fwhm)

        mean, median, std = sigma_clipped_stats(err[(~np.isnan(err)) & (~mask)], sigma=3.0)

        # initialize daofind 
        # FWHM is given in arcsec. one pixel is 0.2" 
        finder = StarFinder(fwhm      = fwhm * PSF_size, 
                            threshold = threshold*median,
                            sharplo   = 0.2, 
                            sharphi   = 0.8,
                            roundlo   = -0.7,
                            roundhi   = 0.7)
        
        peaks_part = finder(data, mask=mask)
            
        # save fwhm in an additional column
        peaks_part['fwhm'] = fwhm
        
        n_sources = len(peaks_part)
        logger.info(f'{fwhm:>7.3f}px{n_sources:>5.0f}{mean:>8.3f}{median:>8.3f}{std:>8.3f}')
        
        # concatenate new sources with output table
        if 'peak_tbl' in locals():
            peaks_part['id'] += np.amax(peak_tbl['id'],initial=0)
            peak_tbl = vstack([peak_tbl,peaks_part])
        # if no output table exists we create a new one
        else:
            peak_tbl = peaks_part

    logger.info(f'  total{len(peak_tbl):>7.0f}')

    # for consistent table output
    for col in peak_tbl.colnames:
        peak_tbl[col].info.format = '%.8g'  
    peak_tbl['fwhm'].info.format = '%.3g' 
    peak_tbl.rename_column('xcentroid','x')
    peak_tbl.rename_column('ycentroid','y')
          
    # calculate astronomical coordinates
    peak_tbl['SkyCoord'] = SkyCoord.from_pixel(peak_tbl['x'],peak_tbl['y'],self.wcs)
    peak_tbl['RaDec'] = peak_tbl['SkyCoord'].to_string(style='hmsdms',precision=2)

    # save the result to the object
    setattr(self,'peaks_tbl',peak_tbl)

    # we save the found positions to a file
    if save:
        filename = basedir / 'reports' / 'catalogues' / f'peaks_{self.name}.txt'
        with open(filename,'w',newline='\n') as f:
            ascii.write(peak_tbl[['id','x','y','peak','flux','RaDec','fwhm']],
                        f,format='fixed_width',overwrite=True)

    return peak_tbl


