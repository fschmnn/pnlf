import logging              # use instead of print for more control
from pathlib import Path    # filesystem related stuff
import numpy as np          # numerical computations

import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table                # useful data structure
from astropy.table import vstack               # combine multiple tables

from astropy.stats import sigma_clipped_stats  # calcualte statistics of images
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
from astropy.coordinates import SkyCoord

from photutils import DAOStarFinder            # DAOFIND routine to detect sources
from photutils import IRAFStarFinder           # IRAF starfind routine to detect stars

from collections import OrderedDict                           # make random table reproducable
from photutils.datasets import make_gaussian_sources_image    # create table with mock sources

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
    threshold : float=4.,
    oversize_PSF : float=1.,
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

        mean, median, std = sigma_clipped_stats(err[(~np.isnan(PSF)) & (~mask)], sigma=3.0)

        # initialize daofind 
        # FWHM is given in arcsec. one pixel is 0.2" 
        finder = StarFinder(fwhm      = fwhm * oversize_PSF, 
                            threshold = np.abs(threshold*median),
                            sharplo   = 0.2, 
                            sharphi   = 0.8,
                            roundlo   = -0.4,
                            roundhi   = 0.4)
        
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


def match_catalogues(matchcoord,catalogcoord):
    '''match elements between two catalogues by distance
    
    Parameters
    ----------
    matchcoord : astropy Table
        Table with two columns for x and y positions in pixels. For each
        element, the distance to the nearest element in `catalogcoord` 
        is computed.

    catalogcoord : astropy Table
        Table with two columns for x and y positions in pixels. The 
        distance to the elements in this catalog are computed.

    Returns
    -------
    idx : ndarray
        For each entry in `matchcoord`, this array contains the index of
        the nearest neighbor in `catalogcoord`.

    sep : ndarray
        For each entry in `matchcoord`, this array contains the distance
        to the nearest neighbor in `catalogcoord`.
    '''
        
    if len(matchcoord.columns) !=2 or  len(catalogcoord.columns)!=2:
        raise ValueError('input tables must have exactly two columns')
    
    x_cat_name, y_cat_name = catalogcoord.columns
    
    idx = np.empty(len(matchcoord),dtype=int)   
    sep = np.empty(len(matchcoord),dtype=float)
    
    for i, row in enumerate(matchcoord):
        x,y = row
        sep_i = np.sqrt((x-catalogcoord[x_cat_name])**2+(y-catalogcoord[y_cat_name])**2)
        idx[i], sep[i] = np.argmin(sep_i), np.min(sep_i)        
        
    return idx, sep

def completeness_limit(
    self,
    line,
    StarFinder,
    threshold,
    stars_per_mag=10,
    iterations=1,
    oversize_PSF=1
    ): 
    '''determine completness limit 

    1. Insert mock sources of different brightness
    2. Run the source detection algorithm
    3. Compare mock sources to detected sources and determine
       the faintest sources that have been detected.

    Parameters
    ----------
    data : ndarray
        image with


    Returns
    -------
    Table
    '''

    data = getattr(self,line).copy()
    err  = getattr(self,f'{line}_err').copy()
    PSF  = getattr(self,'PSF')
    tshape = data.shape
    
    #----------------------------------------------------------------
    # craete mock data
    #----------------------------------------------------------------
    apparent_magnitude = np.arange(27,30,0.5)    
    n_sources = len(apparent_magnitude) * stars_per_mag
    
    for i in range(iterations):

        mock_sources = Table(data=np.zeros((n_sources,7)),names=['magnitude','flux','x_mean','y_mean','x_stddev','y_stddev','theta'])
        for i,m in enumerate(apparent_magnitude):
            mock_sources[i*stars_per_mag:(i+1)*stars_per_mag]['magnitude'] = m
        mock_sources['flux'] = 10**(-(mock_sources['magnitude']+13.74)/2.5) *1e20
        
        indices = np.empty((*data.shape,2),dtype=int) 
        indices[...,0] = np.arange(data.shape[0])[:,None]
        indices[...,1] = np.arange(data.shape[1])
        indices = indices[~np.isnan(data)]
        mock_sources['x_mean'], mock_sources['y_mean'] = np.transpose(indices[np.random.choice(len(indices),n_sources)])
        mock_sources['x_stddev'] = PSF[mock_sources['x_mean'],mock_sources['y_mean']] / (2*np.sqrt(2*np.log(2))) * oversize_PSF
        mock_sources['y_stddev'] = mock_sources['x_stddev']
        mock_sources['amplitude'] = mock_sources['flux'] / (mock_sources['x_stddev']*np.sqrt(2*np.pi))

        logger.info(f'{len(mock_sources)} mock sources created')

        mock_img = make_gaussian_sources_image(tshape,mock_sources)
        mock_img += data
        
        logger.info('mock sources inserted into image')
        
        #----------------------------------------------------------------
        # detection run
        #----------------------------------------------------------------
        for fwhm in np.unique(PSF[~np.isnan(PSF)]):
            mask = ~(PSF == fwhm)
            mean, median, std = sigma_clipped_stats(err[(~np.isnan(PSF)) & (~mask)], sigma=3.0)
            #print(f'mean={mean:.2f}, mediam={median:.2f}, std={std:.2f}')
            
            finder = StarFinder(fwhm      = fwhm*oversize_PSF, 
                                threshold = threshold*median,
                                sharplo   = 0.2, 
                                sharphi   = 0.8,
                                roundlo   = -0.5,
                                roundhi   = 0.5)
            peaks_part = finder(mock_img, mask=mask)
            if peaks_part:
                #logger.info(f'fwhm={fwhm:.3f}: {len(peaks_part)} sources found')
                if 'peak_tbl' in locals():
                    peaks_part['id'] += np.amax(peak_tbl['id'],initial=0)
                    peak_tbl = vstack([peak_tbl,peaks_part])
                else:
                    peak_tbl = peaks_part
            else:
                pass
                #logger.info(f'fwhm={fwhm:>7.3f}: no sources found')
        
        logger.info(f'{len(peak_tbl)} sources found')

        #----------------------------------------------------------------
        # compare detected sources to known mock stars
        #----------------------------------------------------------------
        logger.info(f'searching for best match')
        
        idx , sep = match_catalogues(mock_sources[['x_mean','y_mean']],peak_tbl[['xcentroid','ycentroid']])
        mock_sources['sep'] = sep
        mock_sources['peak'] = peak_tbl[idx]['peak']

        bins = np.arange(np.min(apparent_magnitude)-0.25,np.max(apparent_magnitude)+0.75,0.5)
        h,_ = np.histogram(mock_sources[mock_sources['sep']<0.5]['magnitude'],bins=bins)

        if 'hist' in locals():
            hist += h 
        else:
            hist = h

        del peak_tbl
    #----------------------------------------------------------------
    
    #----------------------------------------------------------------
    # create the histogram
    #----------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.bar(apparent_magnitude,hist/stars_per_mag*100/iterations,width=0.4)
    ax.set(xlabel='m$_{[\mathrm{OIII}]}$',
           ylabel='detected sources in %',
           ylim=[0,100])
    plt.savefig(basedir / 'reports' / 'figures' / f'{self.name}_completness.pdf')
    plt.show()
    #----------------------------------------------------------------

    for col in mock_sources.colnames:
        mock_sources[col].info.format = '%.8g' 

    return mock_sources