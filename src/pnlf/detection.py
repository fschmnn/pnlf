import logging              # use instead of print for more control
import inspect              # get signature of functions (e.g. to pass kwargs)
from pathlib import Path    # filesystem related stuff
import numpy as np          # numerical computations
import re

import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table                # useful data structure
from astropy.table import vstack               # combine multiple tables

from astropy.stats import sigma_clipped_stats  # calcualte statistics of images
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm

from photutils import DAOStarFinder            # DAOFIND routine to detect sources
from photutils import IRAFStarFinder           # IRAF starfind routine to detect stars

from collections import OrderedDict                           # make random table reproducable
from photutils.datasets import make_gaussian_sources_image    # create table with mock sources
from photutils import make_source_mask, CircularAperture

#from astropy.convolution import convolve, Gaussian2DKernel

from .io import ReadLineMaps
from .auxiliary import correct_PSF
from .analyse import sample_pnlf

from .constants import tab10, single_column, two_column

basedir = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)

# IRAFStarFinder complains about NaN values in lt and gt
np.warnings.filterwarnings('ignore')

def detect_unresolved_sources(
    self : ReadLineMaps,
    line : list,
    StarFinder,
    threshold : float=3.,
    oversize: float=1.,
    exclude_region=None,
    save=False,
    **kwargs
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

    oversize :
        increase the size of the PSF (becaues it is a Moffat)

    save : bool
        save the result is to a file in `reports/catalogues/`

    kwargs : dict
        other parameters are passed to StarFinder
    '''
    
    if not isinstance(self,ReadLineMaps):
        logger.warning('input should be of type ReadLineMaps')
    
    daoargs = {k:kwargs.pop(k) for k in dict(kwargs) if k in inspect.signature(DAOStarFinder).parameters.keys()}

    for k,v in kwargs.items():
        logger.warning(f'unused kwargs: {k}={v}')

    # for convenience only, to make accessing the data easier
    data = getattr(self,line)
    err  = getattr(self,f'{line}_err')
    PSF  = getattr(self,'PSF') 

    if not np.any(exclude_region):
        exclude_region = np.zeros(data.shape,dtype=bool)
    else:
        logger.info(f'masking {np.sum(exclude_region)/np.prod(exclude_region.shape)*100:.2f} % of the image')

    logger.info(f'searching for sources in {self.name} with [{line}] line map (using ' + \
          str(StarFinder).split('.')[-1][:-2] + ')\n' )
    logger.info(','.join([f'{k}: {v} ' for k,v in daoargs.items()])) 
    #mean, median, std = sigma_clipped_stats(data[~np.isnan(PSF)], sigma=3.,maxiters=None)

    try:
        wavelength =int(re.findall(r'\d{4}', line)[0])
        PSF_correction = correct_PSF(wavelength)
    except:
        PSF_correction = 0

    # loop over all pointings with different PSFs
    pointings = np.unique(PSF[~np.isnan(PSF)])
    logger.info(f'searching for sources in {len(pointings)} pointings')
    
    # header for the print information
    logger.info(f'{"fwhm":>9}{"#N":>5}{"mean":>8}{"median":>8}{"std":>8}')

    for fwhm in pointings:
                
        # we create a mask for the current pointing (must be inverted)
        psf_mask = (PSF == fwhm) #& (~np.isnan(PSF))
        #source_mask = make_source_mask(data, nsigma=2, npixels=5, dilate_size=int(3*fwhm)) | ~psf_mask
        mean, median, std = sigma_clipped_stats(data, sigma=3.0,maxiters=5,mask=~psf_mask)
        
        # initialize and run StarFinder (DAOPHOT or IRAF)
        finder = StarFinder(fwhm      = (fwhm - PSF_correction) * oversize, 
                            threshold = threshold*std,
                            **daoargs)
        peaks_part = finder(data-median, mask=(~psf_mask | exclude_region))
        
        if not peaks_part:
            logger.warning('no sources found in pointing')
            continue
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
    #setattr(self,'peaks_tbl',peak_tbl)

    # we save the found positions to a file
    if save:
        filename = basedir / 'data' / 'interim' / f'peaks_{self.name}.txt'
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


def plot_completeness_limit(mock_sources,max_sep,limit,filename=None):

    hist = []
    width = 0.5
    bins = np.arange(26,30,width)

    for center in bins:
        tmp = mock_sources[(mock_sources['magnitude']>center-width/2) & (mock_sources['magnitude']<=center+width/2)]
        if len(tmp)>0:
            hist.append(np.sum(tmp['sep']<max_sep)/len(tmp))
        else:
            hist.append(0)
        #print(f'{center}: {len(tmp)} objects')
    hist = np.array(hist)
    
    completeness_limit = np.max(bins[hist>=limit])
    logger.info(f'completeness limit = {completeness_limit} (above {limit*100}%)')
    
    fig, ax = plt.subplots(figsize=(single_column,single_column/1.618))
    ax.axhline(100*limit,color='black',lw=0.6)
    
    ax.bar(bins[hist>=limit],hist[hist>=limit]*100,width=width*0.9,color=tab10[0])
    ax.bar(bins[hist<limit],hist[hist<limit]*100,width=width*0.9,fc='white',ec=tab10[0])
    
    for b,h in zip(bins,hist):
        if h>=0.8:
            ax.text(b,5,f'{h*100:.0f}\%',horizontalalignment='center',color='white',fontsize=6)
        else:
            ax.text(b,5,f'{h*100:.0f}\%',horizontalalignment='center',color=tab10[0],fontsize=6)
            
    ax.set(xlabel='m$_{[\mathrm{OIII}]}$',
           ylabel='percentage of recovered objects',
           ylim=[0,100])
    #ax.set_xticks(bins)
    #ax.set_title(f'completeness limit = {completeness_limit} (above {limit*100:.0f}\%)')
    
    if filename:
        plt.savefig(filename)
    plt.show()

def completeness_limit(
    self,
    line,
    StarFinder,
    threshold,
    distance_modulus,
    limit = 0.8,
    max_sep = 0.5,
    oversize=1.,
    exclude_region=None,
    iterations=1,
    n_sources = 500,
    plot=False,
    **kwargs
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

    daoargs = {k:kwargs.pop(k) for k in dict(kwargs) if k in inspect.signature(DAOStarFinder).parameters.keys()}

    for k,v in kwargs.items():
        logger.warning(f'unused kwargs: {k}={v}')

    data = getattr(self,line).copy()
    err  = getattr(self,f'{line}_err').copy()
    PSF  = getattr(self,'PSF')
    tshape = data.shape

    if not np.any(exclude_region):
        exclude_region = np.zeros(data.shape,dtype=bool)
    else:
        print(f'masking {np.sum(exclude_region)/np.prod(exclude_region.shape)*100:.2f} % of the image')

    #----------------------------------------------------------------
    # craete mock data
    #----------------------------------------------------------------
    
    try:
        wavelength = int(re.findall(r'\d{4}', line)[0])
        PSF_correction = correct_PSF(wavelength)
    except:
        PSF_correction = 0

    j = 0
    while j < iterations:
    #for i in range(iterations):
        
        logger.info(f'iteration {j+1} of {iterations}')

        mock_sources = Table(data=np.zeros((n_sources,7)),names=['magnitude','flux','x_mean','y_mean','x_stddev','y_stddev','theta'])
        mock_sources['magnitude'] = sample_pnlf(n_sources,distance_modulus,29.75)
        mock_sources['flux'] = 10**(-(mock_sources['magnitude']+13.74)/2.5) *1e20
        
        # create a number of random points (more than we need because
        # some will fall in unobserved areas)
        f = 1.5 # create more points than we need 
        while True:
            # number we create is f * number of sources we want / 
            # (observed area / total area ) 
            N = f * n_sources / (np.sum(~np.isnan(PSF)) / np.prod(self.shape))
            indices = np.random.uniform((0,0),self.shape,(int(N),2))
            x_mean = indices[:,1]
            y_mean = indices[:,0]
            PSF_arr = np.array([PSF[int(y),int(x)] for x,y in zip(x_mean,y_mean)])
            in_frame = ~np.isnan(PSF_arr)

            x_mean  = x_mean[in_frame]
            y_mean  = y_mean[in_frame]
            PSF_arr = PSF_arr[in_frame]
            
            if len(x_mean) > n_sources:
                x_mean = x_mean[:n_sources]
                y_mean = y_mean[:n_sources]
                PSF_arr = PSF_arr[:n_sources]
                break
            else:
                # it might happen that more points fall in unobserved
                # areas. In this case we have to repeat
                f *=1.1
        
        mock_sources['x_mean'], mock_sources['y_mean'] = x_mean, y_mean
        # get PSF size at the generated position
        mock_sources['x_stddev'] = (PSF_arr-PSF_correction) * gaussian_fwhm_to_sigma * oversize
        mock_sources['y_stddev'] = mock_sources['x_stddev']
        mock_sources['amplitude'] = mock_sources['flux'] / (mock_sources['x_stddev']*np.sqrt(2*np.pi))

        logger.info(f'{len(mock_sources)} mock sources created')

        if plot:
            fig = plt.figure(figsize=(6,6))
            ax  = fig.add_subplot(111,projection=self.wcs)

            norm = simple_norm(data,'linear',clip=False,max_percent=95)
            ax.imshow(data,norm=norm,cmap=plt.cm.Blues_r,origin='lower')
            len(mock_sources)
            positions = np.transpose([mock_sources['x_mean'],mock_sources['y_mean']])
            apertures = CircularAperture(positions, r=6)
            ax.scatter(mock_sources['x_mean'],mock_sources['y_mean'],color='tab:red')
            #apertures.plot(color='tab:red',lw=.2, alpha=1,ax=ax)
            plt.show()

        mock_img = make_gaussian_sources_image(tshape,mock_sources)
        mock_img += data

        logger.info('mock sources inserted into image')
        
        #----------------------------------------------------------------
        # detection run
        #----------------------------------------------------------------
        for fwhm in np.unique(PSF[~np.isnan(PSF)]):
            # we create a mask for the current pointing (must be inverted)
            psf_mask = (PSF == fwhm) #& (~np.isnan(PSF))
            source_mask = make_source_mask(data, nsigma=2, npixels=5, dilate_size=int(3*fwhm)) | ~psf_mask
            mean, median, std = sigma_clipped_stats(data, sigma=3.0,maxiters=5,mask=source_mask)
            
            # initialize and run StarFinder (DAOPHOT or IRAF)
            finder = StarFinder(fwhm      = (fwhm - PSF_correction) * oversize, 
                                threshold = threshold*std,
                                **daoargs)
            peaks_part = finder(mock_img, mask=(~psf_mask | exclude_region))

            if peaks_part:
                #logger.info(f'fwhm={fwhm:.3f}: {len(peaks_part)} sources found')
                if 'peak_tbl' in locals():
                    peaks_part['id'] += np.amax(peak_tbl['id'],initial=0)
                    peak_tbl = vstack([peak_tbl,peaks_part])
                else:
                    peak_tbl = peaks_part
            else:
                logger.info(f'fwhm={fwhm:>7.3f}: no sources found')

        if 'peak_tbl' in locals():
            logger.info(f'{len(peak_tbl)} sources found')
        else:
            logger.warning('no sources found')
            continue

        #----------------------------------------------------------------
        # compare detected sources to known mock stars
        #----------------------------------------------------------------
        logger.info(f'compare detected sources to injected sources')

        idx , sep = match_catalogues(mock_sources[['x_mean','y_mean']],peak_tbl[['xcentroid','ycentroid']])
        mock_sources['sep'] = sep
        mock_sources['peak'] = peak_tbl[idx]['peak']
        
        if 'out' in locals():
            out = vstack([out,mock_sources])
        else:
            out = mock_sources

        del peak_tbl
        j+= 1

    #----------------------------------------------------------------
    # create the histogram
    #----------------------------------------------------------------
    
    filename = basedir / 'reports' / self.name / f'{self.name}_completness.pdf'
    plot_completeness_limit(out,max_sep=max_sep,limit=limit,filename=filename)

    #----------------------------------------------------------------

    for col in out.colnames:
        out[col].info.format = '%.8g' 

    return out


def guess_coordinate_column(columns,c1='x',c2='y'):
    '''Guess coordinate coordinates from list of columns

    Coordinate columns in table are usually named similar, with the 
    difference being `x` and `y` or `1` and `2`. This function looks
    for a pair of columns that differ only by the given character
    c1 and c2.
    
    e.g. if two columns are named `x_cen` and `y_cen` they are identified.
    
    Parameters
    ----------
    
    columns : list
        A list of column names.
    c1,c2 : str
        Difference between the two coordinate columns
    
    
    Returns 
    -------
    
    (column1,column2) : str
    '''
    
    matches = []
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:]):
            
            if col1.replace(c1,'') == col2.replace(c2,''):
                matches.append((col1,col2))
    
    if len(matches) <1:
        logger.warning('no match found')
    elif len(matches) >1:
        logger.warning('more than one possible match found')
        
    return matches
