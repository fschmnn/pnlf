from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show

import astropy
from astropy.table import Table

import astropy.units as u
from astropy.visualization import simple_norm

from astropy.nddata import Cutout2D
import random
from scipy.stats import spearmanr

from photutils import CircularAperture         # define circular aperture

from ..constants import single_column, two_column
from ..analyse import PNLF

def figsize(scale=1):
    '''Create nicely proportioned figure

    This function calculates the optimal figuresize for any given scale
    (the ratio between figuresize and textwidth. A figure with scale 1
    covers the entire writing area). Therefor it is important to know 
    the textwidth of your target document. This can be obtained by using
    the command "\the\textwidth" somewhere inside your document.
    '''

    # for one column: 504.0p
    width_pt  = 240                           # textwidth from latex
    in_per_pt = 1.0/72.27                     # Convert pt to inch
    golden    = 1.61803398875                 # Aesthetic ratio 
    width  = width_pt * in_per_pt * scale     # width in inches
    height = width / golden                   # height in inches
    return [width,height]

def plot_sky_with_detected_stars(data,wcs,positions,filename=None):
    '''plot line map with detected sources
    
    Parameters
    ----------

    data : 2d array
        numpy array that contains the image data

    wcs : 
        wcs information for the projection

    positions : array or tuple
        (n,2) shaped array with positions. Can also be a tuple of
        multiple such arrays.

    filename : Path
        if given, a PDF of the plot is saved to filename
    '''

    apertures = []
    if isinstance(positions,tuple) or isinstance(positions,list):
        for position in positions:
            apertures.append(CircularAperture(position, r=8))
    else:
        apertures.append(CircularAperture(positions, r=8))

    fig = figure(figsize=(6.974,6.974/2))
    ax = fig.add_subplot(111, projection=wcs)
    norm = simple_norm(data,'asinh',clip=False,min_percent=1,max_percent=99.5)
    cmap = plt.cm.hot
    cmap.set_bad('w')

    plt.imshow(data, 
               origin='lower',
               cmap=cmap, 
               norm=norm,
               #interpolation='none',
               #vmax=vmax
              )

    for aperture in apertures:
        aperture.plot(color='blue',lw=.5, alpha=1)

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    if filename:
        if not isinstance(filename,Path):
            filename = Path(filename)

        plt.savefig(filename,dpi=600)


def sample_cutouts(data,peaks_tbl,wcs,nrows=10,ncols=10,filename=None):
    '''create cutouts of the detected sources and plot them
    
    
    '''
    
    # exclude stars that are too close to the border
    size = 16
    hsize = (size - 1) / 2
    x = peaks_tbl['x']  
    y = peaks_tbl['y']  
    mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
           (y > hsize) & (y < (data.shape[0] -1 - hsize)))  

    stars_tbl = Table()
    stars_tbl['x'] = x[mask]  
    stars_tbl['y'] = y[mask]  

    # extract_stars does not include wcs information
    #nddata = NDData(data=data,wcs=self.wcs)  
    #stars = extract_stars(nddata, stars_tbl, size=size)  

    # defien the size of the cutout region
    size = u.Quantity((size, size), u.pixel)

    stars = []
    for row in stars_tbl:
        stars.append(Cutout2D(data, (row['x'],row['y']), size,wcs=wcs))
    
    #fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),squeeze=True)
    #ax = ax.ravel()
    
    fig = figure(figsize=(100,100))
    
    # get an index idx from 0 to nrows*ncols and a random index i from 0 to len(stars)
    for idx,i in enumerate(random.sample(range(len(stars)), nrows*ncols)):
        ax = fig.add_subplot(nrows,ncols,idx+1,projection=stars[i].wcs)

        if np.any(np.isnan(stars[i].data)):
            print('this should not be')

        norm = simple_norm(stars[i].data, 'log', percent=99.)
        ax.imshow(stars[i].data, norm=norm, origin='lower', cmap='Blues_r')

    if filename:
        if not isinstance(filename,Path):
            filename = Path(filename)
            
        plt.savefig(filename)

def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile
 
def single_cutout(self,x,y,size=32,aperture_size=None,percentile=99,extension='OIII5006'):
    '''create cutouts of a single sources and plot it'''

    
    data = getattr(self,extension)
    wcs  = self.wcs
    
    # defien the size of the cutout region
    star = Cutout2D(data, (x,y), u.Quantity((size, size), u.pixel),wcs=wcs)

    profile = radial_profile(star.data,star.input_position_cutout)
    #fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),squeeze=True)
    #ax = ax.ravel()    labels=['SII6716','HA6562','OIII5006'
    r = Cutout2D(getattr(self,'SII6716'), (x,y), size,wcs=wcs).data
    g = Cutout2D(getattr(self,'HA6562'), (x,y), size,wcs=wcs).data
    b = Cutout2D(getattr(self,'OIII5006'), (x,y), size,wcs=wcs).data

    #rgb = create_RGB(r,g,b,percentile=99)

    rgb = create_RGB(self.HA6562,self.OIII5006,self.SII6716,percentile=percentile)
    #rgb = Cutout2D(rgb,(x,y),size,wcs=wcs).data

    fig = figure(figsize=(two_column,two_column/3))
    
    # get an index idx from 0 to nrows*ncols and a random index i from 0 to len(stars)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)

    norm = simple_norm(data,percent=99,clip=False)#, percent=99.)
    yslice = slice(int(x-size/2),int(x+size/2))
    xslice = slice(int(y-size/2),int(y+size/2))
    im1 = ax1.imshow(data[xslice,yslice], norm=norm, origin='lower', cmap='Greens')
    im2 = ax2.imshow(rgb[xslice,yslice,:],origin='lower')

    if aperture_size:
        aperture = CircularAperture((size/2+(x-int(x)),size/2+(y-int(y))),aperture_size)
        aperture.plot(color='black',lw=0.8,axes=ax1)
        aperture.plot(color='black',lw=0.8,axes=ax2)
        ax3.axvline(aperture_size,color='black',lw=0.8)
        fwhm = aperture_size/2.5*2
        ax3.axvline(2*fwhm,color='gray',lw=0.8)
        ax3.axvline(np.sqrt((4*fwhm)**2+(1.25*fwhm)**2),color='gray',lw=0.8)
        

    #ax1.set_xlim([x-size/2,x+size/2])
    #ax1.set_ylim([y-size/2,y+size/2])
    #ax2.set_xlim([x-size/2,x+size/2])
    #ax2.set_ylim([y-size/2,y+size/2])

    #im1 = ax1.imshow(star.data, norm=norm, origin='lower', cmap='Blues_r')
    #im2 = ax2.imshow(rgb,origin='lower')
    #fig.colorbar(im,ax=ax1)

    ax3.plot(profile)
    #ax3.set_xlabel(r'radius in px')
    #cor= spearmanr(np.arange(0,10,1),profile[:10]).correlation
    #ax3.set_title(f'rho = {cor:.2f}')
    ax2.set_yticks([])

    return ax1,ax2,ax3

from astropy.visualization import AsymmetricPercentileInterval

def create_RGB(r,g,b,stretch='linear',weights=None,percentile=95):
    '''combie three arrays to one RGB image
    
    Parameters
    ----------
    r : ndarray
        (n,m) array that is used for the red channel
        
    g : ndarray
        (n,m) array that is used for the green channel
        
    b : ndarray
        (n,m) array that is used for the blue channel
    
    percentile : float
        percentile that is used for the normalization
        
    Returns
    -------
    rgb : ndarray
        (n,m,3) array that is normalized to 1
    '''

    if not r.shape == g.shape == b.shape:
        raise ValueError('input arrays must have the dimensions')
    
    # create an empty array with the correct size
    rgb = np.empty((*r.shape,3))
    
    if type(percentile)==float or type(percentile)==int:
        percentile = 3*[percentile]

    # assign the input arrays to the 3 channels and normalize them to 1
    rgb[...,0] = r / np.nanpercentile(r,percentile[0])
    rgb[...,1] = g / np.nanpercentile(g,percentile[1])
    rgb[...,2] = b / np.nanpercentile(b,percentile[2])

    if weights:
        rgb[...,0] *= weights[0]
        rgb[...,1] *= weights[1]
        rgb[...,2] *= weights[2]

    #rgb /= np.nanpercentile(rgb,percentile)
    
    # clip values (we use percentile for the normalization) and fill nan
    rgb = np.clip(np.nan_to_num(rgb,nan=1),a_min=0,a_max=1)
    
    return rgb
    

def quick_plot(data,wcs=None,filename=None,**kwargs):
    '''create a quick plot 

    uses norm     
    '''
    
    fig = plt.figure(figsize=(two_column,two_column))
    
    if isinstance(data,astropy.nddata.nddata.NDData):
        ax = fig.add_subplot(projection=data.wcs)
        img = data.data
    elif wcs:
        ax = fig.add_subplot(projection=wcs)
        img = data
    else:
        ax = fig.add_subplot()
        img = data
        
    norm = simple_norm(img,clip=False,percent=99)
    ax.imshow(img,norm=norm,cmap=plt.cm.hot)
    ax.set(**kwargs)
    
    if filename:
        plt.savefig(filename,dpi=600)
    plt.show()
