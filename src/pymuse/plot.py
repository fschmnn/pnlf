from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table

import astropy.units as u
from astropy.visualization import simple_norm

from astropy.nddata import Cutout2D
import random

from photutils import CircularAperture         # define circular aperture

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
            apertures.append(CircularAperture(position, r=4))
    else:
        apertures.append(CircularAperture(positions, r=4))

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection=wcs)
    vmax = np.nanpercentile(data,95)
    norm = simple_norm(data[~np.isnan(data)], 'log',max_percent=95.,clip=False)
    cmap = plt.cm.Blues_r
    cmap.set_bad('w')

    plt.imshow(data, 
               origin='lower',
               cmap=cmap, 
               #norm=norm,
               #interpolation='none',
               vmax=vmax
              )

    colors = ['red','yellow','orange','green']
    for i,aperture in enumerate(apertures):
        aperture.plot(color=colors[i],lw=.6, alpha=0.9)

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    if filename:
        if not isinstance(filename,Path):
            filename = Path(filename)

        plt.savefig(filename)


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
    
    fig = plt.figure(figsize=(100,100))
    
    # get an index idx from 0 to nrows*ncols and a random index i from 0 to len(stars)
    for idx,i in enumerate(random.sample(range(len(stars)), nrows*ncols)):
        ax = fig.add_subplot(nrows,ncols,idx+1,projection=stars[i].wcs)

        if np.any(np.isnan(stars[i].data)):
            print('this should not be')

        norm = simple_norm(stars[i].data, 'log', percent=99.)
        ax.imshow(stars[i].data, norm=norm, origin='lower', cmap='viridis')

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
 
def single_cutout(self,extension,x,y):
    '''create cutouts of a single sources and plot it'''
    
    data = getattr(self,extension)
    wcs  = self.wcs
    
    # exclude stars that are too close to the border
    size = 32

    # defien the size of the cutout region
    size = u.Quantity((size, size), u.pixel)
    star = Cutout2D(data, (x,y), size,wcs=wcs)
    
    profile = radial_profile(star.data,star.input_position_cutout)
    #fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),squeeze=True)
    #ax = ax.ravel()
    
    fig = plt.figure(figsize=(10,5))
    
    # get an index idx from 0 to nrows*ncols and a random index i from 0 to len(stars)
    ax1 = fig.add_subplot(1,2,1,projection=star.wcs)
    ax2 = fig.add_subplot(1,2,2)
    
    norm = simple_norm(star.data, 'log')#, percent=99.)
    im = ax1.imshow(star.data, norm=norm, origin='lower', cmap='viridis')
    fig.colorbar(im,ax=ax1)

    ax2.plot(profile)
    ax2.set_xlabel('radius in px')


def create_RGB(r,g,b,percentile=90):
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
    
    # create an empty array with teh correct size
    rgb = np.empty((*r.shape,3))
    
    # assign the input arrays to the 3 channels and normalize them to 1
    rgb[...,0] = r / np.nanpercentile(r,percentile)
    rgb[...,1] = g / np.nanpercentile(g,percentile)
    rgb[...,2] = b / np.nanpercentile(b,percentile)
    
    # clip values (we use percentile for the normalization) and fill nan
    rgb = np.clip(np.nan_to_num(rgb,nan=1),a_min=0,a_max=1)
    
    return rgb
    

    
