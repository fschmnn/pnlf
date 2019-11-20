import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.table import Table

import astropy.units as u

from astropy.visualization import simple_norm
from astropy.nddata import Cutout2D
import random

from photutils import CircularAperture         # define circular aperture

def plot_sources(data,wcs,positions,references=None,filename=None):
    '''plot line map with detected sources'''

    apertures = []
    if isinstance(positions,tuple):
        for position in positions:
            apertures.append(CircularAperture(position, r=4))
    else:
        apertures.append(CircularAperture(positions, r=4))

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection=wcs)
    vmax = np.nanpercentile(data,95)
    #norm = simple_norm(data[~np.isnan(data)], 'log',max_percent=99.99)

    plt.imshow(data, 
               origin='lower',
               cmap=plt.cm.Blues_r, 
               #norm=norm,
               interpolation='none',
               vmax=vmax
              )

    colors = ['red','yellow','orange','green']
    for i,aperture in enumerate(apertures):
        aperture.plot(color=colors[i],lw=.6, alpha=0.5)

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    if filename:
        plt.savefig(filename)




def plot_detected_stars(self,nrows=10,ncols=10,filename=None):
    '''create cutouts of the detected sources and plot them
    
    
    '''
    
    # the data we need
    peaks_tbl = self.peaks_tbl
    data      = self.OIII5006
    wcs       = self.wcs
    
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
        
        norm = simple_norm(stars[i].data, 'log', percent=99.)
        ax.imshow(stars[i].data, norm=norm, origin='lower', cmap='viridis')

        
    if filename:
        plt.savefig(filename)

 
def plot_single_stars(self,extension,x,y):
    '''create cutouts of a single sources and plot it'''
    
    data = getattr(self,extension)
    wcs  = self.wcs
    
    # exclude stars that are too close to the border
    size = 32

    # defien the size of the cutout region
    size = u.Quantity((size, size), u.pixel)
    star = Cutout2D(data, (x,y), size,wcs=wcs)
    
    #fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),squeeze=True)
    #ax = ax.ravel()
    
    fig = plt.figure(figsize=(6,6))
    
    # get an index idx from 0 to nrows*ncols and a random index i from 0 to len(stars)
    ax = fig.add_subplot(1,1,1,projection=star.wcs)
    norm = simple_norm(star.data, 'log', percent=99.)
    ax.imshow(star.data, norm=norm, origin='lower', cmap='viridis')