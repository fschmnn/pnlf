import logging
import warnings

from astropy.coordinates import SkyCoord
import matplotlib as mpl 
import matplotlib.pyplot as plt  
from astropy.visualization import simple_norm
from astropy.stats import sigma_clipped_stats
import numpy as np 

from astropy.nddata import Cutout2D
from photutils import CircularAperture

from ..constants import two_column,tab10
from ..auxiliary import annulus_mask, circular_mask
from .plot import create_RGB


logger = logging.getLogger(__name__)

def cutout_spectrum(position,img,data_cube,wcs,title=None,filename=None):
    '''Plot one spectra of a MUSE data cube with Image
    
    Parameters
    ----------
    
    position :
        Tuple of coordinates or SkyCoord object. Object at
        which the spectra is extracted.
        
    img : 
        2D Image that is displayed to illustrate the position
        of the spectra
        
    data_cube :
        3D data cube (spectra is first dimension) with the 
        same shape as img
        
    wcs : 
        World coordinate information for img and data_cube
        
    title : str (optional)
        Set title for plot
    '''
    
    
    if isinstance(position,SkyCoord):
        x,y = position.to_pixel(wcs=wcs)
    else:
        x,y = position
        
    # plot it
    fig = plt.figure(figsize=(two_column,two_column/3)) 
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1,2]) 
    ax1 = fig.add_subplot(gs[0],projection=wcs)

    norm = simple_norm(img,'linear',clip=False,percent=95)
    ax1.imshow(img, origin='lower',norm=norm,cmap='Greys')
    
    circle  = circular_mask(*data_cube.shape[1:],(x,y),4)
    annulus = annulus_mask(*data_cube.shape[1:],(x,y),8,12) 
    _, bkg, _ = sigma_clipped_stats(data_cube[...,annulus],axis=1)
    
    spectra = np.sum(data_cube[...,circle],axis=1)    
    # the background is the median * the number of non zero pixel
    spectra_without_bkg = spectra - bkg * np.sum(circle)

    #spectra = np.sum(data_cube[...,int(x)-1:int(x)+1,int(y)-1:int(y)+1],axis=(1,2))    
    # the wavelenght coverage of MUSE
    wavelength = np.linspace(4749.88,9349.88,data_cube.shape[0]) 
    
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(wavelength,spectra,color=tab10[1],label='with background')
    ax2.plot(wavelength,spectra_without_bkg,color=tab10[0],label='background subtracted')
    #ax2.legend() 

    ax1.set(title=title,
            xlabel='R.A. (J2000)',
            ylabel='Dec. (J2000)')
    
    ax2.set(title='Spectrum',
            yscale='linear',
            xlim=[4750,7000],
            #ylim=[1e2,7e2],
            xlabel=r'$\lambda$\,/\,\AA',
            ylabel=r'erg\,/\,s\,/\,\AA')
    
    ax2.yaxis.tick_right()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_label_position("right")
    
    ax1.add_patch(mpl.patches.Rectangle((x,y),1,1,linewidth=0.3,edgecolor='k',facecolor='none'))
    plt.subplots_adjust(wspace=0.1)
    
    x = int(x)
    y = int(y)
    con = mpl.patches.ConnectionPatch(xyA=((x+1)/img.shape[1],(y+1)/img.shape[0]), xyB=(0,1), 
                                      coordsA="axes fraction", coordsB="axes fraction",
                                      axesA=ax1, axesB=ax2, color="black",linewidth=0.3)
    ax2.add_artist(con)
    con = mpl.patches.ConnectionPatch(xyA=((x+1)/img.shape[1],(y)/img.shape[0]), xyB=(0,0), 
                                      coordsA="axes fraction", coordsB="axes fraction",
                                      axesA=ax1, axesB=ax2, color="black",linewidth=0.3)
    ax2.add_artist(con)
    
    if filename:
        plt.savefig(filename,bbox_inches='tight',dpi=800)

    plt.show()
    return spectra, wavelength



def spectrum_and_rgb(position,galaxy,data_cube,wcs,aperture_size,title=None,filename=None,xlim=[4750,7000]):
    '''Plot one spectra of a MUSE data cube with Image
    
    Parameters
    ----------
    
    position :
        Tuple of coordinates or SkyCoord object. Object at
        which the spectra is extracted.
        
    img : 
        2D Image that is displayed to illustrate the position
        of the spectra
        
    data_cube :
        3D data cube (spectra is first dimension) with the 
        same shape as img
        
    wcs : 
        World coordinate information for img and data_cube
        
    title : str (optional)
        Set title for plot
    '''
        
    if isinstance(position,SkyCoord):
        x,y = position.to_pixel(wcs=wcs)
    else:
        x,y = position
    
    logger.info(f'Pixel position: x={x:.1f}, y={y:.1f}')
    # plot it
    fig = plt.figure(figsize=(two_column,two_column/4)) 
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1,3]) 
    ax1 = fig.add_subplot(gs[0],projection=wcs)

    size = 40
    r = Cutout2D(galaxy.HA6562, (x,y), size,wcs=galaxy.wcs)
    g = Cutout2D(galaxy.OIII5006, (x,y), size,wcs=galaxy.wcs)
    b = Cutout2D(galaxy.SII6716, (x,y), size,wcs=galaxy.wcs)

    rgb = create_RGB(r.data,g.data,b.data,weights=[0.9,0.9,0.9],percentile=[97,97,97])
    ax1.imshow(rgb)

    radius = aperture_size * galaxy.PSF[int(y),int(x)] / 2
    aperture = CircularAperture(r.input_position_cutout,radius)
    aperture.plot(color='tab:orange',lw=1,axes=ax1)
    
    ax1.coords[0].set_ticks(number=2)
    ax1.coords[1].set_ticks(number=2)


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # there will be NaNs in the subcube that is used for the sigma clipping
        # astropy will issue a warning which we ignore in this enviornment
        circle  = circular_mask(*data_cube.shape[1:],(x,y),4)
        annulus = annulus_mask(*data_cube.shape[1:],(x,y),8,12) 
        _, bkg, _ = sigma_clipped_stats(data_cube[...,annulus],axis=1)
    
    spectra = np.sum(data_cube[...,circle],axis=1)    
    # the background is the median * the number of non zero pixel
    spectra_without_bkg = spectra - bkg * np.sum(circle)

    #spectra = np.sum(data_cube[...,int(x)-1:int(x)+1,int(y)-1:int(y)+1],axis=(1,2))    
    # the wavelenght coverage of MUSE
    wavelength = np.linspace(4749.88,9349.88,data_cube.shape[0]) 
    
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(wavelength,spectra,color=tab10[1],label='with background')
    ax2.plot(wavelength,spectra_without_bkg,color=tab10[0],label='background subtracted')
    #ax2.legend() 

    ax1.set(title=title,
            xlabel='R.A. (J2000)',
            ylabel='Dec. (J2000)')
    
    ax2.set(title='Spectrum',
            yscale='linear',
            xlim=xlim,
            #ylim=[1e2,7e2],
            xlabel=r'$\lambda$\,/\,\AA',
            ylabel=r'erg\,/\,s\,/\,\AA')
    
    ax2.yaxis.tick_right()
    ax2.yaxis.set_ticks_position('both')
    ax2.yaxis.set_label_position("right")
    plt.subplots_adjust(wspace=0)
    
    if filename:
        plt.savefig(filename,bbox_inches='tight',dpi=800)

    plt.show()
    return spectra, wavelength