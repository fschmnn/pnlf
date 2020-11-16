import numpy as np
import matplotlib.pyplot as plt
import astropy 
from astropy.visualization import simple_norm
import astropy.units as u
from astropy.coordinates import SkyCoord

from ..constants import single_column, two_column

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


def quick_plot(data,wcs=None,figsize=(two_column,two_column),cmap=plt.cm.hot,filename=None,**kwargs):
    '''create a quick plot 

    uses norm     
    '''
    
    fig = plt.figure(figsize=figsize)
    
    if isinstance(data,astropy.nddata.nddata.NDData):
        ax = fig.add_subplot(projection=data.wcs)
        img = data.data
    if isinstance(data,astropy.nddata.utils.Cutout2D):
        ax = fig.add_subplot(projection=data.wcs)
        img = data.data
    elif wcs:
        ax = fig.add_subplot(projection=wcs)
        img = data
    else:
        ax = fig.add_subplot()
        img = data
        
    norm = simple_norm(img,clip=False,percent=99)
    ax.imshow(img,norm=norm,cmap=cmap)
    ax.set(**kwargs)
    
    if filename:
        plt.savefig(filename,dpi=600)
    
    #plt.show()

    return ax

def add_scale(ax,length,label=None,color='black',fontsize=10):
    '''add a scale to a plot
    
    The scale if calculated from the wcs information of the plot
    '''
    
    if not hasattr(ax,'wcs'):
        raise AttributeError('axis is missing wcs information')
        
    wcs = ax.wcs
    
    w,h = wcs._naxis
    x,y = 0.05*w, 0.05*h
    start = SkyCoord.from_pixel(0,y,wcs)
    end   = SkyCoord.from_pixel(w,y,wcs)
    scale = length / start.separation(end)
    
    ax.plot([x,x+w*scale],[y,y],color=color,marker='|')
    
    if label:
        ax.text(x+0.5*w*scale,y*1.2,label,horizontalalignment='center',color=color,fontsize=fontsize)
    else:
        ax.text(x+0.5*w*scale,y*1.2,length,horizontalalignment='center',color=color,fontsize=fontsize)

    return w*scale