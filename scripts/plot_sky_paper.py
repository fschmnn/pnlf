import logging 
from pathlib import Path
import sys

logging.basicConfig(stream=sys.stdout,
                    #format='(levelname)s %(name)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

import numpy as np 
import matplotlib as mpl

from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt

plt.style.use('TeX.mplstyle')

#rc('font',**{'family':'serif','serif':['Times'],'size':10})


from astropy.io import ascii 
from photutils import CircularAperture         # define circular aperture
from astropy.visualization import simple_norm

from pymuse.io import ReadLineMaps

basedir = Path('..')
data_raw = Path('d:\downloads\MUSEDAP')



name = 'NGC628'
filename = basedir / 'reports' / 'catalogues' / f'pn_candidates_{name}.txt'
tbl = ascii.read(str(filename),format='fixed_width',delimiter='\t')

galaxy = ReadLineMaps(data_raw / name)

completeness = 29.
mu = 29.966 

criteria = ((tbl['type']=='PN'))
data = tbl[np.where(criteria & (tbl['mOIII']<completeness))]['mOIII']


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
            apertures.append(CircularAperture(position, r=5))
    else:
        apertures.append(CircularAperture(positions, r=5))

    #fig = figure(figsize=(6.974,6.974))
    fig = figure(figsize=(3.321,3.321))

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

    colors = ['tab:red','yellow','orange','green']
    for i,aperture in enumerate(apertures):
        aperture.plot(color=colors[i],lw=.8, alpha=1)

    ax.set_xlabel(r'RA (J2000)')
    ax.set_ylabel(r'Dec (J2000)')

    plt.tight_layout()

    if filename:
        if not isinstance(filename,Path):
            filename = Path(filename)
        savefig(filename,bbox_inches='tight')


tmp = tbl[np.where(criteria & (tbl['mOIII']<28.5))]
pos = np.transpose((tmp['x'],tmp['y']))

save_file = Path.cwd() / '..' / 'reports' / f'{galaxy.name}_map_PN.pdf'
plot_sky_with_detected_stars(data=galaxy.OIII5006_DAP,
                             wcs=galaxy.wcs,
                             positions=pos,
                             filename=save_file)