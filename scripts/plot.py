'''

some plotting routines

'''

import sys
from pathlib import Path
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


logging.basicConfig(stream=sys.stdout,
                    #format='(levelname)s %(name)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



from pymuse.io import ReadLineMaps

data_raw = Path('d:\downloads\MUSEDAP')
NGC628 = ReadLineMaps(data_raw / 'NGC628')


from pymuse.plot import create_RGB


'''
Plot a single galaxy as an RGB image where the channels correspond
to different lines
''' 

# ====== define input parameters =============================
rgb = create_RGB(NGC628.SII6716,NGC628.HA6562,NGC628.OIII5006)
labels=['SII6716','HA6562','OIII5006']
wcs=NGC628.wcs
# ============================================================



# create an empty figure with correct projection
fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection':wcs})

# plot the image
plt.imshow(rgb,origin='lower')

# create a legend
if labels:
    # first we create a legend with three invisible handles
    handles = 3*[mpl.patches.Rectangle((0, 0), 0, 0, alpha=0.0)]
    leg = ax.legend(handles,labels, frameon=False,handlelength=0,prop={'size': 16})

    # next we set the color of the three labels
    for color,text in zip(['red','green','blue'],leg.get_texts()):
        text.set_color(color)

plt.show()
#plt.savefig(basedir / 'reports' / 'figures' / 'NGC628_rgb.pdf')


'''
Plot all galaxies as an RGB image where the channels correspond
to different lines
''' 

ncols = 3
nrows = int(np.ceil(len(galaxies[:-1])/ncols))

fig = plt.figure(figsize=(15,15*nrows/ncols))

for i, name in enumerate(galaxies[:-1]):
    galaxy = ReadLineMaps(data_raw / name)
    rgb = create_RGB(galaxy.SII6716,galaxy.HA6562,galaxy.OIII5006)

    ax = fig.add_subplot(nrows,ncols,i+1,projection=galaxy.wcs)
    ax.imshow(rgb,origin='lower')
    ax.set_title(name)
    
plt.savefig(basedir / 'reports' / 'figures' / 'all_objects_rgb.pdf')