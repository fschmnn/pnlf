from pathlib import Path
import numpy as np

import matplotlib as mpl
from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt

from photutils import CircularAperture
from astropy.visualization import simple_norm

def classification_map(galaxy,parameters,tbl,filename):
    '''
    Plot data in galaxy for different lines and overplot the position
    of detected sources
    '''

    if "zoomin" in parameters[galaxy.name]:
        x1,x2,y1,y2 = parameters[galaxy.name]['zoomin']
    else:
        x1,x2,y1,y2 = 200,600,200,600

    # ====== define input parameters =============================
    labels=['SII6716','HA6562','OIII5006']
    wcs=galaxy.wcs
    # ============================================================

    from pymuse.plot.plot import create_RGB
    rgb = create_RGB(galaxy.SII6716,galaxy.HA6562,galaxy.OIII5006,percentile=96)

    table = tbl[tbl['mOIII']<galaxy.completeness_limit]

    fig = plt.figure(figsize=(6.974,6.974/2))
    ax1 = fig.add_subplot(131,projection=wcs)
    ax2 = fig.add_subplot(132,projection=wcs,sharey=ax1)
    ax3 = fig.add_subplot(133)

    #fig, (ax2,ax3,ax1) = plt.subplots(ncols=3,figsize=(6.974,6.974/2),subplot_kw={'projection':wcs})


    norm = simple_norm(galaxy.OIII5006_DAP,'linear',clip=False,max_percent=95)
    ax1.imshow(galaxy.OIII5006_DAP,norm=norm,cmap=plt.cm.Blues_r)

    norm = simple_norm(galaxy.HA6562,'linear',clip=False,max_percent=95)
    ax2.imshow(galaxy.HA6562,norm=norm,cmap=plt.cm.Greens_r)

    ax3.imshow(rgb)

    print(f'{len(table)} sources')
    for t,c in zip(['HII','SNR','PN'],['black','red','white']):
        
        sub = table[table['type']==t]
        print(f'{t:<3}: {len(sub):>3}')
        positions = np.transpose([sub['x'],sub['y']])
        apertures = CircularAperture(positions, r=6)
        apertures.plot(color=c,lw=.2, alpha=1,ax=ax1)
        apertures.plot(color=c,lw=.2, alpha=1,ax=ax2)
        apertures.plot(color=c,lw=.2, alpha=1,ax=ax3)

    for row in table:
        txt,x,y = row['id'], row['x']+5, row['y']
        
        if x1<x<x2 and y1<y<y2:
            #ax1.annotate(txt, (x, y),fontsize=0.5)
            #ax2.annotate(txt, (x, y),fontsize=0.5)
            ax3.annotate(txt, (x, y),fontsize=2,color='white')


    ax1.add_patch(mpl.patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=0.2,edgecolor='k',facecolor='none'))
    ax2.add_patch(mpl.patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=0.2,edgecolor='k',facecolor='none'))
    ax3.set_xlim([x1,x2])
    ax3.set_ylim([y1,y2])

    ax1.set_title(r'O[III]')
    ax2.set_title(r'$\mathrm{H}\alpha$')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('RGB')

    plt.subplots_adjust(wspace=0.35)

    plt.savefig(filename,bbox_inches='tight',dpi=600)