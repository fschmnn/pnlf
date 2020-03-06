from pathlib import Path
import numpy as np

import matplotlib as mpl
from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt

from photutils import CircularAperture
from astropy.visualization import simple_norm

tab10 = ['#e15759','#4e79a7','#f28e2b','#76b7b2','#59a14e','#edc949','#b07aa2','#ff9da7','#9c755f','#bab0ac']    

def classification_map(galaxy,parameters,tbl,filename):
    '''
    Plot data in galaxy for different lines and overplot the position
    of detected sources
    '''

    if "zoomin" in parameters[galaxy.name]:
        x,y,width = parameters[galaxy.name]['zoomin']
        height = width * galaxy.shape[0] / galaxy.shape[1]

        if x+width > galaxy.shape[1]:
            print(f'cutout is too wide')
            new_width =  galaxy.shape[1] - x
            height *= new_width / width 
            width = new_width

        if y+height > galaxy.shape[0]:
            print(f'cutout is to high')
            new_height =  galaxy.shape[0] - y
            width *= new_height / height
            height = new_height

    else:
        x,y=0,0
        height,width = galaxy.shape

    # ====== define input parameters =============================
    labels=['SII6716','HA6562','OIII5006']
    wcs=galaxy.wcs
    # ============================================================

    from pymuse.plot.plot import create_RGB
    #rgb = create_RGB(galaxy.HA6562,galaxy.OIII5006,galaxy.SII6716,percentile=97)
    rgb = create_RGB(galaxy.HA6562,galaxy.OIII5006_DAP,galaxy.SII6716,weights=[0.8,1,0.9],percentile=[97,97,97])

    table = tbl
    #table = tbl[tbl['mOIII']<galaxy.completeness_limit]

    fig = plt.figure(figsize=(6.974,6.974/2))
    ax1 = fig.add_subplot(131,projection=wcs)
    ax2 = fig.add_subplot(132,projection=wcs)
    ax3 = fig.add_subplot(133)

    #fig, (ax2,ax3,ax1) = plt.subplots(ncols=3,figsize=(6.974,6.974/2),subplot_kw={'projection':wcs})


    norm = simple_norm(galaxy.OIII5006_DAP,'linear',clip=False,max_percent=95)
    ax1.imshow(galaxy.OIII5006_DAP,norm=norm,cmap=plt.cm.Greens)

    norm = simple_norm(galaxy.HA6562,'linear',clip=False,max_percent=95)
    ax2.imshow(galaxy.HA6562,norm=norm,cmap=plt.cm.Reds)

    ax3.imshow(rgb)

    print(f'{len(table)} sources')
    for t,c in zip(['SNR','PN'],['royalblue','goldenrod']):
        
        sub = table[table['type']==t]
        print(f'{t:<3}: {len(sub):>3}')
        positions = np.transpose([sub['x'],sub['y']])
        apertures = CircularAperture(positions, r=6)
        #ax1.scatter(sub['x'],sub['y'],marker='o',s=5,lw=0.4,edgecolor=c,facecolors='none')
        apertures.plot(color=c,lw=.2, alpha=1,ax=ax1)
        apertures.plot(color=c,lw=.2, alpha=1,ax=ax2)
        apertures.plot(color=c,lw=.4, alpha=1,ax=ax3)

    '''
    for row in table[table['type']!='HII']:
        txt,x,y = row['id'], row['x']+5, row['y']
        
        if x1<x<x2 and y1<y<y2:
            #ax1.annotate(txt, (x, y),fontsize=0.5)
            #ax2.annotate(txt, (x, y),fontsize=0.5)
            ax3.annotate(txt, (x, y),fontsize=2,color='white')
    '''
    
    # first we create a legend with three invisible handles
    labels=['HA6562','OIII5006','SII6716']
    labels=[r'H$\alpha$',r'[OIII]',r'[SII]']
    handles = 3*[mpl.patches.Rectangle((0, 0), 0, 0, alpha=0.0)]
    leg = ax3.legend(handles,labels, frameon=True,framealpha=0.7,handlelength=0,prop={'size': 6},loc=3)

    # next we set the color of the three labels
    for color,text in zip(['red','green','blue'],leg.get_texts()):
        text.set_color(color)


    ax1.add_patch(mpl.patches.Rectangle((x,y),width,height,linewidth=0.3,edgecolor='k',facecolor='none'))
    ax2.add_patch(mpl.patches.Rectangle((x,y),width,height,linewidth=0.3,edgecolor='k',facecolor='none'))
    ax3.set_xlim([x,x+width])
    ax3.set_ylim([y,y+height])

    ax1.set(title=r'O[III]',
            xlabel='R.A. (J2000)',
            ylabel='Dec. (J2000)')

    ax2.set(title=r'$\mathrm{H}\alpha$',
            xlabel='R.A. (J2000)')
            
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('RGB')

    # format ticks with wcs
    # https://docs.astropy.org/en/stable/visualization/wcsaxes/ticks_labels_grid.html
    ax1.coords[0].set_ticks(number=3)
    ax1.coords[1].set_ticks(number=4)
    ax2.coords[0].set_ticks(number=3)
    ax2.coords[1].set_ticklabel_visible(False)

    #plt.subplots_adjust(wspace=-0.4)

    # it is a bit tricky to get the coordinates right (because data uses the wcs coordinates)
    # the easiest thing is to use fractions from the figure size
    # https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.patches.ConnectionPatch.html
    con = mpl.patches.ConnectionPatch(xyA=((x+width)/galaxy.shape[1],(y+height)/galaxy.shape[0]), xyB=(0,1), 
                                      coordsA="axes fraction", coordsB="axes fraction",
                                      axesA=ax2, axesB=ax3, color="black",linewidth=0.3)
    ax3.add_artist(con)
    con = mpl.patches.ConnectionPatch(xyA=((x+width)/galaxy.shape[1],(y)/galaxy.shape[0]), xyB=(0,0), 
                                      coordsA="axes fraction", coordsB="axes fraction",
                                      axesA=ax2, axesB=ax3, color="black",linewidth=0.3)
    ax3.add_artist(con)

    plt.savefig(filename,bbox_inches='tight',dpi=600)