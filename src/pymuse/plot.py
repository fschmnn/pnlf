import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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

    
