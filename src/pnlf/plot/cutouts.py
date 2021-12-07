from pathlib import Path
import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from matplotlib.backends.backend_pdf import PdfPages

import astropy
from astropy.table import Table

import astropy.units as u
from astropy.visualization import simple_norm

from astropy.nddata import Cutout2D
import random
from scipy.stats import spearmanr

from photutils import CircularAperture         # define circular aperture

from ..constants import single_column, two_column
from .utils import create_RGB

from .utils import radial_profile


def cutout_with_profile(self,table,size=32,diagnostics=False,filename=None):
    '''create cutouts of a single sources and plot it'''

    data = getattr(self,'OIII5006')
    wcs  = self.wcs
    
    ncols = 5
    nrows = int(np.ceil(len(table)/ncols))

    width = 2*two_column
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,width/ncols*nrows))
    axes_iter = iter(axes.flatten())
    
    #print(f'{filename.stem.split("_")[0]}: plotting RGB+profile for {len(table)} objects')

    for row in table:
        
        ax1 = next(axes_iter)
        x,y = row[['x','y']]
        aperture_size=2.5*row['fwhm']/2

        # defien the size of the cutout region
        star = Cutout2D(data, (x,y), u.Quantity((size, size), u.pixel),wcs=wcs)

        rgb = create_RGB(self.HA6562,self.OIII5006,self.SII6716,percentile=99)
        yslice = slice(int(x-size/2),int(x+size/2))
        xslice = slice(int(y-size/2),int(y+size/2))
        
        ax1.set_yticks([])
        ax1.set_xticks([])
        try:
            im = ax1.imshow(rgb[xslice,yslice,:],origin='lower')
        except:
            text = f'{row["type"]}: {row["id"]}'
            t = ax1.text(0.05,0.9,text, transform=ax1.transAxes,color='black',fontsize=8)
            continue    


        # plot the radial profile
        profile = radial_profile(star.data,star.input_position_cutout)

        ax2 = ax1.inset_axes([0.01, 0.01, 0.32, 0.25])
        ax2.plot(profile,color='black')

        aperture = CircularAperture((size/2+(x-int(x)),size/2+(y-int(y))),aperture_size)
        aperture.plot(color='tab:red',lw=0.8,axes=ax1)
        ax2.axvline(aperture_size,color='tab:red',lw=0.5)
        ax2.axhline(0,color='black',lw=0.5)
        #fwhm = aperture_size/2.5*2
        #ax2.axvline(2*fwhm,color='gray',lw=0.8)
        #ax2.axvline(np.sqrt((4*fwhm)**2+(1.25*fwhm)**2),color='gray',lw=0.8)

        ax2.set_yticks([])
        ax2.set_xticks([])

        if diagnostics:
            # plot the line ratio
            ax3 = ax1.inset_axes([0.33, 0.01, 0.33, 0.25])
            mu = np.nanmean(table['mOIII']-table['MOIII'])
            MOIII = np.linspace(-5,-1)
            OIII_Ha = 10**(-0.37*(MOIII)-1.16)
            ax3.plot(MOIII,OIII_Ha,c='black',lw=0.6)
            ax3.axhline(10**4)
            ax3.axvline(-4.47,ls='--',c='grey',lw=0.5)
            ax3.scatter(row['mOIII']-mu,row['OIII5006_flux']/(row['HA6562_flux']+row['NII6583_flux']),marker='o',s=5,color='black') 

            if not row['HA6562_detection']:
                ax3.errorbar(row['mOIII']-mu,1.11*row['OIII5006_flux']/(row['HA6562_flux']+row['NII6583_flux']),
                                marker=r'$\uparrow$',ms=4,mec='black',ls='none') 
            ax3.set(xlim=[-5,-1],ylim=[0.03,200],yscale='log')
            ax3.set_yticks([])
            ax3.set_xticks([])

            # plot the line ratio 
            ax4 = ax1.inset_axes([0.66, 0.01, 0.32, 0.25])
            ax4.scatter(np.log10(row['HA6562_flux']/row['SII_flux']),np.log10(row['HA6562_flux']/row['NII6583_flux']),marker='o',s=5,color='black')
            ax4.axvline(np.log10(2.5),c='black',lw=0.6) 

            if not row['HA6562_detection'] or not row['SII_detection']:
                ax4.errorbar(0.03+np.log10(row['HA6562_flux']/row['SII_flux']),np.log10(row['HA6562_flux']/row['NII6583_flux']),
                            marker=r'$\!\rightarrow$',ms=4,mec='black',ls='none') 
            ax4.set(xlim=[-0.5,1.5],ylim=[-0.5,1.5])
            ax4.set_yticks([])
            ax4.set_xticks([])

        if row['exclude']:
            for loc in ['bottom','top','right','left']:
                ax1.spines[loc].set_color('tab:orange')
                ax1.spines[loc].set_linewidth(3)

        if  row['overluminous']:
            for loc in ['bottom','top','right','left']:
                ax1.spines[loc].set_color('tab:red')
                ax1.spines[loc].set_linewidth(3)

        text = f'{row["type"]}: {row["id"]}'
        t = ax1.text(0.05,0.9,text, transform=ax1.transAxes,color='black',fontsize=8)
        t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))
        t = ax1.text(0.05,0.8,f'mOIII={row["mOIII"]:.1f}', transform=ax1.transAxes,color='black',fontsize=8)
        t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))

    for i in range(nrows*ncols-len(table)):

        # remove the empty axes at the bottom
        ax = next(axes_iter)
        ax.remove()

    plt.subplots_adjust(wspace=-0.01,hspace=0.05)
    if filename:
        #plt.savefig(filename.with_suffix('.png'),dpi=600)
        plt.savefig(filename.with_suffix('.pdf'),dpi=600)
    plt.show()


def multipage_cutout_with_profile(galaxy,sample,filename,size=40):
    
    ncols = 4
    nrows = 8

    width = 8.27
    N = len(sample)
    Npage = nrows # number we get on each page

    with PdfPages(filename.with_suffix('.pdf')) as pdf:

        for i in range(int(np.ceil(N/Npage))):
            print(f'working on page {i+1}')

            sub_sample = sample[i*Npage:(i+1)*Npage]

            fig, axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(width,width/ncols*nrows))

            for row, (ax1,ax2,ax3,ax4) in zip(sub_sample,axes):  

                x,y = row[['x','y']]
                aperture_size=2.5*row['fwhm']/2
                aperture = CircularAperture((size/2+(x-int(x)),size/2+(y-int(y))),aperture_size)

                star = Cutout2D(galaxy.OIII5006, (x,y), u.Quantity((size, size), u.pixel),wcs=galaxy.wcs)
                profile = radial_profile(star.data,star.input_position_cutout)

                ax1.plot(profile,color='black')
                ax1.axvline(aperture_size,color='tab:red',lw=0.5)
                text = f'ID={row["id"]}: {row["mOIII"]} mag'
                t = ax1.text(0.07,0.87,text, transform=ax1.transAxes,color='black',fontsize=7)
                t.set_bbox(dict(facecolor='white', alpha=1, ec='white'))
                ax1.set_yticks([])
                ax1.set_xticks([])

                norm = simple_norm(star.data,clip=False,percent=99)
                im = ax2.imshow(star.data,norm=norm,origin='lower',cmap=plt.cm.Greens)
                aperture.plot(color='black',lw=0.8,axes=ax2)
                ax2.axis('off')

                star = Cutout2D(galaxy.HA6562, (x,y), u.Quantity((size, size), u.pixel),wcs=galaxy.wcs)
                norm = simple_norm(star.data,clip=False,percent=99)
                im = ax3.imshow(star.data,norm=norm,origin='lower',cmap=plt.cm.Reds)
                aperture.plot(color='black',lw=0.8,axes=ax3)
                ax3.axis('off')

                rgb = create_RGB(galaxy.HA6562,galaxy.OIII5006,galaxy.SII6716,percentile=99)
                yslice = slice(int(x-size/2),int(x+size/2))
                xslice = slice(int(y-size/2),int(y+size/2))

                try:
                    im = ax4.imshow(rgb[xslice,yslice,:],origin='lower')
                    aperture.plot(color='tab:red',lw=0.8,axes=ax4)
                except:
                    text = f'{row["id"]}: {row["mOIII"]} mag'
                    t = ax4.text(0.06,0.87,text, transform=ax4.transAxes,color='black',fontsize=7)
                    continue
                ax4.axis('off')

                if row['overluminous']:
                    for loc in ['bottom','top','right','left']:
                        ax1.spines[loc].set_color('tab:orange')
                        ax1.spines[loc].set_linewidth(3)
                if row['exclude']:
                    for loc in ['bottom','top','right','left']:
                        ax1.spines[loc].set_color('tab:red')
                        ax1.spines[loc].set_linewidth(3)


            for (ax1,ax2,ax3,ax4) in axes[len(sub_sample):]:
                ax1.axis('off')    
                ax2.axis('off')    
                ax3.axis('off')    
                ax4.axis('off')    


            plt.subplots_adjust(wspace=-0.1, hspace=0)



            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()


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
