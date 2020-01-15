from .style import figsize, newfig, final

from pathlib import Path
import numpy as np

import matplotlib as mpl
from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt

from ..analyse import PNLF

def plot_pnlf(data,mu,completeness,binsize=0.25,mlow=None,mhigh=None,
              filename=None,color='tab:red'):
    '''Plot Planetary Nebula Luminosity Function
    
    
    Parameters
    ----------
    data : ndarray
        apparent magnitude of the individual nebulae.
        
    mu : float
        distance modulus.
        
    completeness : float
        maximum magnitude for which we can detect all objects.
        
    binsize : float
        Size of the bins in magnitudes.
        
    mlow : float or None
        Lower edge of the bins (the root of the PNLF if None)
    
    mhigh : float
        Upper edge of the bins.
    '''
    
    Mmax = -4.47
    
    # the fit is normalized to 1 -> multiply with number of objects
    N = len(data[data<completeness])
    if not mlow:
        mlow = Mmax+mu
    if not mhigh:
        mhigh = completeness+2
    
    hist, bins  = np.histogram(data,np.arange(mlow,mhigh,binsize),normed=False)
    err = np.sqrt(hist)
    # midpoint of the bins is used as position for the plots
    m = (bins[1:]+bins[:-1]) / 2
    
    # for the fit line we use a smaller binsize
    binsize_fine = 0.05
    bins_fine = np.arange(mlow,mhigh,binsize_fine)
    m_fine = (bins_fine[1:]+bins_fine[:-1]) /2
    
    # create an empty figure
    fig = newfig(ratio=0.5)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # scatter plot
    ax1.errorbar(m[m<completeness],hist[m<completeness],yerr=err[m<completeness],
                 marker='o',ms=6,mec=color,mfc=color,ls='none',ecolor=color)
    ax1.errorbar(m[m>=completeness],hist[m>=completeness],yerr=err[m>completeness],
                 marker='o',ms=6,mec=color,mfc='white',ls='none',ecolor=color)
    ax1.plot(m_fine,binsize/binsize_fine*N*PNLF(bins_fine,mu=mu,mhigh=completeness),c=color,ls='dotted')
    ax1.axvline(completeness,c='black',lw=0.2)
    ax1.axvline(mu+Mmax,c='black',lw=0.2)

    # adjust plot
    ax1.set_yscale('log')
    ax1.set_xlim([1.1*mlow-0.1*mhigh,mhigh])
    ax1.set_ylim([0.8,1.5*np.max(hist)])
    ax1.set_xlabel(r'$m_{[\mathrm{OIII}]}$ / mag')
    ax1.set_ylabel(r'$N$')
    
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.2g}'.format(y)))
    ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))

    # cumulative
    ax2.plot(m[m<completeness],np.cumsum(hist[m<completeness]),ls='none',mfc=color,mec=color,ms=6,marker='o')
    ax2.plot(m,N*np.cumsum(PNLF(bins,mu=mu,mhigh=completeness)),ls='dotted',color=color)

    
    # adjust plot    
    ax2.set_xlim([mlow,completeness])
    ax2.set_ylim([-0.1*N,1.1*N])
    ax2.set_xlabel(r'$m_{[\mathrm{OIII}]}$ / mag')
    ax2.set_ylabel(r'Cumulative N')
    
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
    ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    plt.tight_layout()

    if filename:
        #savefig(filename.with_suffix('.pgf'),bbox_inches='tight')
        savefig(filename.with_suffix('.pdf'),bbox_inches='tight')

    if not final:
        plt.show()


def plot_emission_line_ratio(table,mu,filename=None):
    
    
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    
    style = {'SNR':{"marker":'o',"s":30,"edgecolors":'tab:red',"facecolors":'white'},
             'HII':{"marker":'+',"s":50,"color":'black'},
             'PN':{"marker":'o',"s":30,"color":'black'}
            }

    # ------------------------------------------------
    # left plot [OIII]/Ha over mOIII
    # ------------------------------------------------
    mOIII = np.linspace(24,29)
    OIII_Ha = 10**(-0.37*(mOIII-mu)-1.16)
    ax1.plot(mOIII,OIII_Ha,c='black',lw=0.6)
    

    for t in ['HII','SNR','PN']:
        tbl = table[table['type']==t]
        ax1.scatter(tbl['mOIII'],tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']),**style[t],label=t)
    ax1.legend()
    
    ax1.set(xlim=[24,30],
           ylim=[0.03,30],
           yscale='log',
           xlabel='m_OIII',
           ylabel='OIII/Ha')
    
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    # ------------------------------------------------
    # right plot Ha/[NII] over Ha/[SII]
    # ------------------------------------------------
    #mOIII = np.linspace(24,29)
    #mu = 29.91
    #OIII_Ha = 10**(-0.37*(mOIII-mu)-1.16)
    #ax1.plot(mOIII,OIII_Ha)
    
    for t in ['HII','SNR','PN']:
        tbl = table[table['type']==t]
        ax2.scatter(np.log10(tbl['HA6562']/tbl['SII6716']),
                    np.log10(tbl['HA6562']/tbl['NII6583']),**style[t],label=t)
    ax2.legend()

    ax2.axvline(np.log10(2.5),c='black',lw=0.6) 
    vert_SNR = np.array([[-0.1,-0.5],[-0.1,0.05],[0.3,0.25],[0.3,0.05],[0.1,-0.05],[0.1,-0.5]])
    ax2.add_patch(mpl.patches.Polygon(vert_SNR,Fill=False,edgecolor='black'))
    vert_SNR = np.array([[0.5,0.2],[0.5,0.7],[0.9,0.7],[0.9,0.2]])
    ax2.add_patch(mpl.patches.Polygon(vert_SNR,Fill=False,edgecolor='black'))
    ax2.plot([0.1,1.3],[-0.45,0.8],c='black',lw=0.6)
    ax2.text(-0.1,-0.6,'SNR')
    
    ax2.set(xlim=[-1,1.5],
           ylim=[-1,1],
           #yscale='log',
           xlabel=r'Log (H$\alpha$ / [SII])',
           ylabel=r'Log (H$\alpha$ / [NII])')    
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))    

    plt.tight_layout()

    if filename:
        #savefig(filename.with_suffix('.pgf'),bbox_inches='tight')
        savefig(filename.with_suffix('.pdf'),bbox_inches='tight')

    if not final:
        plt.show()
    
