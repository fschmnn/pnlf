from .style import figsize, newfig, final

import logging
from pathlib import Path
import numpy as np

import matplotlib as mpl
#from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt

from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import Distance
import astropy.units as u

from scipy.stats import kstest

from ..analyse import PNLF, cdf

basedir = Path(__file__).parent.parent.parent.parent
logger = logging.getLogger(__name__)

from ..constants import tab10, single_column, two_column

def _plot_pnlf(data,mu,completeness,mask=None,binsize=0.4,mlow=None,mhigh=None,Mmax=-4.47,color=tab10[0],alpha=1,ax=None,ms=6):
    '''Plot PNLF

    this function plots a minimalistic PNLF (without labels etc.)
    '''

    if mask is None:
        mask = np.zeros_like(data,dtype=bool)
    
    # the fit is normalized to 1 -> multiply with number of objects
    N = len(data[data<completeness])-np.sum(mask)
    if not mlow:
        mlow = Mmax+mu
    if not mhigh:
        mhigh = completeness+2

    bins = np.arange(mlow,mhigh,binsize)
    hist, bins  = np.histogram(data[~mask],bins,normed=False)

    # we need to extend the bins
    bins_OL = np.arange(mlow-binsize*np.ceil((mlow-np.min(data))/binsize),mhigh,binsize)
    hist_OL, _  = np.histogram(data[mask],bins_OL,normed=False)
    m_OL = (bins_OL[1:]+bins_OL[:-1]) / 2

    err = np.sqrt(hist)
    # midpoint of the bins is used as position for the plots
    m = (bins[1:]+bins[:-1]) / 2
    
    # for the fit line we use a smaller binsize
    binsize_fine = 0.1
    bins_fine = np.arange(mlow-binsize_fine,mhigh+binsize_fine,binsize_fine)
    m_fine = (bins_fine[1:]+bins_fine[:-1]) /2
    #hist_fine, _ = np.histogram(data[~mask],bins_fine,normed=False)

    if not ax:
        # create an empty figure
        fig = figure(figsize=(single_column,single_column))
        #fig = figure(figsize=(6.974,6.974/2))
        ax = fig.add_subplot(1,1,1)
    else:
        fig = ax.get_figure()

    
    # scatter plot
    ax.errorbar(m[m<completeness],hist[m<completeness],yerr=err[m<completeness],xerr=binsize/2,
                 marker='o',ms=ms,mec=color,mfc=color,ls='none',ecolor=color,alpha=alpha,label=r'below completeness limit')
    ax.errorbar(m[m>=completeness],hist[m>=completeness],yerr=err[m>completeness],xerr=binsize/2,
                 marker='o',ms=ms,mec=color,mfc='white',ls='none',ecolor=color,alpha=alpha,label=r'above completeness limit')
    
    # for overluminous objects
    ax.errorbar(m_OL,hist_OL,yerr=np.sqrt(hist_OL),xerr=binsize/2,
                marker='o',ms=ms,mec='tab:blue',mfc='tab:blue',ls='none',ecolor='tab:blue',alpha=alpha,label=r'overluminous')

    ax.plot(m_fine,binsize/binsize_fine*N*PNLF(bins_fine,mu=mu,mhigh=completeness),c='black',ls='dotted',label='fit')
    #ax.axvline(completeness,ls='--',color='black')

    ax.set_yscale('log')
    if np.any(mask):
        ax.set_xlim([1.1*np.min(m_OL)-0.1*mhigh,mhigh])
    else:
        ax.set_xlim([1.1*mlow-0.1*mhigh,mhigh])

    # dirty hack to avoid 100 in NGC3351
    ax.set_ylim([0.6,min(2*np.max(hist),99)])


    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.3g}'.format(y)))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))

    plt.tight_layout()

    return ax


def _plot_cum_pnlf(data,mu,completeness=None,binsize=None,mlow=None,mhigh=None,Mmax=-4.47,color=tab10[0],alpha=1,ax=None):
    '''Plot cumulative PNLF

    this function plots a minimalistic cumulative PNLF (without labels etc.)
    '''
    
    if not completeness:
        completeness = max(data)+1
    N = len(data[data<completeness])
    if not mlow:
        #mlow = Mmax+mu
        mlow = np.min(data)-0.1
    if not mhigh:
        mhigh = completeness+2
    
    hist, bins  = np.histogram(data,np.arange(mlow,mhigh,binsize),normed=False)
    err = np.sqrt(hist)
    # midpoint of the bins is used as position for the plots
    m = (bins[1:]+bins[:-1]) / 2
    
    # for the fit line we use a fixed binsize
    binsize_fine = 0.05
    bins_fine = np.arange(mlow,mhigh,binsize_fine)
    m_fine = (bins_fine[1:]+bins_fine[:-1]) /2
    hist_fine, _ = np.histogram(data,bins_fine,normed=False)

    if not ax:
        # create an empty figure
        fig = figure(figsize=(single_column,single_column))
        #fig = figure(figsize=(6.974,6.974/2))
        ax = fig.add_subplot(1,1,1)
    else:
        fig = ax.get_figure()

    data.sort()

    # old version with binned cdf
    ax.plot(m_fine[m_fine<completeness],N*cdf(m_fine[m_fine<completeness],mu,completeness),ls=':',color='k')
    #ax.plot(data[data<completeness],N*cdf(data[data<completeness],mu,completeness),ls=':',color='k')

    if binsize:
        bins = np.arange(mlow,mhigh,binsize)
        m = (bins[1:]+bins[:-1]) /2
        hist,_ = np.histogram(data,bins,density=False)
        ax.plot(m[m<completeness],np.cumsum(hist[m<completeness]),ls='none',mfc=color,mec=color,ms=2,marker='o',label='data')
    else:
        ax.plot(data[data<completeness],np.arange(1,N+1,1),ls='none',mfc=color,mec=color,ms=1,marker='o',alpha=alpha,label='data')



    diff = np.abs(cdf(data[data<completeness],mu,completeness)-np.arange(1,N+1)/N)
    i = np.argmax(diff)
    y = (N*cdf(data,mu,completeness)[i],np.arange(1,N+1)[i])
    ax.plot([data[i],data[i]], [min(y),max(y)],color='black')

    '''
    ks,pv = kstest(data[data<completeness],cdf,args=(mu,completeness))
    #print(f'statistic={ks:.3f}, pvalue={pv:.3f}')
    ax.text(0.1,0.9,f'$D_{{max}}={ks:.3f}$', transform=ax.transAxes)
    '''

    # adjust plot    
    ax.set_xlim([mlow,completeness])
    ax.set_ylim([0,1.02*N])
    
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    plt.tight_layout()

    return ax

def plot_pnlf(data,mu,completeness,mask=None,binsize=0.25,mlow=None,mhigh=None,Mmax=-4.47,
              filename=None,color='tab:red',alpha=1,axes=None):
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
    
    # sometimes the pgf backend crashes
    #mpl.use('agg')

    #logger.info(f'PNLF plot with {len(data)} points')
    if not axes:
        # create an empty figure
        fig = figure(figsize=(two_column,two_column/2))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
    else:
        ax1,ax2 = axes 
        fig = ax1.get_figure()

    ax1 = _plot_pnlf(data,mu,completeness,mask=mask,binsize=binsize,mlow=mlow,mhigh=mhigh,Mmax=Mmax,color=color,alpha=alpha,ax=ax1)
    ax2 = _plot_cum_pnlf(data,mu,completeness,binsize=None,mlow=mlow,mhigh=mhigh,Mmax=Mmax,color=color,alpha=alpha,ax=ax2)

    ax1.set_xlabel(r'$m_{[\mathrm{OIII}]}$ / mag')
    ax1.set_ylabel(r'$N$')

    ax2.set_xlabel(r'$m_{[\mathrm{OIII}]}$ / mag')
    ax2.set_ylabel(r'Cumulative N')

    plt.tight_layout()

    if filename:
        savefig(filename.with_suffix('.pdf'),bbox_inches='tight')
    else:
        plt.show()

    return (ax1,ax2)




def plot_emission_line_ratio(table,mu,completeness=None,filename=None):
    
    Mmax = -4.47
    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(0.9*single_column,single_column*1.8))
    
    style = {'SNR':{"marker":'o',"ms":3,"mfc":'None',"mec":tab10[0],'ls':'none','ecolor':tab10[0]},
             'SNRorPN':{"marker":'o',"ms":4,"mfc":'white',"mec":'tab:green','ls':'none','ecolor':'tab:green'},
             'HII':{"marker":'+',"ms":3,"mec":tab10[1],'ls':'none'},
             'PN':{"marker":'o',"ms":2,"mfc":'black','mec':'black','ls':'none','ecolor':'black'}
            }

    style = {'SNR':{"marker":'o',"ms":2,"mfc":'black',"mec":'black','ls':'none','ecolor':'black'},
             'HII':{"marker":'+',"ms":3,"mec":tab10[1],'ls':'none'},
             'PN':{"marker":'o',"ms":3,"mfc":'white','mec':tab10[0],'ls':'none','ecolor':tab10[0],'mew':0.8}
            }

    # ------------------------------------------------
    # left plot [OIII]/Ha over mOIII
    # ------------------------------------------------
    
    MOIII = np.linspace(-5,-1)
    OIII_Ha = 10**(-0.37*(MOIII)-1.16)
    ax1.plot(MOIII,OIII_Ha,c='black',lw=0.6)
    ax1.axhline(10**4)

    if completeness:
        ax1.axvline(completeness-mu,ls='--',c='grey',lw=0.5)
    ax1.axvline(Mmax,ls='--',c='grey',lw=0.5)

    for t in ['HII','PN','SNR']:
        tbl = table[table['type']==t]
        print(f'{t}: {len(tbl[tbl["mOIII"]<completeness])} objects')
        
        #ax1.errorbar(tbl['mOIII']-mu,tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']),**style[t],label=t) 

        if t=='PN':
            # indicate for which PN we don't have a detection in HA6562
            tmp = tbl[tbl['HA6562_detection']]
            ax1.errorbar(tmp['mOIII']-mu,10**tmp['R'],marker='o',ms=3,mfc=tab10[0],mec=tab10[0],ls='none',label=t) 
            #ax1.errorbar(tbl['mOIII']-mu,1.11*10**tbl['R'],
            #             marker=r'$\uparrow$',ms=4,mec=tab10[0],ls='none') 
            
            tmp = tbl[~tbl['HA6562_detection']]
            ax1.errorbar(tmp['mOIII']-mu,10**tmp['R'],**style[t]) 
        else:
            ax1.errorbar(tbl['mOIII']-mu,10**tbl['R'],**style[t],label=t) 

        #if t=='SNR':
        #   tbl = tbl[tbl['SNRorPN']] 
        #   print(f'SNR or PN: {len(tbl[tbl["mOIII"]<completeness])} objects')
        #   ax1.errorbar(tbl['mOIII']-mu,tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']), marker='o',ms=2,mfc='black',mec='black',ls='none') 
   
    # objects that were rejeceted by eye
    tbl = table[table['overluminous']]
    ax1.errorbar(tbl['mOIII']-mu,tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']),marker='o',ms=3,ls='none',color='tab:green') 

    ax1.set(xlim=[-5,np.ceil(completeness-mu)-0.7],
           ylim=[0.03,200],
           yscale='log',
           xlabel=r'$M_{[\mathrm{O}\,\textsc{iii}]}$',
           ylabel=r'$I_{[\mathrm{O}\,\textsc{iii}]} \; /\;(I_{\mathrm{H}\,\alpha+\mathrm{[N\,\textsc{ii}]}})$')
    
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    ax1t = ax1.twiny()
    xlim1,xlim2 = ax1.get_xlim()
    ax1t.set_xticks(np.arange(np.ceil(xlim1+mu),np.floor(xlim2+mu)+1),minor=False)
    ax1t.set(xlim   = [xlim1+mu,xlim2+mu],
            xlabel = r'$m_{[\mathrm{O}\,\textsc{iii}]}$')

    # ------------------------------------------------
    # middle plot Ha/[NII] over Ha/[SII]
    # ------------------------------------------------
    
    for t in ['HII','SNR','PN']:
        tbl = table[(table['type']==t)] #& (table['HA6562_detection'] | table['HA6562_detection'])]


        if t=='PN':
            # indicate for which PN we don't have a detection in HA6562
            tmp = tbl[tbl['SII_detection']]
            ax2.errorbar(np.log10(tmp['SII']/tmp['HA6562']),np.log10(tmp['NII6583']/tmp['HA6562']),
                         marker='o',ms=3,mfc=tab10[0],mec=tab10[0],ls='none',label=t) 
            tmp = tbl[~tbl['SII_detection']]
            ax2.errorbar(np.log10(tmp['SII']/tmp['HA6562']),np.log10(tmp['NII6583']/tmp['HA6562']),
                         **style[t]) 
        else:
            ax2.errorbar(np.log10(tbl['SII']/tbl['HA6562']),np.log10(tbl['NII6583']/tbl['HA6562']),
                     **style[t],label=t)
        #if t=='SNR':
        #   tbl = tbl[tbl['SNRorPN']] 
        #   ax2.errorbar(np.log10(tbl['SII']/tbl['HA6562']),np.log10(tbl['NII6583']/tbl['HA6562']), marker='o',ms=2,mfc='black',mec='black',ls='none') 

    tbl = table[table['overluminous']]
    ax2.errorbar(np.log10(tbl['SII']/tbl['HA6562']),np.log10(tbl['NII6583']/tbl['HA6562']),marker='o',ms=3,ls='none',color='tab:green') 

    ax2.legend()

    ax2.axvline(-0.3979,c='black',lw=0.6) 
    vert_SNR = np.array([[-0.1,-0.5],[-0.1,0.05],[0.3,0.25],[0.3,0.05],[0.1,-0.05],[0.1,-0.5]])
    #ax2.add_patch(mpl.patches.Polygon(vert_SNR,Fill=False,edgecolor='black'))
    vert_SNR = np.array([[0.5,0.2],[0.5,0.7],[0.9,0.7],[0.9,0.2]])
    #ax2.add_patch(mpl.patches.Polygon(vert_SNR,Fill=False,edgecolor='black'))
    #ax2.plot([0.1,1.3],[-0.45,0.8],c='black',lw=0.6)
    #ax2.text(-0.1,-0.6,'SNR')
    
    ax2.set(xlim=[-1.5,1],
           ylim=[-1.5,1],
           #yscale='log',
           xlabel=r'$\log_{10} \left(I_{[\mathrm{S}\,\textsc{ii}]} \; /\; I_{\mathrm{H}\,\alpha} \right)$',
           ylabel=r'$\log_{10} (I_{[\mathrm{N}\,\textsc{ii}]} \; /\; I_{\mathrm{H}\,\alpha})$')    
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))    

    # ------------------------------------------------
    # right plot with velocity dispersion
    # ------------------------------------------------
    '''
    bins = np.arange(0,150,10)
    SN_cut = 3

    ax3.hist(table[(table['type']=='PN') & (table['v_SIGMA_S/N']>SN_cut)]['v_SIGMA'],bins=bins,alpha=1,label='PN',color='black')
    ax3.hist(table[(table['type']=='SNR') & (table['v_SIGMA_S/N']>SN_cut)]['v_SIGMA'],bins=bins,alpha=0.7,label='SNR',color=tab10[0])

    ax3.set(xlabel=r'$\sigma_V$ / km s$^{-1}$',ylabel='counts')
    #ax3.axvline(100,c='black',lw=0.6) 
    ax3.legend()
    '''

    plt.subplots_adjust(hspace=0.25)

    #plt.tight_layout()

    if filename:
        #savefig(filename.with_suffix('.pgf'),bbox_inches='tight')
        savefig(filename.with_suffix('.pdf'),bbox_inches='tight')

    if not final:
        plt.show()

