from .style import figsize, newfig, final

from pathlib import Path
import numpy as np

import matplotlib as mpl
from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt

from astropy.io import ascii

from ..analyse import PNLF

basedir = Path(__file__).parent.parent.parent.parent
tab10 = ['#e15759','#4e79a7','#f28e2b','#76b7b2','#59a14e','#edc949','#b07aa2','#ff9da7','#9c755f','#bab0ac']    

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
    binsize_fine = 0.1
    bins_fine = np.arange(mlow,mhigh,binsize_fine)
    m_fine = (bins_fine[1:]+bins_fine[:-1]) /2
    hist_fine, _ = np.histogram(data,bins_fine,normed=False)

    # create an empty figure
    fig = figure(figsize=(6.974,6.974/2))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # scatter plot
    ax1.errorbar(m[m<completeness],hist[m<completeness],yerr=err[m<completeness],
                 marker='o',ms=6,mec=color,mfc=color,ls='none',ecolor=color)
    ax1.errorbar(m[m>=completeness],hist[m>=completeness],yerr=err[m>completeness],
                 marker='o',ms=6,mec=color,mfc='white',ls='none',ecolor=color)
    ax1.plot(m_fine,binsize/binsize_fine*N*PNLF(bins_fine,mu=mu,mhigh=completeness),c=color,ls='dotted')
    #ax1.axvline(completeness,c='black',lw=0.2)
    #ax1.axvline(mu+Mmax,c='black',lw=0.2)

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
    ax2.plot(m_fine[m_fine<=completeness],np.cumsum(hist_fine[m_fine<=completeness]),ls='none',mfc=color,mec=color,ms=4,marker='o')
    ax2.plot(m_fine,N*np.cumsum(PNLF(bins_fine,mu=mu,mhigh=completeness)),ls='dotted',color=color)

    
    # adjust plot    
    ax2.set_xlim([mlow,completeness+0.2])
    ax2.set_ylim([-0.1*N,1.2*N])
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
    
    
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(6.974,6.974/2))
    
    style = {'SNR':{"marker":'o',"ms":6,"mfc":'white',"mec":'tab:red','ls':'none','ecolor':'tab:red'},
             'HII':{"marker":'+',"ms":6,"mec":'black','ls':'none'},
             'PN':{"marker":'o',"ms":3,"mfc":'black','mec':'black','ls':'none','ecolor':'black'}
            }

    # ------------------------------------------------
    # left plot [OIII]/Ha over mOIII
    # ------------------------------------------------
    MOIII = np.linspace(-5,-1)
    OIII_Ha = 10**(-0.37*(MOIII)-1.16)
    ax1.plot(MOIII,OIII_Ha,c='black',lw=0.6)
    ax1.axhline(10**4)

    for t in ['HII','SNR','PN']:
        tbl = table[table['type']==t]
        ax1.errorbar(tbl['mOIII']-mu,tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']),**style[t],label=t) 

        if t=='PN':
            # indicate for which PN we don't have a detection in HA6562
            tbl = tbl[~tbl['HA6562_detection']]
            ax1.errorbar(tbl['mOIII']-mu,1.1*tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']),
                         marker='|',ms=10,mec='black',ls='none') 

    #wax1.legend()
    
    ax1.set(xlim=[-5,-1],
           ylim=[0.03,100],
           yscale='log',
           xlabel=r'$M_{\mathrm{[OIII]}}$',
           ylabel=r'[OIII] / $(\mathrm{H}\alpha + \mathrm{[NII]})$')
    
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    # ------------------------------------------------
    # right plot Ha/[NII] over Ha/[SII]
    # ------------------------------------------------
    #mOIII = np.linspace(24,29)
    #mu = 29.91
    #OIII_Ha = 10**(-0.37*(mOIII-mu)-1.16)
    #ax1.plot(mOIII,OIII_Ha)
    
    for t in ['HII','SNR','PN']:
        tbl = table[(table['type']==t)] #& (table['HA6562_detection'] | table['HA6562_detection'])]
        ax2.errorbar(np.log10(tbl['HA6562']/tbl['SII6716']),np.log10(tbl['HA6562']/tbl['NII6583']),
                     **style[t],label=t)

        if t=='PN':
            # indicate for which PN we don't have a detection in HA6562
            tbl = tbl[~tbl['SII6716_detection'] | ~tbl['HA6562_detection']]
            ax2.errorbar(0.03+np.log10(tbl['HA6562']/tbl['SII6716']),np.log10(tbl['HA6562']/tbl['NII6583']),
                         marker='_',ms=10,mec='black',ls='none') 

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
           xlabel=r'$\log (\mathrm{H}\alpha$ / [SII])',
           ylabel=r'$\log (\mathrm{H}\alpha$ / [NII])')    
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
    

def compare_distances(name,distance,plus,minus,filename=None):
    
    
    distances = ascii.read(basedir / 'data' / 'external' / f'{name}.csv')
    references = ascii.read(basedir / 'data' / 'external' / f'{name}ref.csv',encoding='utf8')

    references.add_index('Refcode')
    references['year'] = [int(row['Refcode'][:4]) for row in references]
    references['firstAuthor'] = [row['Authors'].split(',')[0] for row in references]
    references['name'] = [f'{row["firstAuthor"]}+{str(row["year"])[2:]}' for row in references]

    distances['year'] = [int(row['Refcode'][:4]) for row in distances]
    distances['name'] = [references.loc[ref]['name'] for ref in distances['Refcode']]

    distances.sort(['Method','year'])
    distances['y'] = np.arange(1,len(distances)+1)

    fig = plt.figure(figsize=(3.321,0.15*len(distances)))
    ax = fig.add_subplot(1,1,1)

    colors = {
     'Brightest Stars':'#9c755f',
     'Cepheids':'#59a14e',
     'Grav. Stability Gas. Disk':'#76b7b2',
     'IRAS':'#ff9da7',
     'PNLF':'#edc949',
     'Ring Diameter':'#bab0ac',
     'SNII optical':'#4e79a7',
     'SNIa':'#76b7b2',
     'Sosies':'#9c755f',
     'Statistical':'#bab0ac',
     'TRGB':'#f28e2b',
     'Tully est':'#b07aa2',
     'Tully-Fisher':'#b07aa2'
    }

    ax.fill_betweenx(np.arange(0,len(distances)+2), distance-minus, distance+plus,facecolor=tab10[0], alpha=0.4)
    ax.fill_betweenx(np.arange(0,len(distances)+2), distance-3*minus, distance+3*plus,facecolor=tab10[0], alpha=0.1)
    ax.axvline(distance,color=tab10[0])

    method_ticks = []
    method_labels = np.unique(distances['Method'])
    for i,group in enumerate(distances.group_by('Method').groups):
        m = group[0]['Method']
        ax.errorbar(group['(m-M)'],group['y'],xerr=group['err(m-M)'],color=colors[m],ls='none',fmt='o',ms=3)
        method_ticks.append(np.mean(group['y']))
        ax.axhline(np.max(group['y'])+0.5,color='gray',lw=0.5)

    ax.set_yticks(distances['y'],minor=False)
    ax.set_yticklabels(distances['name'])
    ax.set_xlabel(r'$(m-M)\ /\ \mathrm{mag}$')
    ax.set_ylim([0.5,len(distances)+0.5])

    ax2 = ax.twinx()
    ax2.set_yticks(method_ticks,minor=False)
    ax2.set_yticklabels(method_labels)
    ax2.set_ylim([0.5,len(distances)+0.5])

    #plt.tight_layout()

    if filename:
        plt.savefig(filename.with_suffix('.pdf'),bbox_inches='tight')
    plt.show()