from .style import figsize, newfig, final

from pathlib import Path
import numpy as np

import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt

from astropy.io import ascii
from astroquery.nasa_ads import ADS


from ..analyse import PNLF

basedir = Path(__file__).parent.parent.parent.parent

ADS.TOKEN = open(basedir/'notebooks'/'ADS_DEV_KEY','r').read()

from ..constants import tab10, single_column, two_column


def plot_pnlf(data,mu,completeness,binsize=0.25,mlow=None,mhigh=None,
              filename=None,metadata=None,color='tab:red',alpha=1,axes=None):
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
    mpl.use('agg')

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

    if not axes:
        # create an empty figure
        fig = figure(figsize=(two_column,two_column/2))
        #fig = figure(figsize=(6.974,6.974/2))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
    else:
        ax1,ax2 = axes 
        fig = ax1.get_figure()

    #fig.set_facecolor((223/255,207/255,187/255))
    #ax1.set_facecolor((223/255,207/255,187/255))

    # scatter plot
    ax1.errorbar(m[m<completeness],hist[m<completeness],yerr=err[m<completeness],
                 marker='o',ms=6,mec=color,mfc=color,ls='none',ecolor=color,alpha=alpha)
    ax1.errorbar(m[m>=completeness],hist[m>=completeness],yerr=err[m>completeness],
                 marker='o',ms=6,mec=color,mfc='white',ls='none',ecolor=color,alpha=alpha)
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
    #ax2.plot(m_fine[m_fine<=completeness],np.cumsum(hist_fine[m_fine<=completeness]),ls='none',mfc=color,mec=color,ms=4,marker='o',alpha=alpha)
    data.sort()
    ax2.plot(data,np.arange(1,len(data)+1,1),ls='none',mfc=color,mec=color,ms=2,marker='o',alpha=alpha)
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
        #savefig(filename.with_suffix('.png'),bbox_inches='tight',dpi=600,facecolor=(223/255,207/255,187/255))

        #with PdfPages(filename.with_suffix('.pdf')) as pdf:
        #    pdf.savefig(fig,metadata= {'Creator': 'matplotlib', 'Author': 'FS', 'Title': 'NGC628'})
    else:
        plt.show()

    return (ax1,ax2)


def plot_emission_line_ratio(table,mu,completeness=None,filename=None):
    
    Mmax = -4.47
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(two_column,two_column/3.3))
    
    style = {'SNR':{"marker":'o',"ms":3,"mfc":'None',"mec":tab10[0],'ls':'none','ecolor':tab10[0]},
             'SNRorPN':{"marker":'o',"ms":4,"mfc":'white',"mec":'tab:green','ls':'none','ecolor':'tab:green'},
             'HII':{"marker":'+',"ms":3,"mec":tab10[1],'ls':'none'},
             'PN':{"marker":'o',"ms":2,"mfc":'black','mec':'black','ls':'none','ecolor':'black'}
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
        
        ax1.errorbar(tbl['mOIII']-mu,tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']),**style[t],label=t) 

        if t=='PN':
            # indicate for which PN we don't have a detection in HA6562
            tbl = tbl[~tbl['HA6562_detection']]
            ax1.errorbar(tbl['mOIII']-mu,1.11*tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']),
                         marker=r'$\uparrow$',ms=4,mec='black',ls='none') 
        if t=='SNR':
           tbl = tbl[tbl['SNRorPN']] 
           print(f'SNR or PN: {len(tbl[tbl["mOIII"]<completeness])} objects')
           ax1.errorbar(tbl['mOIII']-mu,tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']), marker='x',ms=2,mec=tab10[0],ls='none') 


    ax1.set(xlim=[-5,np.ceil(completeness-mu)],
           ylim=[0.03,200],
           yscale='log',
           xlabel=r'$M_{\mathrm{[OIII]}}$',
           ylabel=r'[OIII] / $(\mathrm{H}\alpha + \mathrm{[NII]})$')
    
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: '{:.16g}'.format(y)))

    ax1t = ax1.twiny()
    xlim1,xlim2 = ax1.get_xlim()
    ax1t.set_xticks(np.arange(np.ceil(xlim1+mu),np.floor(xlim2+mu)+1),minor=False)
    ax1t.set(xlim   = [xlim1+mu,xlim2+mu],
            xlabel = r'$m_{\mathrm{[OIII]}}$')

    # ------------------------------------------------
    # middle plot Ha/[NII] over Ha/[SII]
    # ------------------------------------------------
    
    for t in ['HII','SNR','PN']:
        tbl = table[(table['type']==t)] #& (table['HA6562_detection'] | table['HA6562_detection'])]
        ax2.errorbar(np.log10(tbl['HA6562']/tbl['SII6716']),np.log10(tbl['HA6562']/tbl['NII6583']),
                     **style[t],label=t)

        if t=='PN':
            # indicate for which PN we don't have a detection in HA6562
            tbl = tbl[~tbl['SII6716_detection'] | ~tbl['HA6562_detection']]
            ax2.errorbar(0.03+np.log10(tbl['HA6562']/tbl['SII6716']),np.log10(tbl['HA6562']/tbl['NII6583']),
                         marker=r'$\!\rightarrow$',ms=4,mec='black',ls='none') 
        if t=='SNR':
           tbl = tbl[tbl['SNRorPN']] 
           ax2.errorbar(np.log10(tbl['HA6562']/tbl['SII6716']),np.log10(tbl['HA6562']/tbl['NII6583']), marker='x',ms=2,mec=tab10[0],ls='none') 


    ax2.legend()

    ax2.axvline(np.log10(2.5),c='black',lw=0.6) 
    vert_SNR = np.array([[-0.1,-0.5],[-0.1,0.05],[0.3,0.25],[0.3,0.05],[0.1,-0.05],[0.1,-0.5]])
    #ax2.add_patch(mpl.patches.Polygon(vert_SNR,Fill=False,edgecolor='black'))
    vert_SNR = np.array([[0.5,0.2],[0.5,0.7],[0.9,0.7],[0.9,0.2]])
    #ax2.add_patch(mpl.patches.Polygon(vert_SNR,Fill=False,edgecolor='black'))
    #ax2.plot([0.1,1.3],[-0.45,0.8],c='black',lw=0.6)
    #ax2.text(-0.1,-0.6,'SNR')
    
    ax2.set(xlim=[-0.5,1.5],
           ylim=[-0.2,1],
           #yscale='log',
           xlabel=r'$\log (\mathrm{H}\alpha$ / [SII])',
           ylabel=r'$\log (\mathrm{H}\alpha$ / [NII])')    
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))    

    # ------------------------------------------------
    # right plot with velocity dispersion
    # ------------------------------------------------

    bins = np.arange(0,200,10)
    SN_cut = 9

    ax3.hist(table[(table['type']=='PN') & (table['v_SIGMA_S/N']>SN_cut)]['v_SIGMA'],bins=bins,alpha=1,label='PN',color='black')
    ax3.hist(table[(table['type']=='SNR') & (table['v_SIGMA_S/N']>SN_cut)]['v_SIGMA'],bins=bins,alpha=0.8,label='SNR',color=tab10[0])

    ax3.set_xlabel(r'$\sigma_V$ / km s$^{-1}$')
    ax3.axvline(100,c='black',lw=0.6) 
    ax3.legend()

    plt.subplots_adjust(wspace=0.35)

    #plt.tight_layout()

    if filename:
        #savefig(filename.with_suffix('.pgf'),bbox_inches='tight')
        savefig(filename.with_suffix('.pdf'),bbox_inches='tight')

    if not final:
        plt.show()

importance = [
'PNLF',
'TRGB',
'Cepheids',
'SNIa',
'SNII optical',
'Tully-Fisher',  
'Tully est',
'Brightest Stars',
'Grav. Stability Gas. Disk',
'Disk Stability',
'IRAS',
'Ring Diameter',
'Sosies',
'Statistical',
]
importance = importance[::-1]

colors = {
'Brightest Stars':'#9c755f',
'Cepheids':'#59a14e',
'Grav. Stability Gas. Disk':'#76b7b2',
'Disk Stability':'#76b7b2',
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

def compare_distances(name,distance,plus,minus,filename=None):
    
    mpl.use('pgf')
    mpl.rcParams['pgf.preamble'] = [r'\usepackage[hidelinks]{hyperref}', ]

    distances = ascii.read(basedir / 'data' / 'external' / f'{name}.csv',delimiter=',')
    references = ascii.read(basedir / 'data' / 'external' / f'paper_list.csv',encoding='utf8',delimiter=',')

    ref_dict = {
    'Refcode' : list(references['Refcode']),
    'Authors' : list(references['Authors']),
    'Title' : list(references['Title'])
    }

    new = 0
    # search for missing references
    for bibcode in distances['Refcode']:
        if bibcode not in ref_dict['Refcode']:
            try:
                result = ADS.query_simple(bibcode)
                ref_dict['Refcode'].append(bibcode)
                ref_dict['Authors'].append(';'.join(result['author'][0]))
                ref_dict['Title'].append(result['title'][0][0])
                new += 1
            except:
                print(f'can not find {bibcode} for {name}')

    if new>0:
        print(f'{new} new items added')
        references = Table(ref_dict)

        references.sort('Refcode',reverse=True)
        with open(basedir / 'data' / 'external' / f'paper_list.csv','w',encoding='utf8',newline='\n') as f:
            ascii.write(references,f,format='csv',overwrite=True,delimiter=',')

    references.add_index('Refcode')
    references['year'] = [int(row['Refcode'][:4]) for row in references]
    references['firstAuthor'] = [row['Authors'].split(',')[0] for row in references]
    references['name'] = [f'{row["firstAuthor"]}+{str(row["year"])[2:]}' for row in references]

    distances['year'] = [int(row['Refcode'][:4]) for row in distances]
    distances['name'] = [references.loc[ref]['name'] for ref in distances['Refcode']]
    base_url = 'https://ui.adsabs.harvard.edu/abs/'
    distances['link'] = [f'\href{{{base_url + row["Refcode"]}}}{{{row["name"]}}}' for row in distances]


    # ugly workaround 
    # some papers publish more than one distance. We use only the one with the smallest uncertainty
    distances = distances[np.abs(distances['(m-M)']-distance)<1]
    
    remove = []
    for i,row in enumerate(distances):

        name = row['name']
        sub = distances[np.where(distances['name']==name)]
        if len(sub) > 1:
            if row['err(m-M)'] > np.min(sub['err(m-M)']):
                remove.append(i)
    remove.sort(reverse=True)
    for i in remove:
        distances.remove_row(i)
    

    distances['sort_order'] = [importance.index(row['Method']) for row in distances]
    distances.sort(['sort_order','year'])
    distances['y'] = np.arange(1,len(distances)+1)

    fig = plt.figure(figsize=(single_column,0.25*len(distances)))
    ax = fig.add_subplot(1,1,1)

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

    #for row in distances:
    #    ax.annotate(row['link'], (0, row['y']),fontsize=2,color='black')

    ax.set_yticks(distances['y'],minor=False)
    ax.set_yticklabels(distances['link'])
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
    ax.set_xlabel(r'$(m-M)\ /\ \mathrm{mag}$')
    ax.set_ylim([0.5,len(distances)+0.5])

    ax2 = ax.twinx()
    ax2.set_yticks(method_ticks,minor=False)
    ax2.set_yticklabels(method_labels)
    ax2.set_ylim([0.5,len(distances)+0.5])

    plt.tight_layout()

    if filename:
        plt.savefig(filename.with_suffix('.pdf'),bbox_inches='tight')
        plt.savefig(filename.with_suffix('.pgf'),bbox_inches='tight')

    plt.show()