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

def _plot_pnlf(data,mu,completeness,binsize=0.4,mlow=None,mhigh=None,Mmax=-4.47,color=tab10[0],alpha=1,ax=None,ms=6):
    '''Plot PNLF

    this function plots a minimalistic PNLF (without labels etc.)
    '''
    
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

    if not ax:
        # create an empty figure
        fig = figure(figsize=(single_column,single_column))
        #fig = figure(figsize=(6.974,6.974/2))
        ax = fig.add_subplot(1,1,1)
    else:
        fig = ax.get_figure()

    # scatter plot
    ax.errorbar(m[m<completeness],hist[m<completeness],yerr=err[m<completeness],
                 marker='o',ms=ms,mec=color,mfc=color,ls='none',ecolor=color,alpha=alpha,label=r'$m_\mathrm{[OIII]}<$completeness limit')
    ax.errorbar(m[m>=completeness],hist[m>=completeness],yerr=err[m>completeness],
                 marker='o',ms=ms,mec=color,mfc='white',ls='none',ecolor=color,alpha=alpha,label=r'$m_\mathrm{[OIII]}>$completeness limit')
    ax.plot(m_fine,binsize/binsize_fine*N*PNLF(bins_fine,mu=mu,mhigh=completeness),c='black',ls='dotted',label='fit')

    ax.set_yscale('log')
    ax.set_xlim([1.1*mlow-0.1*mhigh,mhigh])
    ax.set_ylim([0.8,1.5*np.max(hist)])
    
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
    binsize_fine = 0.1
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
    #ax.plot(m_fine,N*np.cumsum(PNLF(bins_fine,mu=mu,mhigh=completeness)),ls='dotted',color=color)
    ax.plot(data[data<completeness],N*cdf(data[data<completeness],mu,completeness),ls=':',color='k')

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

def plot_pnlf(data,mu,completeness,binsize=0.25,mlow=None,mhigh=None,Mmax=-4.47,
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
    mpl.use('agg')

    #logger.info(f'PNLF plot with {len(data)} points')
    if not axes:
        # create an empty figure
        fig = figure(figsize=(two_column,two_column/2))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
    else:
        ax1,ax2 = axes 
        fig = ax1.get_figure()

    ax1 = _plot_pnlf(data,mu,completeness,binsize=binsize,mlow=mlow,mhigh=mhigh,Mmax=Mmax,color=color,alpha=alpha,ax=ax1)
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
           ax1.errorbar(tbl['mOIII']-mu,tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']), marker='o',ms=2,mfc=tab10[0],mec=tab10[0],ls='none') 
   
    # objects that were rejeceted by eye
    tbl = table[table['overluminous']]
    ax1.errorbar(tbl['mOIII']-mu,tbl['OIII5006']/(tbl['HA6562']+tbl['NII6583']),marker='o',ms=3,ls='none',color='tab:green') 

    ax1.set(xlim=[-5,np.ceil(completeness-mu)-0.7],
           ylim=[0.03,200],
           yscale='log',
           xlabel=r'$M_{\mathrm{[O\,\textsc{iii}]}}$',
           ylabel=r'$I_{[\mathrm{O}\,\textsc{iii}]} \; /\;(I_{\mathrm{H}\,\alpha} + I_{\mathrm{[N\,\textsc{ii}]}})$')
    
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
        ax2.errorbar(np.log10(tbl['HA6562']/tbl['SII']),np.log10(tbl['HA6562']/tbl['NII6583']),
                     **style[t],label=t)

        if t=='PN':
            # indicate for which PN we don't have a detection in HA6562
            tbl = tbl[~tbl['SII_detection'] | ~tbl['HA6562_detection']]
            ax2.errorbar(0.03+np.log10(tbl['HA6562']/tbl['SII']),np.log10(tbl['HA6562']/tbl['NII6583']),
                         marker=r'$\!\rightarrow$',ms=4,mec='black',ls='none') 
        if t=='SNR':
           tbl = tbl[tbl['SNRorPN']] 
           ax2.errorbar(np.log10(tbl['HA6562']/tbl['SII']),np.log10(tbl['HA6562']/tbl['NII6583']), marker='o',ms=2,mfc=tab10[0],mec=tab10[0],ls='none') 

    tbl = table[table['overluminous']]
    ax2.errorbar(np.log10(tbl['HA6562']/tbl['SII']),np.log10(tbl['HA6562']/tbl['NII6583']),marker='o',ms=3,ls='none',color='tab:green') 

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
           xlabel=r'$\log_{10} (I_{\mathrm{H}\,\alpha} \; /\; I_{[\mathrm{S}\,\textsc{ii}]})$',
           ylabel=r'$\log_{10} (I_{\mathrm{H}\,\alpha} \; /\; I_{[\mathrm{N}\,\textsc{ii}]})$')    
    ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax2.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))    

    # ------------------------------------------------
    # right plot with velocity dispersion
    # ------------------------------------------------

    bins = np.arange(0,120,10)
    SN_cut = 9

    ax3.hist(table[(table['type']=='PN') & (table['v_SIGMA_S/N']>SN_cut)]['v_SIGMA'],bins=bins,alpha=1,label='PN',color='black')
    ax3.hist(table[(table['type']=='SNR') & (table['v_SIGMA_S/N']>SN_cut)]['v_SIGMA'],bins=bins,alpha=0.7,label='SNR',color=tab10[0])

    ax3.set(xlabel=r'$\sigma_V$ / km s$^{-1}$',ylabel='counts')
    #ax3.axvline(100,c='black',lw=0.6) 
    ax3.legend()

    plt.subplots_adjust(wspace=0.35)

    #plt.tight_layout()

    if filename:
        #savefig(filename.with_suffix('.pgf'),bbox_inches='tight')
        savefig(filename.with_suffix('.pdf'),bbox_inches='tight')

    if not final:
        plt.show()

Methods = {
 'Ring Diameter' : 'RD',
 'Grav. Stability Gas. Disk' : 'GSGD',
 'Brightest Stars' : 'BS',
 'SNII optical' : 'SNII',
 'Tully-Fisher' : 'TF',
 'Disk Stability' : 'DS',
 'Statistical' : 'Stat',
 'Tully est' : 'TE'
}

importance = [
'PNLF',
'TRGB',
'Cepheids',
'SNIa',
'SNII',
'NAM',
'Group',
'TF',  
'TE',
'BS',
'GSGD',
'DS',
'IRAS',
'RD',
'Sosies',
'Stat',
]
importance = importance[::-1]

##f28e2b orange
##edc949 gelb
colors = {
'BS':'#9c755f',
'Cepheids':'#59a14e',
'GSGD':'#76b7b2',
'DS':'#76b7b2',
'IRAS':'#ff9da7',
'NAM' : '#edc949',
'Group' : '#edc949',
'PNLF':'#e15759',
'RD':'#bab0ac',
'SNII':'#4e79a7',
'SNIa':'#76b7b2',
'Sosies':'#9c755f',
'Stat':'#bab0ac',
'TRGB':'#f28e2b',
'TE':'#b07aa2',
'TF':'#b07aa2'
}

def compare_distances(name,distance,plus,minus,filename=None):
    '''Compare the measured distance to literature values from NED

    '''

    mpl.use('pgf')
    mpl.rcParams['pgf.preamble'] = [r'\usepackage[hidelinks]{hyperref}',]

    distances = ascii.read(basedir / 'data' / 'literature distances' / f'{name}.csv',delimiter=',',comment='#')
    references = ascii.read(basedir / 'data' / 'literature distances' / f'paper_list.csv',encoding='utf8',delimiter=',')

    ref_dict = {
    'Refcode' : list(references['Refcode']),
    'Authors' : list(references['Authors']),
    'Title' : list(references['Title'])
    }

    new = 0
    # search for missing references
    for bibcode in distances['Refcode']:
        if bibcode not in ref_dict['Refcode']:
            # this shouldn't happen too often. So we open it in this loop
            from astroquery.nasa_ads import ADS
            ADS.TOKEN = open(basedir/'notebooks'/'ADS_DEV_KEY','r').read()
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
        with open(basedir / 'data' / 'literature distances' / f'paper_list.csv','w',encoding='utf8',newline='\n') as f:
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
    #distances = distances[np.abs(distances['(m-M)']-distance)<1]
    
    '''
    distances['i'] = np.arange(0,len(distances))
    remove = []
    for i,row in enumerate(distances):
        name = row['name']
        sub = distances[np.where(distances['name']==name)]
        if len(sub) > 1:
            if row['err(m-M)'] > np.min(sub['err(m-M)']):
                remove.append(i)
    
    # only show the 5 most recent results for each method
    for method in np.unique(distances['Method']):
        sub = distances[distances['Method']==method].copy()
        if len(sub)>5:
            sub.sort('year')
            remove += list(sub[:-5]['i'])
    remove = list(set(remove))

    remove.sort(reverse=True)
    for i in remove:
        distances.remove_row(i)
    '''

    methods = []
    year    = []
    DM      = []
    errDM   = []
    links   = []
    marker  = []
    ref     = []
    names   = []

    for g in distances.group_by('Refcode').groups:
        methods.append(Methods.get(g['Method'][0],g['Method'][0]))
        year.append(g['year'][0])
        links.append(g['link'][0])
        DM.append(g['(m-M)'].mean())
        errDM.append(np.sqrt(np.sum(g['err(m-M)']**2)))
        ref.append(g['Refcode'][0])
        names.append(g['name'][0])
        if len(g)==1:
            marker.append('o')
        else:
            marker.append('D')

    distances = Table([methods,year,DM,errDM,links,marker,ref,names],names=['Method','year','(m-M)','err(m-M)','link','marker','Refcode','name'])

    # only show the 5 most recent results for each method
    distances['i'] = np.arange(0,len(distances))
    remove=[]
    for method in np.unique(distances['Method']):
        sub = distances[distances['Method']==method].copy()
        if len(sub)>5:
            sub.sort('year')
            remove += list(sub[:-5]['i'])
    remove = list(set(remove))

    remove.sort(reverse=True)
    for i in remove:
        distances.remove_row(i)

    # distances requires [Method,year,(m-M),err(m-M),link] as columns
    distances['sort_order'] = [importance.index(row['Method']) for row in distances]
    distances.sort(['sort_order','year'])
    distances['y'] = np.arange(1,len(distances)+1)
    if len(distances)>9:
        fig = plt.figure(figsize=(single_column,0.15*len(distances)),tight_layout=True)
    else:
        fig = plt.figure(figsize=(single_column,(0.28-0.012*len(distances))*len(distances)),tight_layout=True)
    ax = fig.add_subplot(1,1,1)

    ax.fill_betweenx(np.arange(0,len(distances)+2), distance-minus, distance+plus,facecolor='black', alpha=0.5)
    ax.fill_betweenx(np.arange(0,len(distances)+2), distance-3*minus, distance+3*plus,facecolor='black', alpha=0.2)
    ax.axvline(distance,color='black',lw=0.8)

    method_ticks = []
    method_labels = np.unique(distances['Method'])
    for i,group in enumerate(distances.group_by('Method').groups):
        m = group[0]['Method']
        for row in group:
            ax.errorbar(row['(m-M)'],row['y'],xerr=row['err(m-M)'],color=colors[m],ls='none',fmt=row['marker'],ms=3)
        method_ticks.append(np.mean(group['y']))
        ax.axhline(np.max(group['y'])+0.5,color='gray',lw=0.5)

    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))
    ax.set_xlabel(r'$(m-M)\ /\ \mathrm{mag}$')

    ax.set_yticks(method_ticks,minor=False)
    ax.set_yticklabels(method_labels,ha='right')
    ax.set_ylim([0.5,len(distances)+0.5])

    ax2 = ax.twinx()
    ax2.set_yticks(distances['y'],minor=False)
    ax2.set_yticklabels(distances['link'],ha="left")
    ax2.set_ylim([0.5,len(distances)+0.5])

    ax2.text
    '''
    # create second x-axis in Mpc
    xmin,xmax=ax.get_xlim()
    n=0.2
    xticks_mpc = np.logspace(np.log10(Distance(distmod=np.ceil(xmin/n)*n).to(u.Mpc).value),np.log10(Distance(distmod=np.floor(xmax/n)*n).to(u.Mpc).value),4)
    xticks_mu  = Distance(xticks_mpc*u.Mpc).distmod
    ax3 = ax.twiny()
    ax3.set_xticks(xticks_mu.value,minor=False)
    ax3.set_xticklabels([f'{x:.2f}' for x in xticks_mpc],ha="left")
    ax3.set(xlim=[xmin,xmax],ylabel='$D$ / Mpc')
    '''
    
    if filename:
        plt.savefig(filename.with_suffix('.pdf'),bbox_inches='tight')
        plt.savefig(filename.with_suffix('.pgf'))

    # replace the link in the pgf document with a 
    #    \defcitealias{bibkey}{Author+year} 	
    #    \citetalias{bibkey} 
    # some papers are cited in the body and hence have a different key
    existing_keys = {
       '2017ApJ...834..174K' : 'Kreckel+2017',
       '2008ApJ...683..630H' : 'Herrmann+2008',
       '2002ApJ...577...31C' : 'Ciardullo+2002',
       '2020+Anand' : 'Anand+2020'
    }

    with open(basedir / 'reports' / 'citealias.tex','r',encoding='utf8') as f:
        citealias = set(f.read().split('\n'))

    with open(filename.with_suffix('.pgf'),'r',encoding='utf8') as f:
        text = f.read()
        for row in distances:
            row['Refcode'] = existing_keys.get(row["Refcode"],row["Refcode"])
            text=text.replace(row['link'],f'\citetalias{{{row["Refcode"]}}}')
            citealias.add(f'\defcitealias{{{row["Refcode"]}}}{{{row["name"]}}}')
    
    with open(filename.with_suffix('.pgf'),'w',encoding='utf8') as f:
        f.write(text)

    with open(basedir / 'reports' / 'citealias.tex','w',encoding='utf8') as f:
        f.write('\n'.join(sorted(citealias)))

    plt.show()