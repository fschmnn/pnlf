from pathlib import Path
import logging 

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import Divider, Size
from astropy.io import ascii 
from astropy.table import Table 
from astropy.coordinates import Distance
import astropy.units as u
import matplotlib as mpl

from ..constants import tab10, single_column, two_column

basedir = Path(__file__).parent.parent.parent.parent
logger = logging.getLogger(__name__)

Methods = {
 'Cepheids' : 'Cepheid',
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
'Cepheid',
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
'Cepheid':'#59a14e',
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


def compile_distances(name):
    '''Compare the measured distance to literature values from NED

    '''

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

    for g in distances.group_by(['Refcode','Method']).groups:
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
    max_entries = 5
    remove=[]
    for method in np.unique(distances['Method']):
        sub = distances[distances['Method']==method].copy()
        if len(sub)>max_entries:
            sub.sort('year')
            remove += list(sub[:-max_entries]['i'])
    remove = list(set(remove))

    remove.sort(reverse=True)
    for i in remove:
        distances.remove_row(i)

    # distances requires [Method,year,(m-M),err(m-M),link] as columns
    distances['sort_order'] = [importance.index(row['Method']) for row in distances]
    distances.sort(['sort_order','year'])
    distances['y'] = np.arange(1,len(distances)+1)

    return distances


def plot_distances(name,distance,plus,minus,distances,filename=None):
    '''plot the measured distances and the literature values in distances

    this function uses the list compiled by compile_distances and plots it
    '''

    mpl.use('pgf')
    mpl.rcParams['pgf.preamble'] = [r'\usepackage[hidelinks]{hyperref}',]

    row_height = 0.10
    height = row_height*(4+len(distances))
    fig = plt.figure(figsize=(single_column,height))
    #ax = fig.add_subplot(1,1,1)
    ax = fig.add_axes([0.12,2*row_height/height,0.6,1-2*row_height/height])

    ax.fill_betweenx(np.arange(0,len(distances)+2), distance-minus, distance+plus,facecolor='black', alpha=0.5,zorder=1)
    ax.fill_betweenx(np.arange(0,len(distances)+2), distance-3*minus, distance+3*plus,facecolor='black', alpha=0.2,zorder=1)
    ax.axvline(distance,color='black',lw=0.8,zorder=1)

    method_ticks = []
    method_labels = np.unique(distances['Method'])
    for i,group in enumerate(distances.group_by('Method').groups):
        m = group[0]['Method']
        for row in group:
            ax.errorbar(row['(m-M)'],row['y'],xerr=row['err(m-M)'],color=colors[m],ls='none',fmt=row['marker'],ms=3,zorder=2)
        method_ticks.append(np.mean(group['y']))
        ax.axhline(np.max(group['y'])+0.5,color='gray',lw=0.5)

    #ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    ax.set_xlabel(r'$(m-M)\ /\ \mathrm{mag}$',labelpad=2)

    ax.set_yticks(method_ticks,minor=False)
    ax.set_yticklabels(method_labels,ha='right',fontsize=7)
    ax.set_ylim([0.5,len(distances)+0.5])
    ax.tick_params(top=False)

    # Create offset transform by 5 points in x direction
    dx = 2/72.; dy = 0/72. 
    offset = mpl.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    # apply offset transform to all x ticklabels.
    for label in ax.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    ax2 = ax.twinx()
    ax2.set_yticks(list(distances['y'])+[len(distances['y'])+1],minor=False)
    ax2.set_yticklabels(list(distances['link'])+[name],ha="left",fontsize=7)
    ax2.set_ylim([0.5,len(distances)+0.5])
    
    # create second x-axis in Mpc

    xmin,xmax=ax.get_xlim()
    if xmax>31.4 and xmax<31.50515:
        ax.set(xlim=[None,31.51])
    ax3 = ax.twiny()
    ax3.set(xlim=[Distance(distmod=xmin).to(u.Mpc).value,Distance(distmod=xmax).to(u.Mpc).value],
            xscale='log')
    ax3.set_xlabel(r'distance / Mpc',labelpad=2)
    ax3.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10,subs='all'))
    ax3.xaxis.set_major_formatter(mpl.ticker.LogFormatter(minor_thresholds=(2, 0.5)))
    ax3.tick_params(axis='x',zorder=0)

    fig.canvas.draw()

    # add the galaxy name
    pos = ax2.axes.get_position()
    xp = pos.x0+pos.width
    yp = pos.height
    #ax.annotate(name, xy=(0.02,yp+0.5*row_height/height), xycoords='figure fraction',fontsize=7)    
    #ax.annotate(name, xy=(pos.width+0.5*pos.x0,yp+0.5*row_height/height), xycoords='figure fraction',fontsize=7)    

    label = f'$(m-M)={distance:.2f}^{{+{plus:.2f}}}_{{-{minus:.2f}}}$'
    #ax.annotate(label, xy=(pos.width+0.5*pos.x0,yp+0.5*row_height/height), xycoords='figure fraction',fontsize=7)    

    #print(xp,yp)
    #print(fig.transFigure.inverted().transform(pos))

    #plt.subplots_adjust(bottom=0.32/height,top=1-0.32/height)
    #ax.set_position(mpl.transforms.Bbox(np.array([[0.12,2*row_height/height],[0.5,1-2*row_height/height]])))
    #ax2.set_position(mpl.transforms.Bbox(np.array([[0.12,2*row_height/height],[0.5,1-2*row_height/height]])))

    '''
    def forward(x):
        return Distance(distmod=x).to(u.Mpc).value

    def inverse(x):
        return Distance(x*u.Mpc).distmod.value
    xmin,xmax=ax.get_xlim()

    secax = ax.secondary_xaxis('top', functions=(forward, inverse))
    #secax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(n=4))
    secax.xaxis.set_minor_locator(mpl.ticker.LogLocator())
    secax.set_xlabel('$D$ / Mpc')
    secax.set(xlim=[Distance(distmod=xmin).to(u.Mpc).value,Distance(distmod=xmax).to(u.Mpc).value])
    '''

    if filename:
        plt.savefig(filename.with_suffix('.pdf'))
        plt.savefig(filename.with_suffix('.pgf'))

        # replace the link in the pgf document with a 
        #    \defcitealias{bibkey}{Author+year} 	
        #    \citetalias{bibkey} 
        # some papers are cited in the body and hence have a different key
        existing_keys = {
        '2021arXiv210501982R' : 'Roth+2021',
        '2017ApJ...834..174K' : 'Kreckel+2017',
        '2008ApJ...683..630H' : 'Herrmann+2008',
        '2002ApJ...577...31C' : 'Ciardullo+2002',
        '2021MNRAS.501.3621A' : 'Anand+2021',
        '2001ApJ...553...47F' : 'Freedman+2001'
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

