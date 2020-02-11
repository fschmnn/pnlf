import logging 
from pathlib import Path
import sys

logging.basicConfig(stream=sys.stdout,
                    #format='(levelname)s %(name)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

import numpy as np 
import matplotlib as mpl

from matplotlib.pyplot import figure, show, savefig
import matplotlib.pyplot as plt

plt.style.use('TeX.mplstyle')

from astropy.io import ascii 

from pymuse.analyse import PNLF, pnlf

basedir = Path('..')


name = 'NGC628'
filename = basedir / 'reports' / 'catalogues' / f'pn_candidates_{name}.txt'
tbl = ascii.read(str(filename),format='fixed_width',delimiter='\t')

completeness = 29.
mu = 29.966 

criteria = ((tbl['type']=='PN'))
data = tbl[np.where(criteria & (tbl['mOIII']<completeness))]['mOIII']



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
    
    hist, bins  = np.histogram(data,np.arange(mlow,mhigh,binsize))
    err = np.sqrt(hist)
    # midpoint of the bins is used as position for the plots
    m = (bins[1:]+bins[:-1]) / 2
    
    # for the fit line we use a smaller binsize
    binsize_fine = 0.05
    bins_fine = np.arange(mlow,mhigh,binsize_fine)
    m_fine = (bins_fine[1:]+bins_fine[:-1]) /2
    
    # create an empty figure
    fig = figure(figsize=(3.321,3.321))
    ax1 = fig.add_subplot(1,1,1)

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

    plt.tight_layout()
    
    if filename:
        savefig(filename,bbox_inches='tight')
    else:
        show()


filename = basedir / 'reports' / f'{name}_PNLF.pdf'
plot_pnlf(tbl[criteria]['mOIII'],mu,completeness,binsize=0.4,mhigh=32,filename=filename)