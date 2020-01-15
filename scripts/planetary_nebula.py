'''production script for planetary nebula

this script is a streamlined version of the code in planetary_nebula.ipynb.
The notebook was used for testing and peaking into some results, while 
this script is used to produce the final plots/tables.
'''

import sys
from pathlib import Path
import logging
import json

import numpy as np 
import matplotlib.pyplot as plt 

from astropy.io import ascii, fits
from astropy.table import Table
from astropy.coordinates import SkyCoord  
from photutils import DAOStarFinder

from extinction import ccm89

from pymuse.auxiliary import search_table
from pymuse.io import ReadLineMaps
from pymuse.detection import detect_unresolved_sources, completeness_limit
from pymuse.photometry import measure_flux
from pymuse.analyse import emission_line_diagnostics, MaximumLikelihood, pnlf, Distance
from pymuse.plot.pnlf import plot_emission_line_ratio, plot_pnlf

logging.basicConfig(#filename='log.txt',
                    #filemode='w',  
                    #format='(levelname)s %(name)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
basedir = Path('..')

# we save 
with open(basedir / 'data' / 'interim' / 'parameters.json') as json_file:
    parameters = json.load(json_file)

with fits.open(basedir / 'data' / 'raw' / 'phangs_sample_table_v1p4.fits') as hdul:
    sample_table = Table(hdul[1].data)

for name in parameters.keys():
    tmp = search_table(sample_table,name)
    if tmp:
        d = Distance(tmp['DIST'][0]*1e6,'pc').to_distance_modulus()
        parameters[name]["mu"] = d
print('using mu from sample table')

'''
IC5332    NGC1087    NGC1365    NGC1512    NGC1566    NGC1672    NGC2835   
NGC3351   NGC3627    NGC4254    NGC4535    NGC5068    NGC628
'''

data_raw = Path('d:\downloads\MUSEDAP')
basedir = Path('..')

for name in parameters.keys():

    '''
    Step 1: Read in the data
    '''

    galaxy = ReadLineMaps(data_raw / name)
    setattr(galaxy,'mu',parameters[galaxy.name]['mu'])
    setattr(galaxy,'alpha',parameters[galaxy.name]['power_index'])
    setattr(galaxy,'completeness_limit',parameters[galaxy.name]['completeness_limit'])

    '''
    Step 2: Detect sources
    '''

    sources = detect_unresolved_sources(galaxy,
                                        'OIII5006',
                                        StarFinder=DAOStarFinder,
                                        threshold=8,
                                        save=False)

    '''
    Step 3: Measure fluxes
    '''

    flux = measure_flux(galaxy,sources, galaxy.alpha,aperture_size=2.,background='local')

    for col in ['HA6562','NII6583','SII6716']:
        flux[col][flux[col]<0] = flux[f'{col}_err'][flux[col]<0]
        flux[col][flux[col]/flux[f'{col}_err']<3] = flux[f'{col}_err'][flux[col]/flux[f'{col}_err']<3]
        
    # calculate astronomical coordinates for comparison
    flux['SkyCoord'] = SkyCoord.from_pixel(flux['x'],flux['y'],galaxy.wcs)

    # calculate magnitudes from measured fluxes
    flux['mOIII'] = -2.5*np.log10(flux['OIII5006']*1e-20) - 13.74
    flux['dmOIII'] = np.abs( 2.5/np.log(10) * flux['OIII5006_err'] / flux['OIII5006'] )

    # correct for milky way extinction
    extinction = ccm89(wave=np.array([5007.]),a_v=0.2,r_v=3.1,unit='aa')[0]
    flux['mOIII'] -= extinction

    '''
    Step 4: Emission line diagnostics
    '''

    tbl = emission_line_diagnostics(flux,galaxy.mu,galaxy.completeness_limit)

    filename = basedir / 'reports' / 'catalogues' / f'pn_candidates_{galaxy.name}.txt'
    with open(filename,'w',newline='\n') as f:
        tbl['RaDec'] = tbl['SkyCoord'].to_string(style='hmsdms',precision=2)
        for col in tbl.colnames:
            if col not in ['id','RaDec','type']:
                tbl[col].info.format = '%.3f' 
        ascii.write(tbl[['id','type','x','y','RaDec','OIII5006','OIII5006_err','mOIII','dmOIII','HA6562','HA6562_err',
                            'NII6583','NII6583_err','SII6716','SII6716_err']][tbl['type']!='NaN'],
                    f,format='fixed_width',delimiter='\t',overwrite=True)

    filename = basedir / 'reports' / 'figures' / f'{galaxy.name}_emission_line'
    plot_emission_line_ratio(tbl,galaxy.mu,filename=filename)

    '''
    Step 5: Fit with maximum likelihood
    '''

    data = tbl[(tbl['type']=='PN') & (tbl['mOIII']<galaxy.completeness_limit)]['mOIII']
    err  = tbl[(tbl['type']=='PN') & (tbl['mOIII']<galaxy.completeness_limit)]['dmOIII']
    #data = data[data>26]

    fitter = MaximumLikelihood(pnlf,
                               data,
                               mhigh=galaxy.completeness_limit)

    # a good guess would be mu_guess = min(data)-Mmax
    mu = fitter([24])[0]

    filename = basedir / 'reports' / 'figures' / f'{galaxy.name}_PNLF'
    plot_pnlf(tbl[tbl['type']=='PN']['mOIII'],mu,galaxy.completeness_limit,binsize=0.25,mhigh=32,filename=filename)

    print(f'{galaxy.name}: {mu:.2f} vs {parameters[galaxy.name]["mu"]:.2f}')