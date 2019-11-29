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

from photutils import DAOStarFinder

from extinction import ccm89

from pymuse.io import ReadLineMaps
from pymuse.detection import detect_unresolved_sources, completeness_limit
from pymuse.photometry import measure_flux
from pymuse.analyse import emission_line_diagnostics, MaximumLikelihood, pnlf

logging.basicConfig(#filename='log.txt',
                    #filemode='w',  
                    #format='(levelname)s %(name)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# we save 
with open(Path('..') / 'data' / 'interim' / 'parameters.json') as json_file:
    parameters = json.load(json_file)

'''
IC5332    NGC1087    NGC1365    NGC1512    NGC1566    NGC1672    NGC2835   
NGC3351   NGC3627    NGC4254    NGC4535    NGC5068    NGC628
'''

data_raw = Path('d:\downloads\MUSEDAP')
name = 'IC5332'

threshold = 8
distance_modulus = parameters[name]['mu']

# read in the MUSEDAP data
galaxy = ReadLineMaps(data_raw / name)

#if 'completeness_limit' not in parameters[name].keys():
if True:
    # determine the completeness limit
    mock_sources = completeness_limit(
                    galaxy,
                    'OIII5006',
                    DAOStarFinder,
                    threshold=threshold,
                    iterations=10,
                    oversize_PSF=1.3
                                    )
    logger.error('determine a completeness limit and save it to parameters.json')
    sys.exit()
else:
    completeness_limit = parameters[name]['completeness_limit']

# find unresolved sources
sources = detect_unresolved_sources(galaxy,
                                    'OIII5006',
                                    StarFinder=DAOStarFinder,
                                    threshold=threshold,
                                    PSF_size = 1.3,
                                    save=False)

# measure flux for the different lines
flux = measure_flux(galaxy,aperture_size=3)

extinction = ccm89(wave=np.array([5007.]),a_v=0.2,r_v=3.1,unit='aa')[0]
flux['mOIII'] -= extinction

tbl = emission_line_diagnostics(flux,distance_modulus=distance_modulus,
                                completeness_limit=completeness_limit)


fitter = MaximumLikelihood(pnlf,tbl[tbl['type']=='PN']['mOIII'])
fit = fitter([distance_modulus,20])

print(f'mu = {fit[0]:.3f}')