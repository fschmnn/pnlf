from pathlib import Path

from astropy.io import ascii 
from astropy.coordinates import SkyCoord

from pymuse.auxiliary import search_table, Distance

basedir = Path(__file__).parent.parent.parent

def string_to_ra(string):
    '''convert coordinates from Kreckel et al. (2017) to astropy
    
    the right ascension in the paper is given as 
    "01:36:42.212" but astropy requires "01h36m42.212s".
    This function replaces the ":" with the appropriate character.
    '''
    return string.replace(':','h',1).replace(':','m') + 's'

def string_to_dec(string):
    '''convert coordinates from Kreckel et al. (2017) to astropy
    
    the declination in the paper is given as "01:36:42.212" 
    but astropy requires "01d36m42.212s".
    This function replaces the ":" with the appropriate character.
    '''
    return string.replace(':','d',1).replace(':','m') + 's'


# from Kreckel+2017

pn_NGC628_kreckel = ascii.read(basedir / 'data' / 'external' / 'kreckel_pn_candidates.txt')
snr_NGC628_kreckel = ascii.read(basedir / 'data' / 'external' / 'kreckel_snr_candidates.txt')

# convert string to astronomical coordinates
pn_NGC628_kreckel['RA'] = list(map(string_to_ra,pn_NGC628_kreckel['RA']))
pn_NGC628_kreckel['DEC'] = list(map(string_to_dec,pn_NGC628_kreckel['DEC']))
pn_NGC628_kreckel['SkyCoord'] = SkyCoord(pn_NGC628_kreckel['RA'],pn_NGC628_kreckel['DEC'])

# select some subsets (PN from Hermann et al. 2008 or bright sources only)
#pn_herrmann = pn_kreckel[[True if i.endswith('a') else False for i in pn_kreckel['ID']]]

snr_NGC628_kreckel['RA'] = list(map(string_to_ra,snr_NGC628_kreckel['RA']))
snr_NGC628_kreckel['DEC'] = list(map(string_to_dec,snr_NGC628_kreckel['DEC']))
snr_NGC628_kreckel['SkyCoord'] = SkyCoord(snr_NGC628_kreckel['RA'],snr_NGC628_kreckel['DEC'])

# Herrmann+2008

pn_NGC628_herrmann = ascii.read(basedir / 'data' / 'external' / 'Herrmann_2008_pn_candidates.txt')
pn_NGC628_herrmann  = search_table(pn_NGC628_herrmann,'M74')

pn_NGC628_herrmann['RA'] = 12*' '
pn_NGC628_herrmann['DEC'] = 12*' '

for row in pn_NGC628_herrmann:
    row['RA'] = f'{row["RAh"]:02d}h{row["RAm"]:02d}m{row["RAs"]:.02f}s'
    row['DEC'] = f'{row["DEd"]:02d}d{row["DEm"]:02d}m{row["DEs"]:.02f}s'


# IC342, M74, M83, M94, M101    
pn_NGC628_herrmann['SkyCoord'] = SkyCoord(pn_NGC628_herrmann['RA'],pn_NGC628_herrmann['DEC'])
    
pn_NGC5068_herrmann = ascii.read(basedir / 'data' / 'external' / 'Herrmann_2009_pn_candidates.txt',format='csv',delimiter=',')

pn_NGC5068_herrmann['RA'] = list(map(string_to_ra,pn_NGC5068_herrmann['alpha(2000)']))
pn_NGC5068_herrmann['DEC'] = list(map(string_to_dec,pn_NGC5068_herrmann['delta(2000)']))
pn_NGC5068_herrmann['SkyCoord'] = SkyCoord(pn_NGC5068_herrmann['RA'],pn_NGC5068_herrmann['DEC'])

# Ciardullo+2002

pn_NGC3351_ciardullo = ascii.read(basedir / 'data' / 'external' / 'Ciardullo_2002_NGC3351.txt')
pn_NGC3351_ciardullo['RA'] = list(map(string_to_ra,pn_NGC3351_ciardullo['RA']))
pn_NGC3351_ciardullo['DEC'] = list(map(string_to_dec,pn_NGC3351_ciardullo['DEC']))
pn_NGC3351_ciardullo['SkyCoord'] = SkyCoord(pn_NGC3351_ciardullo['RA'],pn_NGC3351_ciardullo['DEC'])

pn_NGC3627_ciardullo = ascii.read(basedir / 'data' / 'external' / 'Ciardullo_2002_NGC3627.txt')
pn_NGC3627_ciardullo['RA'] = list(map(string_to_ra,pn_NGC3627_ciardullo['RA']))
pn_NGC3627_ciardullo['DEC'] = list(map(string_to_dec,pn_NGC3627_ciardullo['DEC']))
pn_NGC3627_ciardullo['SkyCoord'] = SkyCoord(pn_NGC3627_ciardullo['RA'],pn_NGC3627_ciardullo['DEC'])