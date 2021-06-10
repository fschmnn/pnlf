from pathlib import Path

import numpy as np 

from astropy.io import ascii 
from astropy.coordinates import SkyCoord
from astropy.table import vstack, MaskedColumn
import astropy.units as u

from .auxiliary import search_table, Distance

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

pn_NGC628_kreckel = ascii.read(basedir / 'data' / 'external' / 'kreckel_pn_candidates.txt',format='csv',delimiter=';')
snr_NGC628_kreckel = ascii.read(basedir / 'data' / 'external' / 'kreckel_snr_candidates.txt',format='csv',delimiter=';')

# convert string to astronomical coordinates
pn_NGC628_kreckel['RA'] = list(map(string_to_ra,pn_NGC628_kreckel['RA']))
pn_NGC628_kreckel['DEC'] = list(map(string_to_dec,pn_NGC628_kreckel['DEC']))
pn_NGC628_kreckel.meta['reference'] = 'Kreckel+2017'
pn_NGC628_kreckel.meta['bibcode'] = '2017ApJ...834..174K'

pn_NGC628_kreckel['OIII'] = 10**((pn_NGC628_kreckel['mOIII']+13.74)/-2.5)
pn_NGC628_kreckel['Ha/NII'][pn_NGC628_kreckel['Ha/NII']=='L'] = '1e-30'
pn_NGC628_kreckel['Ha/SII'][pn_NGC628_kreckel['Ha/SII']=='L'] = '1e-30'
pn_NGC628_kreckel['OIII/Ha'] = [float(x.replace('>','')) for x in pn_NGC628_kreckel['OIII/Ha']]
pn_NGC628_kreckel['Ha/NII'] = [float(x.replace('>','')) for x in pn_NGC628_kreckel['Ha/NII']]
pn_NGC628_kreckel['Ha/SII'] = [float(x.replace('>','')) for x in pn_NGC628_kreckel['Ha/SII']]
pn_NGC628_kreckel['R'] = pn_NGC628_kreckel['OIII/Ha']/ (1+pn_NGC628_kreckel['Ha/NII'])
pn_NGC628_kreckel['dR'] = MaskedColumn(np.zeros(len(pn_NGC628_kreckel)),mask=len(pn_NGC628_kreckel)*[True])

# select some subsets (PN from Hermann et al. 2008 or bright sources only)
#pn_herrmann = pn_kreckel[[True if i.endswith('a') else False for i in pn_kreckel['ID']]]

snr_NGC628_kreckel['RA'] = list(map(string_to_ra,snr_NGC628_kreckel['RA']))
snr_NGC628_kreckel['DEC'] = list(map(string_to_dec,snr_NGC628_kreckel['DEC']))
snr_NGC628_kreckel.meta['reference'] = 'Kreckel+2017'
snr_NGC628_kreckel.meta['bibcode'] = '2017ApJ...834..174K'

snr_NGC628_kreckel['OIII'] = 10**((snr_NGC628_kreckel['mOIII']+13.74)/-2.5)
snr_NGC628_kreckel['Ha/NII'][snr_NGC628_kreckel['Ha/NII']=='L'] = '1e-30'
snr_NGC628_kreckel['Ha/SII'][snr_NGC628_kreckel['Ha/SII']=='L'] = '1e-30'
snr_NGC628_kreckel['OIII/Ha'] = [float(x.replace('>','')) for x in snr_NGC628_kreckel['OIII/Ha']]
snr_NGC628_kreckel['Ha/NII'] = [float(x.replace('>','')) for x in snr_NGC628_kreckel['Ha/NII']]
snr_NGC628_kreckel['Ha/SII'] = [float(x.replace('<','')) for x in snr_NGC628_kreckel['Ha/SII']]
snr_NGC628_kreckel['R'] = snr_NGC628_kreckel['OIII/Ha']/ (1+snr_NGC628_kreckel['Ha/NII'])
snr_NGC628_kreckel['dR'] = MaskedColumn(np.zeros(len(snr_NGC628_kreckel)),mask=len(snr_NGC628_kreckel)*[True])
snr_NGC628_kreckel['ID'] = snr_NGC628_kreckel['ID'].astype('str')

pn_NGC628_kreckel['type'] = 'PN'
snr_NGC628_kreckel['type'] = 'SNR'
NGC628_kreckel = vstack([pn_NGC628_kreckel,snr_NGC628_kreckel])

# Herrmann+2008

pn_NGC628_herrmann = ascii.read(basedir / 'data' / 'external' / 'Herrmann_2008_pn_candidates.txt')
pn_NGC628_herrmann  = search_table(pn_NGC628_herrmann,'M74')

pn_NGC628_herrmann['RA'] = 12*' '
pn_NGC628_herrmann['DEC'] = 12*' '

for row in pn_NGC628_herrmann:
    row['RA'] = f'{row["RAh"]:02d}h{row["RAm"]:02d}m{row["RAs"]:.02f}s'
    row['DEC'] = f'{row["DEd"]:02d}d{row["DEm"]:02d}m{row["DEs"]:.02f}s'


# IC342, M74, M83, M94, M101    
pn_NGC628_herrmann.meta['reference'] = 'Herrmann+2008'
pn_NGC628_herrmann.meta['bibcode'] = '2008ApJ...683..630H'
pn_NGC628_herrmann.rename_column('m5007','mOIII')
pn_NGC628_herrmann.rename_column('e_R','dR')
pn_NGC628_herrmann['OIII'] = 10**((pn_NGC628_herrmann['mOIII']+13.74)/-2.5)

# combine all tables for NGC628
pn_NGC628_kreckel['source'] = 'Kreckel PN' 
snr_NGC628_kreckel['source'] = 'Kreckel SNR'
pn_NGC628_herrmann['source'] = 'Herrmann PN'

NGC628 = vstack([pn_NGC628_kreckel[['source','ID','RA','DEC','mOIII','R','dR']],snr_NGC628_kreckel[['source','ID','RA','DEC','mOIII','R','dR']],pn_NGC628_herrmann[['source','ID','RA','DEC','mOIII','R','dR']]])

pn_NGC628_kreckel['SkyCoord'] = SkyCoord(pn_NGC628_kreckel['RA'],pn_NGC628_kreckel['DEC'])
snr_NGC628_kreckel['SkyCoord'] = SkyCoord(snr_NGC628_kreckel['RA'],snr_NGC628_kreckel['DEC'])
pn_NGC628_herrmann['SkyCoord'] = SkyCoord(pn_NGC628_herrmann['RA'],pn_NGC628_herrmann['DEC'])
NGC628['SkyCoord'] = SkyCoord(NGC628['RA'],NGC628['DEC'])
NGC628_kreckel['SkyCoord'] = SkyCoord(NGC628_kreckel['RA'],NGC628_kreckel['DEC'])


pn_NGC5068_herrmann = ascii.read(basedir / 'data' / 'external' / 'Herrmann_NGC5068_pn_candidates.txt',format='csv',delimiter=',')

pn_NGC5068_herrmann['RA'] = list(map(string_to_ra,pn_NGC5068_herrmann['alpha(2000)']))
pn_NGC5068_herrmann['DEC'] = list(map(string_to_dec,pn_NGC5068_herrmann['delta(2000)']))
pn_NGC5068_herrmann['SkyCoord'] = SkyCoord(pn_NGC5068_herrmann['RA'],pn_NGC5068_herrmann['DEC'])
pn_NGC5068_herrmann.rename_column('m_5007','mOIII')
pn_NGC5068_herrmann.meta['reference'] = 'Herrmann+2008'
pn_NGC5068_herrmann.meta['bibcode'] = '2008ApJ...683..630H'


# Ciardullo+2002

pn_NGC3351_ciardullo = ascii.read(basedir / 'data' / 'external' / 'Ciardullo_2002_NGC3351.txt')
pn_NGC3351_ciardullo['RA'] = list(map(string_to_ra,pn_NGC3351_ciardullo['RA']))
pn_NGC3351_ciardullo['DEC'] = list(map(string_to_dec,pn_NGC3351_ciardullo['DEC']))
pn_NGC3351_ciardullo['SkyCoord'] = SkyCoord(pn_NGC3351_ciardullo['RA'],pn_NGC3351_ciardullo['DEC'])
pn_NGC3351_ciardullo.rename_column('OIII','mOIII')
pn_NGC3351_ciardullo.meta['reference'] = 'Ciardullo+2002'
pn_NGC3351_ciardullo.meta['bibcode'] = '2002ApJ...577...31C'


pn_NGC3627_ciardullo = ascii.read(basedir / 'data' / 'external' / 'Ciardullo_2002_NGC3627.txt')
pn_NGC3627_ciardullo['RA'] = list(map(string_to_ra,pn_NGC3627_ciardullo['RA']))
pn_NGC3627_ciardullo['DEC'] = list(map(string_to_dec,pn_NGC3627_ciardullo['DEC']))
pn_NGC3627_ciardullo['SkyCoord'] = SkyCoord(pn_NGC3627_ciardullo['RA'],pn_NGC3627_ciardullo['DEC'])
pn_NGC3627_ciardullo.rename_column('OIII','mOIII')
pn_NGC3627_ciardullo.meta['reference'] = 'Ciardullo+2002'
pn_NGC3627_ciardullo.meta['bibcode'] = '2002ApJ...577...31C'

# Sonbas+2010

snr_NGC0628_sonbas = ascii.read(basedir / 'data' / 'external' / 'Sonbas_snr_NGC0628.txt',format='csv',delimiter=' ',data_start=1)
snr_NGC0628_sonbas['SkyCoord'] = SkyCoord(snr_NGC0628_sonbas['RA'],snr_NGC0628_sonbas['Dec'])


# Roth+2021

pn_NGC0628_roth = ascii.read(basedir/'data'/'external'/'roth+2021_pn_candidates.txt')

pn_NGC0628_roth.rename_column('err','dmOIII')
pn_NGC0628_roth['RA'] = list(map(string_to_ra,pn_NGC0628_roth['RA']))
pn_NGC0628_roth['DEC'] = list(map(string_to_dec,pn_NGC0628_roth['DEC']))
pn_NGC0628_roth['SkyCoord'] = SkyCoord(pn_NGC0628_roth['RA'],pn_NGC0628_roth['DEC'])
pn_NGC0628_roth['pointing'] = [x.split('-')[0] for x in pn_NGC0628_roth['ID']]
pn_NGC0628_roth.add_index('No')
pn_NGC0628_roth['OIII'] = 10**((pn_NGC0628_roth['mOIII']+13.74)/-2.5)

offset = {'P1': (1.4871026828755427, 1.5842446858756842),
 'P2': (1.3007895927116229, 4.258799715932929),
 'P3b': (1.0871248490453884, 3.6859363973201997),
 'P4': (1.1710318786060663, 4.290050872730859),
 'P5': (0.7821342597616665, 3.067700721973635),
 'P6': (1.6090193254676588, 1.9130460388224684),
 'P7': (2.0721813367113606, 1.9567185746199387),
 'P8': (1.2715106777584466, 4.308244511807938),
 'P9': (0.876431069592869, 3.639543854448167),
 'P12': (1.3801150958283392, 4.310376779358592),
 'K1': (0.2384843703710526, 2.5655582941153243),
 'K2': (0.49506048394801855, 2.0769182209339583)}

pn_NGC0628_roth['SkyCoord_offset'] = pn_NGC0628_roth['SkyCoord']
for k,v in offset.items():
    pn_NGC0628_roth['SkyCoord'][pn_NGC0628_roth['pointing']==k] = pn_NGC0628_roth['SkyCoord'][pn_NGC0628_roth['pointing']==k].directional_offset_by(v[1]*u.rad, v[0]*u.arcsec)  

# this dict was originally used to calculate the offsets
matches = {
    'P1': ('PN27',947),
    'P2': ('PN5',1968),
    'P3b': ('PN9',1025),
    'P4': ('PN15',2024), # really bad
    'P5': ('PN39',1524),
    'P6': ('PN7',1968),
    'P7': ('PN1',56),
    'P8': ('PN23',557),
    'P9': ('PN25',752),
    'P12': ('PN41',1322),
    'K1': ('PN17',1755),
    'K2': ('PN12',1291),
}

