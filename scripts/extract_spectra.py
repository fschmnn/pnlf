'''
take the catalogue with the PNe and extract the spectra for each of them

the catalogue must contain the following columns

gal_name : name of the galaxy
id : id of the region
x,y : coordinates (same as the datacubes)
fwhm : PSF size in the pointing of the PNe

This script creates a mask with the same size that is used to measure
the flux in Scheuermann+2022 and sums up the integrated spectrum (the
wavelength correction is not applied). Then the background is 
estimated from an annulus.
'''

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats

import warnings

filename = Path('/') / 'home' / 'fscheuer' / 'data' / 'nebulae.fits'
catalogue = Table(fits.getdata(filename))
catalogue = catalogue[(catalogue['type']=='PN') & (catalogue['MOIII']<-4)]


def circular_mask(h, w, center=None, radius=None):
    '''Create a circular mask for a numpy array

    from
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    '''

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def annulus_mask(h, w, center, inner_radius,outer_radius):
    '''Create a circular mask for a numpy array

    from
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    '''

    if inner_radius>outer_radius:
        raise ValueError('inner radius must be smaller than outer radius')

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = (inner_radius < dist_from_center) & (dist_from_center <= outer_radius)

    return mask

wlen   = []
fluxes = []
bkgs   = []
names  = []
ids    = []
gal_name = None
for row in catalogue:
    
    if gal_name != row['gal_name']:
        gal_name = row['gal_name']
        print(f'reading {gal_name}')
        cube_filename = Path('/')/'data'/'MUSE'/'DR2.1'/'native'/'datacubes'/f'{gal_name}_DATACUBE_FINAL_WCS_Pall_mad.fits'
        with fits.open(cube_filename, memmap=True, mode='denywrite') as hdul:
            data_cube   = hdul[1].data
            cube_header = hdul[1].header
    
    x,y = row[['x','y']]
    r = 2.5 * row['fwhm'] / 2 
    r_in  = 4 * row['fwhm'] / 2 
    r_out = np.sqrt(5*r**2+r_in**2)
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # there will be NaNs in the subcube that is used for the sigma clipping
        # astropy will issue a warning which we ignore in this enviornment
        circle  = circular_mask(*data_cube.shape[1:],(x,y),r)
        annulus = annulus_mask(*data_cube.shape[1:],(x,y),r_in,r_out) 
        _, bkg, _ = sigma_clipped_stats(data_cube[...,annulus],axis=1)
    
    
    names.append(gal_name)
    ids.append(row['id'])
    fluxes.append(np.sum(data_cube[...,circle],axis=1))  
    bkgs.append(bkg * np.sum(circle))
    wlen.append(np.linspace(cube_header['CRVAL3'],cube_header['CRVAL3']+cube_header['NAXIS3']*cube_header['CD3_3'],cube_header['NAXIS3']))


spectra = Table(data=[names,ids,wlen,fluxes,bkgs],
                names=['gal_name','id','wlen','flux','bkg'])

filename = Path('/') / 'home' / 'fscheuer' / 'data' / 'PN_spectra.fits'
hdu = fits.BinTableHDU(spectra,name='spectra')
hdu.writeto(filename,overwrite=True)