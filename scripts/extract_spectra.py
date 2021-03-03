from astropy.nddata import NDData
from astropy.io import fits 
import numpy as np

from pathlib import Path 


data_path = Path('/') / 'data'

filename = data_ext  / 'MUSE' / 'DR2.0' / 'native' / 'datacubes' / f'{name}_MAPS.fits'
with fits.open(filename) as hdul:
    Halpha = NDData(data=hdul['HA6562_FLUX'].data,
                    mask=np.isnan(hdul['HA6562_FLUX'].data),
                    meta=hdul['HA6562_FLUX'].header,
                    wcs=WCS(hdul['HA6562_FLUX'].header))
    OIII = NDData(data=hdul['OIII5006_FLUX'].data,
                    mask=np.isnan(hdul['OIII5006_FLUX'].data),
                    meta=hdul['OIII5006_FLUX'].header,
                    wcs=WCS(hdul['OIII5006_FLUX'].header)) 

filename = data_ext / 'Products' / 'Nebulae_catalogue' / 'Nebulae_catalogue_v1' /'spatial_masks'/f'{name}_HIIreg_mask.fits'
with fits.open(filename) as hdul:
    nebulae_mask = NDData(hdul[0].data-1,mask=Halpha.mask,meta=hdul[0].header,wcs=WCS(hdul[0].header))
    nebulae_mask.data[nebulae_mask.data==-1] = np.nan


filename = data_ext/'MUSE'/'DR2.0'/'native'/'datacubes'/f'{name}_DATACUBE_FINAL_WCS_Pall_mad.fits'
with fits.open(filename , memmap=True, mode='denywrite') as hdul:
    data_cube   = hdul[1].data
    cube_header = hdul[1].header


print('this worked')


