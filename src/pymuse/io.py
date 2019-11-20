import errno              # handle errors
import logging            # log errors
from pathlib import Path  # filesystem related stuff

from astropy.wcs import WCS    # handle astronomic coordinates
from astropy.io import fits    # read data from .fits files.

logger = logging.getLogger(__name__)

class ReadLineMaps:
    '''load a fits file from the MUSEDAP with the line maps
    
    This class reads the emission line maps from the MUSE datapipeline
    and provides a convienient structure to store the data. It is 
    expected that the data resides in the specified folder which should
    als obe the name of the object. The folder itself may contain 
    multiple fits files.
    The main fits files are named `GalaxyName_MAPS.fits` and contain multiple 
    extenstions of the form `ExtensionName_FLUX` (you must omit the `_FLUX`
    in the name). This script reads only the extensions that are specified 
    by the appropriate keyword. It also tries to read an extension named
    `ExtensionName_FLUX_ERR` which contains the associated error. The header
    of the first extensions is also used to extract WCS information.
    Lastly it tries to open a file called `GalaxyName_seeing.fits` that 
    contains additional information about the seeing of the different 
    pointings and thus impacts the resulting point spread function (PSF).
    '''
    
    def __init__(self,folder,extensions=['OIII5006','HA6562']):
        '''
        Parameters
        ----------
        
        folder : string
            name of the folder with the data for one object. This folder
            must contain a file with name "FolderName_MAPS.fits" and 
            possibly some additional files

        extensions : list
            list of extensions that are read. Each element must be a valid 
            extension in the previously defined fits file (the actual name
            of the extension is `ExtensionName_FLUX` and 
            `ExtensionName_FLUX_ERR` but they are automaticly completed).
        '''

        # PSF is given in arcsec but we need it in pixel
        _arcsec_to_pixel = 5
        
        self.name     = folder.name
        self.filename = folder / f'{self.name}_MAPS.fits'
        self.lines    = []
        
        logger.info(f'loading {self.name}')

        if not self.filename.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.filename)
        
        # make sure lines is a list
        lines = [extensions] if not isinstance(extensions, list) else extensions
        
        with fits.open(self.filename) as hdul:

            # save the white-light image
            header = hdul[f'FLUX'].header
            setattr(self,'header',header)
            setattr(self,'wcs',WCS(header))
            setattr(self,'shape',(header['NAXIS2'],header['NAXIS1']))
            setattr(self,'whitelight',hdul['FLUX'].data)

            for line in lines:
                # save the main data and the associated error
                setattr(self,line,hdul[f'{line}_FLUX'].data)
                setattr(self,f'{line}_err',hdul[f'{line}_FLUX_ERR'].data)
                # append to list of available lines
                self.lines.append(line)

        # and one where OIII is not measured by fitting
        OIII_bkg_map_file = folder / f'{self.name}_oiii_flux.fits'
        if OIII_bkg_map_file.is_file():
            try:
                # replace the old line maps with the new one
                setattr(self,'OIII5006_old',getattr(self,'OIII5006'))
                data = fits.getdata(OIII_bkg_map_file,0)
                setattr(self,'OIII5006',data)
            except:
                logger.info(f'could not read alternate OIII map for {self.name}')
        else:
            logger.warn(f'"{self.name}_oiii_flux.fits" does not exists.')

        # we also load a file with information about the PSF
        seeing_map_file = folder / f'{self.name}_seeing.fits'
        if seeing_map_file.is_file():
            try:
                data = fits.getdata(seeing_map_file,extname=f'DATA')
                setattr(self,'PSF',data*_arcsec_to_pixel)
            except:
                logger.warn(f'could not read seeing information for {self.name}')
        else:
            logger.warn(f'"{self.name}_seeing.fits" does not exists.')

        logger.info(f'file loaded with {len(self.lines)} extensions')

    def __repr__(self):
        '''create an overview of the available attributes'''
        
        string = ''
        for k,v in self.__dict__.items():
            if type(v) == str:
                string += f'{k}: {v}\n'
            else:
                string += k + '\n'
                
        return string
    
# save lines to individual .fits file
def split_fits(filename,extensions):
    '''
    
    Parameters
    ----------
    filename: 
        a fits file containing lines in multiple extensions

    extensions: 
        the extensions to save as single files
    '''
    
    # make sure lines is a list
    extensions = [extensions] if not isinstance(extensions, list) else extensions
    
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    print(f'splitting {filename} into seperate files')
    
    with fits.open(filename) as hdul:
        for ext in extensions:
            data = hdul[f'{ext}']
            data.writeto(f'{ext}.fits',overwrite=True)
            
    print(f'{len(extensions)} extension(s) saved to seperate file(s)')


class ReadMosaicFiles:

    def __init__(self,filename):
        '''open the large MOSAIC files in python

        the MOSAIC files contain the full spectral information 
        '''

        logger.warning('not yet implemented')
        sys.exit()

        with fits.open(filename,memmap=True,mode='denywrite') as hdul:
            wcs = WCS(hdul[1].header)
            data = hdul[1].data
                
            print(data.shape)
            print(hdul[1].header)
            #data = hdul[f'{line}_FLUX']
            #data.writeto(f'{line}.fits',overwrite=True)
