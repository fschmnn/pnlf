import errno              # handle errors
import logging            # log errors
from pathlib import Path  # filesystem related stuff

from astropy.wcs import WCS
from astropy.io import fits

from pymuse import data_raw, basedir

logger = logging.getLogger(__name__)

class ReadLineMaps:
    '''load a fits file from the MUSEDAP with the line maps
    
    This class reads the emission line maps from the MUSE datapipeline and 
    provides a convienient structure to store the data. It is expected that
    the data resides in the folder `data_raw` and is further subdivided by 
    folders with the names of the observed objects. 
    The fits files are named `GalaxyName_MAPS.fits` and contain multiple 
    extenstions of the form `ExtensionName_FLUX` (you must omit the `_FLUX`
    in the name). This script reads only the extensions that are specified 
    by the appropriate keyword. It also tries to read an extension named
    `ExtensionName_FLUX_ERR` which contains the associated error. The header
    of the first extensions is also used to extract WCS information.
    Lastly it tries to open a file called `GalaxyName_seeing.fits` that 
    contains additional information about the seeing of the different 
    pointings and thus impacts the resulting point spread function (PSF).
    '''
    
    def __init__(self,name,extensions=['OIII5006','HA6562','NII6583','SII6716']):
        '''
        Parameters
        ----------
        
        name : string
            name of the file to be read in. There should be a folder "name"
            in the previously defined "data_raw". This folder should then
            contain a file with name "name_MAPS.fits".
        extensions : list
            list of extensions that are read. Each element must be a valid 
            extension in the previously defined fits file (the actual name
            of the extension is `ExtensionName_FLUX` and 
            `ExtensionName_FLUX_ERR` but they are automaticly completed).
        '''
        
        logger.info(f'loading {name}')

        self.name     = name
        self.filename = data_raw / name / f'{name}_MAPS.fits'
        self.lines    = []
        
        if not self.filename.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.filename)
        
        # make sure lines is a list
        lines = [extensions] if not isinstance(extensions, list) else extensions
        
        with fits.open(self.filename) as hdul:
            for line in lines:

                data   = hdul[f'{line}_FLUX'].data
                # save the main data
                setattr(self,line,data)
                
                # we save the header only if it hasn't been already
                if not hasattr(self,'header'):
                    header = hdul[f'{line}_FLUX'].header
                    setattr(self,'header',header)
                    setattr(self,'wcs',WCS(header))
                    setattr(self,'shape',(header['NAXIS2'],header['NAXIS1']))
                    
                # append to list of available lines
                self.lines.append(line)
                
                err = hdul[f'{line}_FLUX_ERR'].data
                setattr(self,f'{line}_err',err)

        
        # we also load a file with information about the PSF
        seeing_map_file = data_raw / name / f'{name}_seeing.fits'
        if seeing_map_file.is_file():
            try:
                data = fits.getdata(seeing_map_file,extname=f'DATA')
                setattr(self,'PSF',data)
            except:
                logger.warn(f'could not read seeing information for {name}')
        else:
            logger.warn(f'"{name}_seeing.fits" does not exists.')

        # and one where OIII is measured not by fitting
        OIII_bkg_map_file = data_raw / name / f'{name}_oiii_flux.fits'
        if OIII_bkg_map_file.is_file():
            try:
                data = fits.getdata(OIII_bkg_map_file,0)
                setattr(self,'OIII5006_bkg',data)
                # the photometry function expects an associated error
                setattr(self,'OIII5006_bkg_err',self.OIII5006_err)
                self.lines.append('OIII5006_bkg')
            except:
                logger.info(f'could not read alternate OIII map for {name}')
        else:
            logger.warn(f'"{name}_oiii_flux.fits" does not exists.')


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
def split_fits(filename,lines):
    '''
    
    Parameters
    ----------
    filename: a fits file containing lines in multiple extensions
    lines: the lines to save as single files
    '''
    
    # make sure lines is a list
    lines = [lines] if not isinstance(lines, list) else lines
    
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    print(f'splitting file: {filename}')
    
    with fits.open(filename) as hdul:
        for line in lines:
            data = hdul[f'{line}_FLUX']
            data.writeto(f'{line}.fits',overwrite=True)
            
    print('all lines saved')


def MOSAIC(filename):
    '''open the large MOSAIC files in python

    the MOSAIC files contain the full spectral information 
    '''
    with fits.open(filename,memmap=True,mode='denywrite') as hdul:
        wcs = WCS(hdul[1].header)
        data = hdul[1].data
            
        print(data.shape)
        print(hdul[1].header)
        #data = hdul[f'{line}_FLUX']
        #data.writeto(f'{line}.fits',overwrite=True)

# convert pixel coordinates to Ra and Dec   
#positions = np.transpose((self.sources['xcentroid'], self.sources['ycentroid']))
#sky_positions = SkyCoord.from_pixel(positions[:,0],positions[:,1],self.wcs)
#print(sky_positions.to_string(style='hmsdms',precision=2))

if __name__ == '__main__':

    NGC628 = MUSEDAP('NGC628')
    print(NGC628)