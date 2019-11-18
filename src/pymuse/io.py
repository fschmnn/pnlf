import errno
import os
import logging

from astropy.wcs import WCS
from astropy.io import fits

logger = logging.getLogger(__name__)
data_folder = os.path.join('d:',os.sep,'downloads','MUSEDAP')

class MUSEDAP:
    '''load a fits file from the MUSEDAP


    '''
    
    def __init__(self,name,lines=['OIII5006','HA6562','NII6583','SII6716']):
        '''
        Parameters
        ----------
        
        name : string
            name of the file to be read in. There should be a folder "name"
            in the previously defined "data_folder". This folder should then
            contain a file with name "name_MAPS.fits".
        lines : list
            list of lines that are used. Each element must be a valid 
            extension in the previously defined fits file (the actual name
            of the extension is "line_FLUX" and "line_FLUX_ERR" but they are
            automaticly completed).
        '''
        
        logger.info(f'loading {name}')

        self.name     = name
        self.filename = os.path.join(data_folder,name,f'{name}_MAPS.fits')
        self.lines    = []
        
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.filename)
        
        # make sure lines is a list
        lines = [lines] if not isinstance(lines, list) else lines
        
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
        seeing_map_file = os.path.join(data_folder,name,f'{name}_seeing.fits')
        if not os.path.isfile(seeing_map_file):
            logger.warn(f'"{name}_seeing.fits" does not exists.')
        else:
            try:
                data,header = fits.getdata(seeing_map_file,extname=f'DATA',header=True)
                setattr(self,'PSF',data)
            except:
                logger.warn(f'could not read seeing information for {name}')
            
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
    
filename = os.path.join(data_folder,'NGC628','NGC628_MAPS.fits')
lines = ['OIII5006','HA6562','NII6583','SII6716']

split_fits(filename,lines)


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

    data_folder = os.path.join('d:',os.sep,'downloads','MUSEDAP')
    NGC628 = MUSEDAP('NGC628')
    print(NGC628)