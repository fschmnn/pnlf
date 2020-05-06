import errno              # handle errors
import logging            # log errors
import os
from pathlib import Path  # filesystem related stuff

import numpy as np
from astropy.wcs import WCS    # handle astronomic coordinates
from astropy.io import fits    # read data from .fits files.
from astropy.io import ascii
from astropy.coordinates import SkyCoord, match_coordinates_sky, Angle

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
    
    def __init__(self,folder,extensions=['OIII5006','HA6562','NII6583','SII6716']):
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
            setattr(self,'V_STARS',hdul['V_STARS'].data)
            setattr(self,'stellar_mass',hdul['STELLAR_MASS_DENSITY'].data)
            setattr(self,'stellar_mass_err',hdul['STELLAR_MASS_DENSITY_err'].data)
            setattr(self,'Ebv_stars',hdul['EBV_STARS'].data)
            setattr(self,'whitelight',hdul['FLUX'].data)
            setattr(self,'whitelight_err',hdul['SNR'].data)

            for line in lines:
                # save the main data and the associated error
                setattr(self,line,hdul[f'{line}_FLUX'].data)
                setattr(self,f'{line}_err',hdul[f'{line}_FLUX_ERR'].data)                
                setattr(self,f'{line}_SIGMA',np.sqrt(hdul[f'{line}_SIGMA'].data**2 - hdul[f'{line}_SIGMA_CORR'].data**2))

                # append to list of available lines
                self.lines.append(line)

        # try to load some additional maps
        star_mask_file = folder / f'{self.name}_starmask.fits'
        if star_mask_file.is_file():
            with fits.open(star_mask_file) as hdul:
                self.star_mask = hdul[0].data
        else:
            logger.warning(f'no starmask available')

        av_file = folder / f'{self.name}_AV.fits'
        if av_file.is_file():    
            with fits.open(av_file) as hdul:
                self.Av = hdul[0].data
        else:
            logger.warning(f'no AV map available')
    

        # and one where OIII is not measured by fitting
        OIII_bkg_map_file = folder / f'{self.name}_oiii_flux.fits'
        if OIII_bkg_map_file.is_file():
            try:
                logger.info(f'replacing OIII5006 map')
                # replace the old line maps with the new one
                setattr(self,'OIII5006_DAP',getattr(self,'OIII5006'))
                setattr(self,'OIII5006_DAP_err',getattr(self,'OIII5006_err'))
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



def write_LaTeX(table,galaxy,filename):
    '''write the table to a file

    this will create a `.tex` file and a machine readable file

    to read this file use
    f = basedir / 'data' / 'catalogues' / f'{galaxy.name}_{typ}_candidates.txt'
    ascii.read(f,format='fixed_width_two_line',position_char='=')
    '''

    threshold = '0.7"'

    table['SkyCoord'] = SkyCoord.from_pixel(table['x'],table['y'],galaxy.wcs)
    table['RA'],table['DEC'] = zip(*[x.split(' ') for x in table['SkyCoord'].to_string(style='hmsdms',precision=2)])

    if galaxy.name =='NGC628':
        from .load_references import NGC628
        cat = {'Kreckel PN':'a','Herrmann PN':'b','Kreckel SNR':'c'}

        ID, angle, Quantity  = match_coordinates_sky(NGC628['SkyCoord'],table['SkyCoord'])

        table['match'] = np.empty(len(table),dtype='U12')
        for i,a,row in zip(ID,angle,NGC628):
            if a.__lt__(Angle('0.8"')):
                table['match'][i] += cat[row['source']]
                
                if False:
                    table['match'][i] += ' ' + row['ID']

    for typ in ['PN','SNR','HII']:

        if typ == 'PN':
            n = 'Planetary Nebula'
        if typ == 'SNR':
            n = 'Supernova Remnants'
        if typ == 'HII':
            n = '\\HII-regions'    


        latexdict = {'tabletype': 'table*',
        'header_start': '\\toprule\\toprule',
        'header_end': '\\midrule',
        'data_end': '\\bottomrule',
        'caption': f'{n} Identifications',
        'units': {'R.A.':'(J2000)','Dec.':'(J2000)','$m_\\OIII$':'mag','d$m_\\OIII$':'mag','$\sigma_V$':'\\si{\\km \per \\second}'},
        'preamble': '\\centering',
        'tablefoot': f'\\label{{tbl:{typ}_Identifications}}'
                    }

        tbl_out = table[table['type']==typ]

        tbl_out.sort('mOIII')

        tbl_out['OIII/Ha']   = np.empty(len(tbl_out),dtype='U8')
        tbl_out['Ha/NII']    = np.empty(len(tbl_out),dtype='U8') 
        tbl_out['Ha/SII']    = np.empty(len(tbl_out),dtype='U8')
        tbl_out['d(OIII/Ha)']= np.empty(len(tbl_out),dtype='U8')
        tbl_out['d(Ha/NII)'] = np.empty(len(tbl_out),dtype='U8')
        tbl_out['d(Ha/SII)'] = np.empty(len(tbl_out),dtype='U8')

        # compare to existing studies

        names = []
        for i,row in enumerate(tbl_out):
            name = str(i+1)
            if galaxy.name == 'NGC628':
                name += row['match']
            names.append(name)        
        tbl_out['name'] = names

        # calculate line ratios with limits (> sign)
        for i,row in enumerate(tbl_out):
            if not row['HA6562_detection']:
                row['OIII/Ha'] += '>'
            row['OIII/Ha'] += f"{row['OIII5006'] / row['HA6562']:.2f}"
            row['d(OIII/Ha)'] = f"{row['OIII5006'] / row['HA6562'] * np.sqrt( (row['HA6562_err'] / row['HA6562'])**2 + (row['OIII5006_err'] / row['OIII5006'])**2):.2f}"

            if not row['HA6562_detection'] and not row['NII6583_detection']:
                row['Ha/NII'] = '...'
                row['d(Ha/NII)'] = '...'
            elif not row['HA6562_detection']:
                row['Ha/NII'] = '<'
                row['Ha/NII'] += f"{row['HA6562'] / row['NII6583']:.2f}"
                row['d(Ha/NII)'] = f"{row['HA6562'] / row['NII6583'] * np.sqrt( (row['NII6583_err'] / row['NII6583'])**2 + (row['HA6562_err'] / row['HA6562'])**2):.2f}"
            elif not row['NII6583_detection']:
                row['Ha/NII'] = '>'
                row['Ha/NII'] += f"{row['HA6562'] / row['NII6583']:.2f}"
                row['d(Ha/NII)'] = f"{row['HA6562'] / row['NII6583'] * np.sqrt( (row['NII6583_err'] / row['NII6583'])**2 + (row['HA6562_err'] / row['HA6562'])**2):.2f}"
            else:
                row['Ha/NII'] += f"{row['HA6562'] / row['NII6583']:.2f}"
                row['d(Ha/NII)'] = f"{row['HA6562'] / row['NII6583'] * np.sqrt( (row['NII6583_err'] / row['NII6583'])**2 + (row['HA6562_err'] / row['HA6562'])**2):.2f}"

            if not row['HA6562_detection'] and not row['SII6716_detection']:
                row['Ha/SII'] = '...'
                row['d(Ha/SII)'] = '...'
            elif not row['HA6562_detection']:
                row['Ha/SII'] += '<'
                row['Ha/SII'] += f"{row['HA6562'] / row['SII6716']:.2f}"
                row['d(Ha/SII)'] = f"{row['HA6562'] / row['SII6716'] * np.sqrt( (row['HA6562_err'] / row['HA6562'])**2 + (row['SII6716_err'] / row['SII6716'])**2):.2f}"
            elif not row['SII6716_detection']:
                row['Ha/SII'] += '>'
                row['Ha/SII'] += f"{row['HA6562'] / row['SII6716']:.2f}"
                row['d(Ha/SII)'] = f"{row['HA6562'] / row['SII6716'] * np.sqrt( (row['HA6562_err'] / row['HA6562'])**2 + (row['SII6716_err'] / row['SII6716'])**2):.2f}"
            else:
                row['Ha/SII'] += f"{row['HA6562'] / row['SII6716']:.2f}"
                row['d(Ha/SII)'] = f"{row['HA6562'] / row['SII6716'] * np.sqrt( (row['HA6562_err'] / row['HA6562'])**2 + (row['SII6716_err'] / row['SII6716'])**2):.2f}"


        tbl_out['mOIII'].info.format = '%.2f' 
        tbl_out['dmOIII'].info.format = '%.2f' 
        tbl_out['v_SIGMA'].info.format = '%.2f' 

        tbl_out.rename_columns(['name','RA','DEC','v_SIGMA'],['ID','R.A.','Dec.','sigmaV'])
        tbl_out = tbl_out[['ID','R.A.','Dec.','mOIII','dmOIII','OIII/Ha','d(OIII/Ha)',
                           'Ha/NII','d(Ha/NII)','Ha/SII','d(Ha/SII)','sigmaV']]

        with open((filename / f'{galaxy.name}_{typ}_candidates').with_suffix('.txt'),'w',newline='\n') as f:
            ascii.write(tbl_out,f,format='fixed_width_two_line',overwrite=True,delimiter_pad=' ',position_char='=')

        
        mOIII  = []
        OIIIHA = []
        HANII  = []
        HASII  = []
        for row in tbl_out:
            if row["mOIII"] == '...':
                mOIII.append('...')
            else:
                mOIII.append(f'{row["mOIII"]:.2f} $\pm$ {row["dmOIII"]:.2f}')
            if row["OIII/Ha"] == '...':
                OIIIHA.append('...')
            else:
                OIIIHA.append(f'{row["OIII/Ha"]} $\pm$ {row["d(OIII/Ha)"]}')
            if row["Ha/NII"] == '...':
                HANII.append('...')
            else:
               HANII.append(f'{row["Ha/NII"]} $\pm$ {row["d(Ha/NII)"]}')
            if row["Ha/SII"] == '...':
                HASII.append('...')
            else:
               HASII.append(f'{row["Ha/SII"]} $\pm$ {row["d(Ha/SII)"]}')
        
        tbl_out['$m_\\OIII$'] = mOIII
        tbl_out['$\\OIII/\\HA$'] = OIIIHA
        tbl_out['$\\HA/\\NII$'] = HANII
        tbl_out['$\\HA/\\SII$'] = HASII
        tbl_out.rename_column('sigmaV',f'$\sigma_V$')
        tbl_out = tbl_out[['ID','R.A.','Dec.','$m_\\OIII$','$\\OIII/\\HA$',
                           '$\\HA/\\NII$','$\\HA/\\SII$',f'$\sigma_V$']]

        with open((filename / f'{galaxy.name}_{typ}_candidates').with_suffix('.tex'),'w',newline='\n') as f:
            ascii.write(tbl_out,f,Writer=ascii.Latex, latexdict=latexdict,overwrite=True)


        # shorten column names for machine readable table
        #for col in tbl_out.colnames:
        #    tbl_out.rename_column(col,col.translate({ord(s): None for s in '$\_'}))


        
    logger.info('table saved to files')


'''
        newnames = ['$m_\\OIII$','d$m_\\OIII$','$\\OIII/\\HA$','d$(\\OIII/\\HA)$',
                '$\\HA/\\NII$','d$(\\HA/\\NII)$','$\\HA/\\SII$','d$(\\HA/\\SII)$','$\sigma_V$']

        oldnames = ['mOIII','dmOIII','OIII/Ha','d(OIII/Ha)',
                'Ha/NII','d(Ha/NII)','Ha/SII','d(Ha/SII)','v_SIGMA']
'''

