import errno              # handle errors
import logging            # log errors
import os
import sys
from pathlib import Path  # filesystem related stuff

import numpy as np
from astropy.wcs import WCS    # handle astronomic coordinates
from astropy.io import fits    # read data from .fits files.
from astropy.io import ascii
from astropy.coordinates import SkyCoord, match_coordinates_sky, Angle
from reproject import reproject_interp

from astropy.visualization import simple_norm
import matplotlib.pyplot as plt

from .constants import two_column, arcsec_to_pixel


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
    
    def __init__(self,folder,name,extensions=['HB4861','OIII5006','HA6562','NII6583','SII6716','SII6730'],**kwargs):
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
        
        logger.info(f'loading {name}')

        self.name     = name
        filename = folder/f'{name}_MAPS.fits'
        if not filename.is_file():
            files = [x for x in folder.iterdir() if name in x.stem]
            if len(files)>1:
                logger.warning(f'more than one file with name "{name}" in folder')
                self.filename = files[0]
            elif len(files)==0:
                logger.warning(f'no file found with name "{name}"')
                sys.exit(0)
            else:
                logger.info(f'using file "{files[0].stem}"')
                self.filename = files[0]
        else: 
            self.filename = filename
        self.lines    = []
        
        for k,v in kwargs.items():
            setattr(self,k,v)

        if not self.filename.is_file():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.filename)
        
        # make sure lines is a list
        lines = [extensions] if not isinstance(extensions, list) else extensions
        
        #==============================================================
        # load the main DAP products
        #==============================================================
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
                setattr(self,f'{line}_SIGMA_ERR',hdul[f'{line}_SIGMA_ERR'])

                # append to list of available lines
                self.lines.append(line)

        #==============================================================
        # load auxiliary maps
        #==============================================================
        
        # and one where OIII is not measured by fitting
        OIII_bkg_map_file = folder.parent / 'AUXILIARY' / 'oiii_from_cubes' / f'{self.name}_oiii_flux.fits'
        if OIII_bkg_map_file.is_file():
            logger.info(f'replacing OIII5006 map')
            # replace the old line maps with the new one
            setattr(self,'OIII5006_DAP',getattr(self,'OIII5006'))
            setattr(self,'OIII5006_DAP_err',getattr(self,'OIII5006_err'))
            data = fits.getdata(OIII_bkg_map_file,0)
            setattr(self,'OIII5006',data)
        else:
            logger.warn(f'"{self.name}_oiii_flux.fits" does not exists.')

        # star mask
        star_mask_file = folder.parent / 'AUXILIARY' / 'starmasks' / f'{self.name}_starmask.fits'
        seeing_map_file = folder.parent / 'AUXILIARY' / 'seeing_maps' / f'{self.name}_seeing.fits'

        for filename, description in zip([star_mask_file,seeing_map_file],["star_mask","PSF"]):

            if filename.is_file():
                with fits.open(filename) as hdul:
                    data   = hdul[0].data
                    
                    if self.shape != data.shape: 
                        logger.warning(f'{description} map has different shape. Reprojecting')
                        data,_ = reproject_interp(hdul,self.header)

                        # star_mask ist 0 or 1 (even for interpolated pixels)
                        if description=='star_mask':
                            data = np.round(data,0)
                        elif description=='PSF':
                            data = np.round(data,2)

                    setattr(self,description,data)

            else:
                logger.warning(f'no {description} available')

        if not hasattr(self,'star_mask'):
            self.star_mask = np.zeros_like(self.OIII5006)

        if not hasattr(self,'PSF'):
            # for DR2 galaxies where no PSF data exists we assume FWHM=1" for all pointings
            logger.warning('creating 1" seeing map')
            self.PSF = np.ones_like(self.OIII5006)
            self.PSF[np.isnan(self.OIII5006)] = np.nan
        self.PSF *= arcsec_to_pixel 
        
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

    def plot(self,line):
        '''plot a single emission line'''

        if not hasattr(self,line):
            raise AttributeError(f'Object has no map {line}')
        
        data = getattr(self,line)

        fig = plt.figure(figsize=(two_column,two_column/1.618))
        ax  = fig.add_subplot(projection=self.wcs)

        norm = simple_norm(data,clip=False,percent=99)
        ax.imshow(data,norm=norm)
        plt.show()
        
        #return fig 

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


def write_table(table,name,filename):
    '''write the table to a file

    this will create a `.tex` file and a machine readable file

    to read this file use
    f = basedir / 'data' / 'catalogues' / f'{galaxy.name}_{typ}_candidates.txt'
    ascii.read(f,format='fixed_width_two_line',position_char='=')
    '''

    threshold = '0.7"'

    table['RA'],table['DEC'] = zip(*[x.split(' ') for x in table['SkyCoord'].to_string(style='hmsdms',precision=2)])
    table['Galaxy'] = name

    if name =='NGC0628':
        logging.info('comparing to existing studies')
        from .load_references import NGC628
        cat = {'Kreckel PN':'K17;','Herrmann PN':'H08;','Kreckel SNR':'K17s;'}
        ID, angle, Quantity  = match_coordinates_sky(NGC628['SkyCoord'],table['SkyCoord'])
        table['match'] = np.empty(len(table),dtype='U12')
        for i,a,row in zip(ID,angle,NGC628):
            if a.__lt__(Angle('0.8"')):
                table['match'][i] += cat[row['source']]

    if name =='NGC3351':
        logging.info('comparing to existing studies')
        from .load_references import pn_NGC3351_ciardullo
        ID, angle, Quantity  = match_coordinates_sky(pn_NGC3351_ciardullo['SkyCoord'],table['SkyCoord'])
        table['match'] = np.empty(len(table),dtype='U12')
        for i,a,row in zip(ID,angle,pn_NGC3351_ciardullo):
            if a.__lt__(Angle('0.8"')):
                table['match'][i] += 'C02;'

    if name =='NGC3627':
        logging.info('comparing to existing studies')
        from .load_references import pn_NGC3627_ciardullo
        ID, angle, Quantity  = match_coordinates_sky(pn_NGC3627_ciardullo['SkyCoord'],table['SkyCoord'])
        table['match'] = np.empty(len(table),dtype='U12')
        for i,a,row in zip(ID,angle,pn_NGC3627_ciardullo):
            if a.__lt__(Angle('0.8"')):
                table['match'][i] += 'C02;'

    if name =='NGC5068':
        logging.info('comparing to existing studies')
        from .load_references import pn_NGC5068_herrmann
        ID, angle, Quantity  = match_coordinates_sky(pn_NGC5068_herrmann['SkyCoord'],table['SkyCoord'])
        table['match'] = np.empty(len(table),dtype='U12')
        for i,a,row in zip(ID,angle,pn_NGC5068_herrmann):
            if a.__lt__(Angle('0.8"')):
                table['match'][i] += 'H08;'

    for col in ['OIII5006','HA6562','NII6583','SII']:
        table[col][np.where(table[col]<table[f'{col}_err'])] = table[f'{col}_err'][np.where(table[col]<table[f'{col}_err'])] 

    table['OIII/Ha']   = table['OIII5006']/table['HA6562']
    table['d(OIII/Ha)']= table['OIII5006'] / table['HA6562'] * np.sqrt( (table['HA6562_err'] / table['HA6562'])**2 + (table['OIII5006_err'] / table['OIII5006'])**2)

    table['Ha/NII']    = table['HA6562']/table['NII6583']
    table['d(Ha/NII)'] = table['HA6562'] / table['NII6583'] * np.sqrt( (table['NII6583_err'] / table['NII6583'])**2 + (table['HA6562_err'] / table['HA6562'])**2)

    table['Ha/SII']    = table['HA6562']/table['SII']
    table['d(Ha/SII)'] = table['HA6562'] / table['SII'] * np.sqrt( (table['SII_err'] / table['SII'])**2 + (table['HA6562_err'] / table['HA6562'])**2)

    for col in ['mOIII','dmOIII','v_SIGMA','OIII/Ha','d(OIII/Ha)','Ha/NII','d(Ha/NII)','Ha/SII','d(Ha/SII)']:
        table[col].info.format = '%.2f' 

    # 
    for typ in ['PN','SNR']:

        tbl_out = table[table['type']==typ]

        if typ == 'PN':
            n = 'Planetary Nebula'
        if typ == 'SNR':
            n = 'Supernova Remnants'  

        tbl_out.sort('mOIII')
        tbl_out['name'] = np.arange(1,len(tbl_out)+1)

        # add marker for existing study or excluded object
        notes = []
        for i,row in enumerate(tbl_out):
            note = ''
            if name == 'NGC0628' or name=='NGC3351' or name=='NGC3627' or name=='NGC5068':
                note += row['match']
            if row['overluminous']:
                note += 'OL;'
            if row['SNRorPN']:
                note += 'PN;'
            notes.append(note)        
        tbl_out['notes'] = [x.strip(';') for x in notes]
      

        tbl_out.rename_columns(['name','RA','DEC','v_SIGMA'],['ID','R.A.','Dec.','sigmaV'])
        tbl_out = tbl_out[['Galaxy','ID','notes','R.A.','Dec.','mOIII','dmOIII','OIII/Ha','d(OIII/Ha)',
                           'Ha/NII','d(Ha/NII)','Ha/SII','d(Ha/SII)','sigmaV']]

        with open((filename / f'{name}_{typ}_candidates').with_suffix('.txt'),'w',newline='\n') as f:
            ascii.write(tbl_out,f,format='fixed_width_two_line',overwrite=True,delimiter_pad=' ',position_char='=')


        with open((filename / f'{name}_{typ}_candidates').with_suffix('.tex'),'w',newline='\n') as f:
            ascii.write(tbl_out,f,Writer=ascii.Latex,overwrite=True)

        
    logger.info(f'table saved to files (for {name})')


def write_LaTeX_old(table,galaxy,filename):
    '''write the table to a file

    this will create a `.tex` file and a machine readable file

    to read this file use
    f = basedir / 'data' / 'catalogues' / f'{galaxy.name}_{typ}_candidates.txt'
    ascii.read(f,format='fixed_width_two_line',position_char='=')
    '''

    threshold = '0.7"'

    table['SkyCoord'] = SkyCoord.from_pixel(table['x'],table['y'],galaxy.wcs)
    table['RA'],table['DEC'] = zip(*[x.split(' ') for x in table['SkyCoord'].to_string(style='hmsdms',precision=2)])

    if galaxy.name =='NGC0628':
        print('comparing to existing studies')
        from .load_references import NGC628
        cat = {'Kreckel PN':'K17','Herrmann PN':'H08','Kreckel SNR':'K17s'}
        ID, angle, Quantity  = match_coordinates_sky(NGC628['SkyCoord'],table['SkyCoord'])
        table['match'] = np.empty(len(table),dtype='U12')
        for i,a,row in zip(ID,angle,NGC628):
            if a.__lt__(Angle('0.8"')):
                table['match'][i] += cat[row['source']]


    if galaxy.name =='NGC3351':
        print('comparing to existing studies')
        from .load_references import pn_NGC3351_ciardullo
        ID, angle, Quantity  = match_coordinates_sky(pn_NGC3351_ciardullo['SkyCoord'],table['SkyCoord'])
        table['match'] = np.empty(len(table),dtype='U12')
        for i,a,row in zip(ID,angle,pn_NGC3351_ciardullo):
            if a.__lt__(Angle('0.8"')):
                table['match'][i] += 'C02'

    if galaxy.name =='NGC3627':
        print('comparing to existing studies')
        from .load_references import pn_NGC3627_ciardullo
        ID, angle, Quantity  = match_coordinates_sky(pn_NGC3627_ciardullo['SkyCoord'],table['SkyCoord'])
        table['match'] = np.empty(len(table),dtype='U12')
        for i,a,row in zip(ID,angle,pn_NGC3627_ciardullo):
            if a.__lt__(Angle('0.8"')):
                table['match'][i] += 'C02'

    if galaxy.name =='NGC5068':
        print('comparing to existing studies')
        from .load_references import pn_NGC5068_herrmann
        ID, angle, Quantity  = match_coordinates_sky(pn_NGC5068_herrmann['SkyCoord'],table['SkyCoord'])
        table['match'] = np.empty(len(table),dtype='U12')
        for i,a,row in zip(ID,angle,pn_NGC5068_herrmann):
            if a.__lt__(Angle('0.8"')):
                table['match'][i] += 'H08'

    for typ in ['PN','SNR']:

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
        tbl_out['Galaxy'] = galaxy.name
        tbl_out.sort('mOIII')

        tbl_out['OIII/Ha']   = np.empty(len(tbl_out),dtype='U8')
        tbl_out['Ha/NII']    = np.empty(len(tbl_out),dtype='U8') 
        tbl_out['Ha/SII']    = np.empty(len(tbl_out),dtype='U8')
        tbl_out['d(OIII/Ha)']= np.empty(len(tbl_out),dtype='U8')
        tbl_out['d(Ha/NII)'] = np.empty(len(tbl_out),dtype='U8')
        tbl_out['d(Ha/SII)'] = np.empty(len(tbl_out),dtype='U8')

        # add marker for existing study or excluded object
        names = np.arange(1,len(tbl_out)+1)
        notes = []
        for i,row in enumerate(tbl_out):
            notes = ''
            if galaxy.name == 'NGC0628' or galaxy.name=='NGC3351' or galaxy.name=='NGC3627' or galaxy.name=='NGC5068':
                name += row['match']
            if row['overluminous']:
                name += '+'
            if row['SNRorPN']:
                name += 'PN'

            names.append(name)        
        tbl_out['name'] = names
        tbl_out['notes'] = notes

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

            if not row['HA6562_detection'] and not row['SII_detection']:
                row['Ha/SII'] = '...'
                row['d(Ha/SII)'] = '...'
            elif not row['HA6562_detection']:
                row['Ha/SII'] += '<'
                row['Ha/SII'] += f"{row['HA6562'] / row['SII']:.2f}"
                row['d(Ha/SII)'] = f"{row['HA6562'] / row['SII'] * np.sqrt( (row['HA6562_err'] / row['HA6562'])**2 + (row['SII_err'] / row['SII'])**2):.2f}"
            elif not row['SII_detection']:
                row['Ha/SII'] += '>'
                row['Ha/SII'] += f"{row['HA6562'] / row['SII']:.2f}"
                row['d(Ha/SII)'] = f"{row['HA6562'] / row['SII'] * np.sqrt( (row['HA6562_err'] / row['HA6562'])**2 + (row['SII_err'] / row['SII'])**2):.2f}"
            else:
                row['Ha/SII'] += f"{row['HA6562'] / row['SII']:.2f}"
                row['d(Ha/SII)'] = f"{row['HA6562'] / row['SII'] * np.sqrt( (row['HA6562_err'] / row['HA6562'])**2 + (row['SII_err'] / row['SII'])**2):.2f}"


        tbl_out['mOIII'].info.format = '%.2f' 
        tbl_out['dmOIII'].info.format = '%.2f' 
        tbl_out['v_SIGMA'].info.format = '%.2f' 

        tbl_out.rename_columns(['name','RA','DEC','v_SIGMA'],['ID','R.A.','Dec.','sigmaV'])
        tbl_out = tbl_out[['Galaxy','ID','R.A.','Dec.','mOIII','dmOIII','OIII/Ha','d(OIII/Ha)',
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


        
    logger.info(f'table saved to files (for {galaxy.name})')


def read_catalogue(filename):

    catalogue = ascii.read(filename,format='fixed_width_two_line',delimiter_pad=' ',position_char='=')
    catalogue['SkyCoord'] = SkyCoord(catalogue['R.A.'],catalogue['Dec.'])

    return catalogue