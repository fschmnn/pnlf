import numpy as np
from scipy.special import hyp2f1
import astropy
import astropy.units as u        
from astropy.coordinates import Distance, SkyCoord
from .constants import arcsec_to_pixel

class Distance_old:
    def __init__(self,value,unit):
        '''save distance in 

        '''

        if unit == 'cm':
            self.value = value
        elif unit == 'm':
            self.value = value * 100
        elif unit in {'pc','parsec'}:
            self.value = value * 3.085678e18
        elif unit in {'ly','lightyear'}:
            self.value = value * 9.460730472e17
        elif unit in {'mag','distance_modulus','mu'}:
            self.value = 10**(1+value/5) *  3.085678e18 
        else:
            raise ValueError(f'unkown unit: {unit}')

    def to_cm(self):
        return self.value

    def to_parsec(self):
        return self.value / 3.085678e18 

    def to_lightyear(self):
        return self.value / 9.460730472e17

    def to_distance_modulus(self):
        return 5*np.log10(self.value/3.085678e18 ) - 5
        

def search_table(table,string):
    
    mask = np.zeros(len(table),dtype=bool)
    
    for i, row in enumerate(table):
        if string in ','.join(map(str,row)):
            mask[i]= True
    return  table[mask]

correct_PSF = lambda lam: arcsec_to_pixel*3e-5*(lam-6483.58)
#correct_PSF = lambda lam: 1- 4.7e-5*(lam-6450)

def fwhm_moffat(alpha,gamma):
    '''calculate the FWHM of a Moffat'''

    return 2*gamma * np.sqrt(2**(1/alpha)-1) 

def light_in_moffat(x,alpha,gamma):
    '''theoretical growth curve for a moffat PSF

    f(x;alpha,gamma) ~ [1+r^2/gamma^2]^-alpha

    Parameters
    ----------
    x : float
        Radius of the aperture in units of pixel
    '''

    return 1-(1+x**2/gamma**2)**(1-alpha)

def light_in_gaussian(x,fwhm):
    '''theoretical growth curve for a gaussian PSF

    Parameters
    ----------
    x : float
        Radius of the aperture in units of pixel

    fwhm : float
        FWHM of the Gaussian in units of pixel
    '''

    return 1-np.exp(-4*np.log(2)*x**2 / fwhm**2)



def light_in_moffat_old(x,alpha,gamma):
    '''
    without r from rdr one gets an hpyerfunction ...
    '''
    r_inf = 105
    f_inf = r_inf*hyp2f1(1/2,alpha,3/2,-r_inf**2/gamma**2)
    return x*hyp2f1(1/2,alpha,3/2,-x**2/gamma**2) / f_inf


def test_convergence(array,length=4,threshold=0.05):
    '''test if the given array approches a constant limit
    
    Parameters
    ----------
    array : list
    
    length : int
        number of elements that must approach the limit
        
    threshold : float
        maximum deviation from the limit
    '''
    
    if len(array) < length:
        return False

    array = np.atleast_1d(array)
    mean  = np.mean(array[-length:])
    return np.all((array[-length:]-mean)/mean < threshold)


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

def create_sources_mask(shape,positions,radius):
    '''mask out all 
    
    source_mask = create_sources_mask(galaxy.shape,sources[['x','y']].as_array(),radius=5)

    '''

    mask = np.zeros(shape,dtype=bool)

    for position in positions:
        mask |= circular_mask(*shape,position,radius=radius)
    
    return mask


class Table(astropy.table.Table):
    '''A class to represent tables of heterogeneous data.

    This is a subclass of the astropy.Table class with an additional method
    to filter the table. Below the original description:
    
    Init signature: Table(data=None, masked=None, names=None, dtype=None, 
                meta=None, copy=True, rows=None, copy_indices=True, **kwargs)

    `Table` provides a class for heterogeneous tabular data, making use of a
    `numpy` structured array internally to store the data values.  A key
    enhancement provided by the `Table` class is the ability to easily modify
    the structure of the table by adding or removing columns, or adding new
    rows of data.  In addition table and column metadata are fully supported.

    `Table` differs from `~astropy.nddata.NDData` by the assumption that the
    input data consists of columns of homogeneous data, where each column
    has a unique identifier and may contain additional metadata such as the
    data unit, format, and description.

    Parameters
    ----------
    data : numpy ndarray, dict, list, Table, or table-like object, optional
        Data to initialize table.
    masked : bool, optional
        Specify whether the table is masked.
    names : list, optional
        Specify column names.
    dtype : list, optional
        Specify column data types.
    meta : dict, optional
        Metadata associated with the table.
    copy : bool, optional
        Copy the input data. If the input is a Table the ``meta`` is always
        copied regardless of the ``copy`` parameter.
        Default is True.
    rows : numpy ndarray, list of lists, optional
        Row-oriented data for table instead of ``data`` argument.
    copy_indices : bool, optional
        Copy any indices in the input data. Default is True.
    **kwargs : dict, optional
        Additional keyword args when converting table-like object.
    '''
    
    def filter(self, **kwargs: 'column name and value'):
        '''Filter the table with the given keywords

        This function takes an arbitrary astropy table and applys the given 
        keyword pairs (name,value) as filters like  data[data[name]==value]. 
        If value is a list, it filters such that each row contains at least
        one value from the list. If a value does not exist in any row, it 
        is ignored. The function returns a new table and leaves the old 
        table unmodified.

        Equality is the only possible relational operator.

        This function uses np.isin which is equivalent to 
        [item in v for item in table[k]] 
        However this syntax takes only lists while np.isn can also handle 
        single numbers.
        '''
        
        # we return a copy of the table
        table = self
        
        for k,v in kwargs.items():
            # only apply filter if the column exists
            if k in table.colnames:
                table = table[np.isin(table[k],v)]
            else:
                raise ValueError(f'WARNING: invalid column name {k}')

        return table


def filter_table(table,**kwargs):
    '''filter a table with the given keyword arguments'''
    
    for k,v in kwargs.items():
        # only apply filter if the column exists
        if k in table.colnames:
            table = table[np.isin(table[k],v)]
        else:
            raise ValueError(f'WARNING: invalid column name {k}')

    return table
    

def mu_to_parsec(mu,error=None):

    d = Distance(distmod=mu)
    error = np.atleast_1d(error)

    if len(error)>=1:
        return d, 0.2*np.log(10)*d*error
    else:
        return d

def parsec_to_mu(d,error=None):

    error = np.atleast_1d(error)
    mu = Distance(d).distmod
    if len(error)>=1:
        return  mu, 5/(np.log(10)) * error/d
    else:
        return mu


def diameter(D,delta):
    '''Calculate physical diameter from angular diameter and distance
    
    Parameters
    ----------
    
    D :
        Distance
        
    delta :
        Angular Diameter
    
    '''
    
    if not D.unit.is_equivalent(u.cm):
        raise u.UnitsError('invalid unit for distance: '+D.unit)
        
    if not delta.unit.is_equivalent(u.degree):
        raise u.UnitsError('invalid unit for distance: '+delta.unit)
        
    return (2 * np.tan(delta/2) * D).to(u.parsec)
    
    
def angular_diameter(D,d):
    '''Calculate angular diameter from real diameter and distance
    
    Parameters
    ----------
    
    D :
        Distance
        
    d :
        physical diameter
    
    '''
    
    if not D.unit.is_equivalent(u.cm):
        raise u.UnitsError(f'invalid unit for distance: {D.unit}')
        
    if not d.unit.is_equivalent(u.cm):
        raise u.UnitsError(f'invalid unit for distance: {d.unit}')
        
    return 2 * np.arctan(d/(2*D)).to(u.arcsec)


def nanunique(arr):
    '''Find the unique elements of an array (excluding NaNs).''' 


    return np.unique(arr[~np.isnan(arr)])


def resolution_from_wcs(wcs):
    '''calculate the resolution in arcsecond for any given wcs
    
    Parameters
    ----------
    wcs
    
    Returns
    -------
    tuple : the extend per pixel in arcseconds
    '''
    
    shape = wcs._naxis
    
    x = [0,0,shape[0]]
    y = [0,shape[1],0]

    coords = SkyCoord.from_pixel(x,y,wcs)
    
    dx = coords[0].separation(coords[1]) / shape[0]
    dy = coords[0].separation(coords[1]) / shape[1]
    
    return (round(dx.to(u.arcsecond).value,3),round(dy.to(u.arcsecond).value,3))


def sample_numerical(x,y,N=100,plot=False):
    '''sample from a PDF given by two arrays

    '''

    xmin,xmax = np.min(x),np.max(x)
    ymin,ymax = np.min(y),np.max(y)

    sample = []
    
    while len(sample)<N:
        
        x0 = np.random.uniform(xmin,xmax)
        y0 = np.interp(x0,x,y)
        yref = np.random.uniform(ymin,ymax)
        
        if y0>yref:
            sample.append(x0)
    sample = np.array(sample)
    
    if plot:
        fig,ax=plt.subplots()
        ax.hist(sample,bins=np.arange(xmin,xmax,0.01),density=True)
        ax.plot(x,y,color='black')
        plt.show()
    
    return sample



def project(x, y, pa, inc):
    """General rotation/projection routine.

    Given coordinates (x, y), will rotate and project given position angle (counter-clockwise from N), and
    inclination. Assumes centre is at (0, 0).
    Args:
        x (float or numpy.ndarray): x-coordinate(s)
        y (float or numpy.ndarray): y-coordinates(s)
        pa (float): Position angle (degrees)
        inc (float): Inclination (degrees)
    Returns:
        x_proj, y_proj: The rotated, projected (x, y) coordinates.
    """

    angle = np.radians(pa)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_proj = x * cos_a + y * sin_a
    y_proj = - x * sin_a + y * cos_a
    # Account for inclination
    x_proj /= np.cos(np.radians(inc))
    return x_proj, y_proj

def r25(x,y,centre,pa,inc,r25):
    '''calculate radius in terms of r25'''

    pass
    

from astropy.coordinates import match_coordinates_sky

def merge_catalogues(catalogues: list,threshold=1*u.arcsec):
    '''merge multiple catalogues based on their position in the sky
    
    does not work yet
    '''

    for catalogue in catalogues:
        if 'SkyCoord' not in catalogue.columns:
            raise ValueError('catalogue must have column `SkyCoord')
    
    # we start with the first catalogue for our master catalogue
    merged = catalogues[0]
    merged['id_1'] = np.arange(len(merged),dtype=float)
    
    for i,catalogue in enumerate(catalogues[1:],start=2):
        # create a unique ID for the catalogue
        catalogue[f'id_{i}'] = np.arange(len(catalogue),dtype=float)

        # match the two catalogues (two ojbects that are closer than threshold)
        idx,sep,_ = match_coordinates_sky(merged['SkyCoord'],catalogue['SkyCoord'])
        merged[f'id_{i}'] = catalogue[idx][f'id_{i}']
        merged[f'id_{i}'][sep>threshold] = np.nan
        catalogue.rename_column('SkyCoord',f'SkyCoord_{i}')
        
        merged = join(merged,catalogue,keys=f'id_{i}',join_type='outer')
    
    # remove the id columns
    for i in range(len(catalogues)):
        del merged[f'id_{i}']
                   
    return merged

