import astropy.constants as c
import astropy.units as u
import numpy as np 

def V_band_to_bolometric(flux,mu):
    '''Convert V-band flux to Lbol
    
    Parameters
    ----------
    flux : flux in erg/s/cm2/AA
    '''
    
    # we can (roughly) convert this to erg / s / cm^2 / Hz via
    F_nu = ((550*u.nm)**2/c.c * flux).to(u.erg / (u.s*u.cm**2*u.Hz))

    # define zero point in different units
    V0_lam = 363.1e-11 * u.erg / (u.s*u.cm**2*u.AA)
    V0_nu  = 3.64e-20 * u.erg / (u.s*u.cm**2*u.Hz)
    B0_lam = 632e-11 * u.erg / (u.s*u.cm**2*u.AA)
    B0_nu  = 4.26e-20 * u.erg / (u.s*u.cm**2*u.Hz)

    AV = 0

    # bolometric correction
    BC_V = -0.85
    BC_B = -1.5

    mV = -2.5*np.log10(flux/V0_lam)
    mV_nu  = -2.5*np.log10(F_nu/V0_nu)

    MV = mV - mu - AV

    Lbol = 10**(-0.4*(MV-4.79)) * 10**(-0.4*(BC_V+0.07)) * u.Lsun
    
    return Lbol
    
def get_bolometric_luminosity(V_band,mu,mask=None):

    if not isinstance(mask,np.ndarray):
        mask = np.zeros_like(V_band)

    # from the DAP we have the spectral flux density in erg / s / cm^2 / AA
    F_lam = np.nansum(V_band[~mask])*1e-20 * u.erg / (u.s*u.cm**2*u.AA)
    
    Lbol = V_band_to_bolometric(F_lam,mu)

    return Lbol