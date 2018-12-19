#import stuff
import numpy as np 
import scipy
from scipy.special import erf
from colossus.cosmology import cosmology
from scipy import interpolate 
from scipy.interpolate import UnivariateSpline
import camb
from camb import model, initialpower
from scipy.special import gamma

#==============================================================================================
#==============================================================================================
# Now create a class that can create CAMB cosmologies for different matter densities and sigma_8

class Cosmology:
    
    def __init__(self,omega_m,sigma_8,h,z):
        self.omega_m = omega_m
        self.sigma_8 = sigma_8
        self.h = h
        self.z = z
        self.k_max = 10.0
        self.c = 2.99792e+5
        #=========================
        
        cosmo = camb.CAMBparams()
        cosmo.set_cosmology(H0=100.0*self.h, ombh2=0.048*(self.h**2.0), omch2=(self.omega_m - 0.048)*(self.h**2.0), mnu=0.06, omk=0, tau=0.06)
        cosmo.InitPower.set_params(As=2.0e-9, ns=0.973)
        results = camb.get_results(cosmo)
        cosmo.set_matter_power(redshifts=[0.0], kmax=10.0)
        cambres= camb.get_transfer_functions(cosmo)
        cosmo.NonLinear = model.NonLinear_both
        kh, z, pk = cambres.get_matter_power_spectrum(minkh=1e-3, maxkh=1.0, npoints = 10)
        sigma_8_temp = cambres.get_sigma8()
        As_new  = ((self.sigma_8/sigma_8_temp)**2.0)*(2.0e-9)
        cosmo.InitPower.set_params(As=As_new, ns=0.973)
        cambres = camb.get_results(cosmo)
        backres = camb.get_background(cosmo)

        self.chi = backres.comoving_radial_distance(self.z)
           
        self.PK = camb.get_matter_power_interpolator(cosmo, nonlinear=True, 
                hubble_units=False, k_hunit=False, kmax=self.k_max, zmin = 0.0, zmax=self.z[-1]) 
        
        self.H_z = (backres.hubble_parameter(self.z))/self.c #Hubble parameter in 1/Mpc 


#===================================================================================================================
#===================================================================================================================

# Selecting cosmologies

# Instantize cosmologies 

omega_m = 0.301
sigma_8 = 0.798
h = 0.682
alpha_om  = omega_m/10.0
alpha_sig = sigma_8/10.0

#==========================
nz = 1000 #number of steps to use for the radial/redshift integration

zarray = np.linspace(0,4.0,nz)
z = zarray[1:-1]

cosmo_fid = Cosmology(omega_m, sigma_8, h, z)
cosmo_1 = Cosmology(omega_m + alpha_om, sigma_8, h, z)
cosmo_2 = Cosmology(omega_m - alpha_om, sigma_8, h, z)
cosmo_3 = Cosmology(omega_m, sigma_8 + alpha_sig, h, z)
cosmo_4 = Cosmology(omega_m, sigma_8 - alpha_sig, h, z)

#=====================================================================================================
#=====================================================================================================


def cosmoselector(omega, sigma):
    #function that selects cosmology
    
    omfid = 0.301
    sigfid = 0.798
    
    cosmo_dict = {'cosmo_fid': cosmo_fid,
                  'cosmo_1' : cosmo_1,
                  'cosmo_2' : cosmo_2,
                  'cosmo_3' : cosmo_3,
                  'cosmo_4' : cosmo_4}
    
    
    if (omega==omfid):
        if (sigma == sigfid):
            cosm_sel = cosmo_dict['cosmo_fid']
        elif (sigma > sigfid):
            cosm_sel = cosmo_dict['cosmo_3']
        else:
            cosm_sel = cosmo_dict['cosmo_4']
    elif (omega > omfid): 
        cosm_sel = cosmo_dict['cosmo_1']
    else:
        cosm_sel = cosmo_dict['cosmo_2']
        
    
    return cosm_sel

#=================================================================================================
#=================================================================================================

# Function that calculates C_l,i

def C_l_i(bias, n_z, Omega_m_var , sig_8_var):
    """
    Function that calculates the C_l between two bins 
    -----------------
    Inputs:
    bias : bias - constant or function
    n_z : redshift distribution at a redshift bin
    Omega_m_var: Omega matter - can change
    sig_8_var : Sigma_8 parameter - can change
    --------------
    Returns:
    ls and C_l betwenn two bins, i and j. It is the auto spectrum if i=j
    """
    # Constant
    h = 0.682
    c = 2.99792e+5
    
    #======================================
    #====================================================================================
    #====================================================================================
    # Selecting cosmology
    
    cosmo = cosmoselector(Omega_m_var, sig_8_var)
    
    #====================================================================================
    #====================================================================================
    #Redshift range for calculations and integration
    
    nz = 1000 #number of steps to use for the radial/redshift integration
    kmax=10.0  #kmax to use

    zarray = np.linspace(0,4.0,nz)
    dzarray = (zarray[2:]-zarray[:-2])/2.0
    zarray = zarray[1:-1]
    
    
    #Calculate square of the window function

    W_sq = (bias*n_z)**2.0
    
    
    #====================================================================================
    #====================================================================================
    #Calculate Hubble parameter and comoving distance
    
    Hubble = cosmo.H_z
    
    # Get comoving distance - in Mpc
    chis = cosmo.chi
    
    #========================================================
    # Get the full prefactor of the integral
    prefact = W_sq*Hubble/(chis**2.0)
    #====================================================================================
    
    #===================================================================================
    #===================================================================================
    #Do integral over z
    
    ls_lin = np.linspace(1.0, np.log10(2000.0), 55, dtype = np.float64)
    ls = 10.0**ls_lin
    
    c_ell=np.zeros(ls.shape)
    w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
    for i, l in enumerate(ls):
        k=(l+0.5)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=kmax]=0
        c_ell[i] = np.dot(dzarray, w*cosmo.PK.P(zarray, k, grid=False)*prefact)
    
    #===================================================================================
    # Retrurn the array of C_ell
    
    return ls, c_ell



#===================================================================================
# Here are the derivatives with respect to matter density and sigma_8
#===================================================================================

def matter_der_C_l_i(bias, n_z, Omega_m , sig_8):
    """
    Function that calculates the derivative of C_l with respect to matter between two bins 
    -----------------
    Inputs:
    z_i : Lower limit of the redshift bin
    z_f : Upper limit of the redshift bin
    bias : the linear galaxy bias
    n_z : the normalized redshift distribution
    Omega_m: Omega matter
    sig_8: Sigma_8 parameter
    
    ---------------
    Returns:
    derivative w/r to matter of C_l betwenn two bins, i and j
    """
    alpha = Omega_m/10.0
    
    C_mat_1 = C_l_i(bias, n_z, Omega_m+alpha , sig_8)[1]
    C_mat_2 = C_l_i(bias, n_z, Omega_m-alpha , sig_8)[1]
    
    mat_der = (C_mat_1 - C_mat_2)/(2.0*alpha)
    return mat_der
    
    
    #===================================================================================
    
def sigma_der_C_l_i(bias, n_z, Omega_m , sig_8):
    """
    Function that calculates the derivative of C_l with respect to sigma_8 between two bins 
    -----------------
    Inputs:
    bias : the linear galaxy bias
    n_z : the normalized redshift distribution
    Omega_m: Omega matter
    sig_8: Sigma_8 parameter
    ---------------
    Returns:
    derivative w/r to matter of C_l betwenn two bins, i and j
    """
    
    alpha = sig_8/10.0
    
    C_sig_1 = C_l_i(bias, n_z, Omega_m, sig_8+alpha)[1]
    C_sig_2 = C_l_i(bias, n_z, Omega_m , sig_8-alpha)[1]
    
    sig_der = (C_sig_1 - C_sig_2)/(2.0*alpha)
    return sig_der

#=================================================================================================
#=================================================================================================
#=================================================================================================

from scipy.interpolate import UnivariateSpline

def Fish_single_bin(z_mean, bias, n_z, f_sky, N_gal):
    """
    Calculates and returns the Fisher matrix for a single bin
    ----------------------------------------
    Inputs:
    z_mean : mean redshift of the bin
    bias : bias - function or constant
    n_z : redshift distribution of the bin
    f_sky : fraction of the sky the survey covers 
    N_gal : number of galaxies in the bin
    
    ---------------------------------------
    Outputs:
    Fisher matrix for a single bin
    """
    
    
    Omega_m = 0.301
    sigma_8 = 0.798
    h = 0.682
    
    #Setting up cosmology - need to calculate chis
    
    # Setting up cosmology
    
    cosmo = camb.CAMBparams()
    cosmo.set_cosmology(H0=68.2, ombh2=0.048*(h**2.0), omch2=(Omega_m - 0.048)*(h**2.0), mnu=0.06, omk=0, tau=0.06)
    backres = camb.get_background(cosmo)
    #=============================================================
    
    #=============================================================
    
    #Redshift range for calculations and integration
    
    nz = 1000 #number of steps to use for the radial/redshift integration
    kmax=10.0  #kmax to use

    zarray = np.linspace(0,4.0,nz)
    dzarray = (zarray[2:]-zarray[:-2])/2.0
    zarray = zarray[1:-1]
    
    #==============================================================================
    # calculation of l_max
    chi_mean = backres.comoving_radial_distance(z_mean)   # comoving distance corresponding to the mean redshift of the bin
    k_cutoff= 0.6*h #Cutoff scale in  Mpc^{-1}
    l_max = int(round(chi_mean*k_cutoff))
    
    #==============================================================================
    #==============================================================================
    #Calculation of the angular number density galaxies / steradian
    
    ster = f_sky*(4.0*np.pi)
    n_bin = N_gal/ster
    
    #===============================================================================
    # Now take the ls, C_ls and the derivatives of the C_ls - then keep only up to lmax
    ell_lin = np.linspace(1.0, np.log10(2000.0), 55, dtype = np.float64)
    
    C_ell_1 = C_l_i(bias, n_z, Omega_m , sigma_8)[1]
    dC_ldOm_1 = matter_der_C_l_i( bias, n_z, Omega_m , sigma_8)
    dC_ldsig8_1 = sigma_der_C_l_i( bias, n_z, Omega_m , sigma_8)
    
    for s in range(0,np.size(dC_ldOm_1)):
        if (np.sign(dC_ldOm_1[s])>=0.0):
            l_break = 0.5*(ell_lin[s]+ell_lin[s-1])
            break
            
    ls = np.arange(10,2000, dtype=np.float64) 
    
    C_ell = np.zeros(np.size(ls))
    dC_ldOm = np.zeros(np.size(ls))
    dC_ldsig8 = np.zeros(np.size(ls))
    #====================================================================
    
    C_l_matr_interp  = UnivariateSpline(ell_lin, np.log10(C_ell_1+ 1.0e-20))
    C_omeg_interp = UnivariateSpline(ell_lin, np.log10(abs(dC_ldOm_1+1.0e-20)))
    C_sig_interp = UnivariateSpline(ell_lin, np.log10(dC_ldsig8_1+1.0e-20))
    for k, l in enumerate(ls):
        ell = np.log10(float(l))
        C_ell[k]  = 10.0**(C_l_matr_interp(ell))
        dC_ldsig8[k] = 10.0**(C_sig_interp(ell))
        if (ell < l_break):
            dC_ldOm[k] = -(10.0**C_omeg_interp(ell))
        else: 
            dC_ldOm[k] = (10.0**C_omeg_interp(ell))
            
    
    ls = ls[:l_max-9]
    C_ell = C_ell[:l_max-9]
    dC_ldOm = dC_ldOm[:l_max-9]
    dC_ldsig8 = dC_ldsig8[:l_max-9]
    
    #Create arrays with sigma^2
    sigma_sq = (2.0/(f_sky*(2.0*ls + 1.0)))*((C_ell + 1.0/n_bin )**2.0)
    
    #===============================================================================
    #===============================================================================
    # Calculation of the elements of the Fisher matrix
    Fish = np.zeros([2,2])
    
    # 0 = matter, 1 = sigma_8
    
    Fish[0,0] = sum((1.0/sigma_sq)*(dC_ldOm**2.0)) 
    Fish[1,1] = sum((1.0/sigma_sq)*(dC_ldsig8**2.0)) 
    Fish[0,1]=Fish[1,0]= sum((1.0/sigma_sq)*(dC_ldOm*dC_ldsig8))
    
    
    return Fish
