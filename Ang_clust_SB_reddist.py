# ===========================================================================#
#============================================================================#
# XXXXXXXXXXXXXXXXXXX FISHER MATRIX CALCULATION XXXXXXXXXXXXXXXXXXXXXXXXXXXXX#
#
# ===========================================================================#
# ================== Author : Dimitrios D. Tanoglidis =======================#
# ============================ December 2018  ===============================#
#
#
# We calculate the Fisher matric for angular galaxy clustering in a single 
# redshift bin. In this code the redshift distribution of the sample is 
# given externally. We don't use any model to calculate it.
#
#
#
# In our forecast we leave as free parameters:
#                 - The cosmological parameters Omega_m and \sigma_8
#                 - The shift parameted Dz used to describe possible 
#					photo-z errors
#
# ==========================================================================#
# ==========================================================================#
# ==========================================================================#
#
import stuff
import numpy as np 
import scipy
from scipy.special import erf
from scipy import interpolate 
from scipy.interpolate import UnivariateSpline
import camb
from camb import model, initialpower


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

# Define a function that calculates angular power spectrum, C_l, in a bin i

def C_l(bias, n_z, Omega_m_var , sig_8_var):
    """
    Function that calculates the C_l in a bin
    -----------------
    Inputs:
    bias : bias - constant or function
    n_z : redshift distribution at a redshift bin
    Omega_m_var: Omega matter - can change
    sig_8_var : Sigma_8 parameter - can change
    --------------
    Returns:
    ls and C_l in a bin 
    """
    # Constants
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
    #====================================================================================
    #====================================================================================
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

# ========================================================================================
# ========================================================================================
# Now define functions to calculate the derivatives of the Angular Power Spectrum
# w/r to the cosmological parameters - matter density and sigma_8
# ========================================================================================
# ========================================================================================
#
#
# Derivative with respect to matter density - Omega m
def matter_der_C_l(bias, n_z, Omega_m , sig_8):
    """
    Function that calculates the derivative of C_l with respect to matter density in a bin
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
    alpha_m = Omega_m/10.0
    
    C_mat_1 = C_l(bias, n_z, Omega_m+alpha_m , sig_8)[1]
    C_mat_2 = C_l(bias, n_z, Omega_m-alpha_m , sig_8)[1]
    
    mat_der = (C_mat_1 - C_mat_2)/(2.0*alpha)
    return mat_der
    
    
# =======================================================================================
# Derivative with respect to sigma_8 

def sigma_der_C_l(bias, n_z, Omega_m , sig_8):
    """
    Function that calculates the derivative of C_l with respect to sigma_8 in a bin
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
    
    alpha_s = sig_8/10.0
    
    C_sig_1 = C_l(bias, n_z, Omega_m, sig_8+alpha_s)[1]
    C_sig_2 = C_l(bias, n_z, Omega_m , sig_8-alpha_s)[1]
    
    sig_der = (C_sig_1 - C_sig_2)/(2.0*alpha_s)
    return 

# ========================================================================================
# ========================================================================================
# Derivative with respect to the shift parameter

def shift_der_C_l(bias, n_z, dn_dDz, Omega_m, sig_8):
	"""
	Function that calculates the derivative of C_l with respect to
	the redshift shift, Dz
	-------------------------------------------------------
	bias : bias - constant or function
    n_z : redshift distribution at a redshift bin
    dn_dDz :
    Omega_m_var: Omega matter - can change
    sig_8_var : Sigma_8 parameter - can change
    --------------
    Returns:
    derivative w/r to the shift parameter
	"""
	# Constants
	h = 0.682
    c = 2.99792e+5
    # =======================================================================
    # Selecting cosmology
    
    cosmo = cosmoselector(Omega_m, sig_8)
    
    #========================================================================
    #========================================================================
    #Redshift range for calculations and integration
    
    nz = 1000 #number of steps to use for the radial/redshift integration
    kmax=10.0  #kmax to use

    zarray = np.linspace(0,4.0,nz)
    dzarray = (zarray[2:]-zarray[:-2])/2.0
    zarray = zarray[1:-1]

    #=========================================================================
    #Calculate Hubble parameter and comoving distance
    
    Hubble = cosmo.H_z
    
    # Get comoving distance - in Mpc
    chis = cosmo.chi
    
    #========================================================================
    #========================================================================
    # Get the prefactor of the integral 

    pref = ((bias/chis)**2.0)*(2.0*n_z*dn_dDz*Hubble)

    #===================================================================================
    #Do integral over z
    
    ls_lin = np.linspace(1.0, np.log10(2000.0), 55, dtype = np.float64)
    ls = 10.0**ls_lin
    
    der_C = np.zeros(ls.shape)
    w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
    for i, l in enumerate(ls):
        k=(l+0.5)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=kmax]=0
        der_C[i] = np.dot(dzarray, w*cosmo.PK.P(zarray, k, grid=False)*pref)
    
    #===================================================================================
    # Retrurn the array of C_ell
    
    return  der_C

# ===========================================================================================
# ===========================================================================================

# Function that finds the breaking ell - if there is such a breaking ell -
# Also returns if the function starts from positive or negative values

def breaking_ell(ells, search_array):
    """
    Function that finds where the array called "search_array" changes sign
    returns this "breaking ell", as well as the behavior - if it starts from positive or negative values
    ------------------------------------------------
    Inputs:
    ells : the array of ell_lin - I'm trying to find the breaking as one of its elements
    search_array : the array whose behavior I'm trying to define - where it changes sign etc
    -----------------------------------------------
    Returns:
    ell_break : the breaking ell - the ell where the array/function changes sign
    0 or 1 : shows if the search array starts from negative or positive values
    """
    # Initialize - the breaking ell is the last value of the array
    ell_break = ells[-1]
    
    #Size of ells
    n_size = len(ells)
    #===========================================================
    #===========================================================
    if (search_array[0] <= 0.0):
        alpha = 0
    else:
        alpha = 1
        
    if (alpha == 0):
        for s in range(n_size):
            if (np.sign(search_array[s])>=0.0):
                ell_break = 0.5*(ells[s]+ells[s-1])
                break
    else:
        for s in range(n_size):
            if (np.sign(search_array[s])<=0.0):
                ell_break = 0.5*(ells[s]+ells[s-1])
                break       
    #========================================================         
    
    return alpha, ell_break 

# =========================================================================================
# =========================================================================================
# =========================================================================================
# Calculation of the Fisher Matrix in a single bin

def Fish_single_bin(z_mean, bias, n_z, dn_dDz, f_sky, N_gal):
	"""
	Calculates and returns the Fisher matrix for a single bin
	------------------------------
	Inputs:
	z_mean : mean redshift of the bin
    bias : bias - function or constant
    n_z : redshift distribution of the bin
    dn_dDz : derivative of the redshift distribution, with respect to the shift parameter Dz
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
    kmax = 10.0  #kmax to use

    zarray = np.linspace(0,4.0,nz)
    dzarray = (zarray[2:]-zarray[:-2])/2.0
    zarray = zarray[1:-1]
    
    #=========================================================================
    # calculation of l_max
    chi_mean = backres.comoving_radial_distance(z_mean)   # comoving distance corresponding to the mean redshift of the bin
    k_cutoff= 0.6*h #Cutoff scale in  Mpc^{-1}
    l_max = int(round(chi_mean*k_cutoff))
    
    #=========================================================================
    #Calculation of the angular number density galaxies / steradian
    
    ster = f_sky*(4.0*np.pi)
    n_bin = N_gal/ster

    # =========================================================================

    # bias - if you want comment out it
    # bias = 1.0 + z_mean

    #=========================================================================
    #========================================================================
    # Now take the ls, C_ls and the derivatives of the C_ls 
    ell_lin = np.linspace(1.0, np.log10(2000.0), 55, dtype = np.float64)

    C_ell_1 = C_l(bias, n_z, Omega_m , sigma_8)[1]
    dC_ldOm_1 = matter_der_C_l( bias, n_z, Omega_m , sigma_8)
    dC_ldsig8_1 = sigma_der_C_l( bias, n_z, Omega_m , sigma_8)
    dC_ldShift_1 = shift_der_C_l(bias, n_z, dn_dDz, Omega_m, sig_8)


   # ======================================================================
   # Find the breaking ells now

   al_1, l_break_1 = breaking_ell(ell_lin, dC_ldOm_1)
   al_2, l_break_2 = breaking_ell(ell_lin, dC_ldShift_1)


   # ======================================================================
   ls = np.arange(10, 2000, dtype=np.float64)
    
    #Initialize
    
    C_ell = np.zeros(np.size(ls)) #C_ell's
    dC_ldOm = np.zeros(np.size(ls)) # matter derivative
    dC_ldsig8 = np.zeros(np.size(ls)) #sigma_8 derivative
    dC_ldShift = np.zeros(np.size(ls)) # Derivative with respect to the shift parameter

    #=================================================================================
    # Interpolate 

    C_l_matr_interp = UnivariateSpline(ell_lin, np.log10(C_ell_1+ 1.0e-20), s=0.0)
    C_om_mat_interp = UnivariateSpline(ell_lin, np.log10(abs(dC_ldOm_1+ 1.0e-20)), s=0.0)
    C_sig_interp = UnivariateSpline(ell_lin, np.log10(dC_ldsig8_1+1.0e-20), s=0.0)
    C_shift_interp = UnivariateSpline(ell_lin, np.log10(abs(dC_ldShift_1+1.0e-20)), s=0.0)

    # ===============================================================================
    # populate 

    for k, l in enumerate(ls):
    	ell = np.log10(float(l))
        C_ell[k]  = 10.0**(C_l_matr_interp(ell))
        dC_ldsig8[k] = 10.0**(C_sig_interp(ell))

        if (al_1 == 0):
            if (ell < l_break_1):
                dC_ldOm[k] = -(10.0**C_om_mat_interp(ell))
            else: 
                dC_ldOm[k] = (10.0**C_om_mat_interp(ell))
        else:
            if (ell < l_break_1):
                dC_ldOm[k] = (10.0**C_om_mat_interp(ell))
            else: 
                dC_ldOm[k] = -(10.0**C_om_mat_interp(ell))
            
         
        if (al_2 == 0):
            if (ell < l_break_2):
                dC_ldShift[k] = -(10.0**C_shift_interp(ell))
            else:
                dC_ldShift[k] = (10.0**C_shift_interp(ell))
        else:
            if (ell < l_break_2):
                dC_ldShift[k] = (10.0**C_shift_interp(ell))
            else:
                dC_ldShift[k] = -(10.0**C_shift_interp(ell))
                
       # =================================================================================
   	# ====================================================================================
   	ls = ls[:l_max-9]
    C_ell = C_ell[:l_max-9]
    dC_ldOm = dC_ldOm[:l_max-9]
    dC_ldsig8 = dC_ldsig8[:l_max-9]
    dC_ldShift = dC_ldShift[:l_max-9]
    
    #Create arrays with sigma^2
    sigma_sq = (2.0/(f_sky*(2.0*ls + 1.0)))*((C_ell + 1.0/n_bin)**2.0)
    #and its inverse - it will be useful 
    inv_sigma = 1.0/sigma_sq # inverse of sigma square - that's what we want
    
    #===============================================================================
    #===============================================================================
    # Calculation of the elements of the Fisher matrix
    
    Fish = np.zeros([3,3])
    # 0 = matter, 1 = sigma_8, 2 = bias, 3 = z_bias, 4 = sigma_z
    
    # Diagonal terms first
    Fish[0,0] = sum(inv_sigma*(dC_ldOm**2.0))
    Fish[1,1] = sum(inv_sigma*(dC_ldsig8**2.0))
    Fish[2,2] = sum(inv_sigma*(dC_ldShift**2.0))
    
    
    # Non-diagonal terms
    Fish[0,1] = Fish[1,0] = sum(inv_sigma*dC_ldOm*dC_ldsig8)
    Fish[0,2] = Fish[2,0] = sum(inv_sigma*dC_ldOm*dC_ldShift)
    
    Fish[1,2] = Fish[2,1] = sum(inv_sigma*dC_ldsig8*dC_ldShift)
    
    return Fish   









