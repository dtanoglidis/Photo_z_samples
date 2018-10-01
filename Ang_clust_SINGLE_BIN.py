# =============================================================================#
# =============================================================================#
# xxxxxxxxxxxxxxxxxxxxxx FISHER MATRIX CALCULATION xxxxxxxxxxxxxxxxxxxxxxxxxxxx#
#
# =============================================================================#
# =================== Author: Dimitrios D. Tanoglidis =========================#
# ============================ September 2018 =================================#
#
#
# In this notebook we calculate the Fisher Matrix for angular galaxy clustering in a single redshift bin.
# 
# In our forecast we leave as free parameters:
#	- The cosmological parameters Omega_m and \sigma_8 
#	- One photo-z scatter parameter #\sigma_{z,0}
#	- One photo-z bias parameter, z_b
#	- One galaxy bias parameter, b_g
#
#
# ===============================================================================#
# ===============================================================================#
# ===============================================================================#
#
#import stuff
import numpy as np 
import scipy
from scipy.special import erf
from scipy import interpolate 
from scipy.interpolate import UnivariateSpline
import camb
from camb import model, initialpower
from scipy.special import gamma
# ==============================================================================#
# Now create a class that can create CAMB cosmologies for different matter densities and sigma_8
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
        
#============================================================================================================
#============================================================================================================

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
# =====================================================================================================
# =====================================================================================================
# Calculation of Window fuction and C_l in a bin

#Function that calculates and returns window function W(z) for clustering in a bin i

def W_z_clust(z, dz, z_i, z_f, sig_z, z_bias, bias):
    """
    Function that calculates the window function for 2D galaxy clustering
    -----------------
    Inputs:
    z: array of redshifts where we are going to calculate the window function
    dz: array of dz's - useful for the integral
    z_i : lower redshift limit of the bin
    z_f : upper redshift limit of the bin
    sig_z : photometric error spread
    z_bias : redshift bias
    bias : galaxy bias
    ---------------
    Returns:
    The window function and its integral over all redshifts for a given bin with given limits
    
    """
    
    # Photometric window function
    x_min = (z - z_i - z_bias)/((1.0+z)*sig_z*np.sqrt(2.0))
    x_max = (z - z_f - z_bias)/((1.0+z)*sig_z*np.sqrt(2.0))
    F_z = 0.5*(erf(x_min) - erf(x_max))
    
    # Normalization
    norm_const = np.dot(dz, F_z)
    
    # Window function 
    
    W_z_bin = bias*F_z/norm_const

    return W_z_bin, norm_const


#==================================================================================================
#==================================================================================================
# Function that calculates C_l,i
def C_l_i(z_i,z_f, sig_z, z_bias, bias, Omega_m_var , sig_8_var):
    """
    Function that calculates the C_l in a bin
    -----------------
    Inputs:
    z_i : lower redshift limit of the bin
    z_f : upper redshift limit of the bin
    sig_z : photometric error
    z_bias : redshift bias
    bias: constant bias factor in a bin
    Omega_m_var: Omega matter - can change
    sig_8_var : Sigma_8 parameter - can change
    --------------
    Returns:
    ls and C_l  in a bin i
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
    W_sq = (W_z_clust(zarray, dzarray, z_i, z_f, sig_z, z_bias, bias)[0])**2.0
    #===================================================================================
    #===================================================================================
    #Calculate Hubble parameter and comoving distance
    Hubble = cosmo.H_z
    # Get comoving distance - in Mpc/h
    chis = cosmo.chi
    #========================================================
    # Get the full prefactor of the integral
    prefact = W_sq*Hubble/(chis**2.0)
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
# =======================================================================================
# =======================================================================================
#
#
#
#
# =======================================================================================
# =======================================================================================
#
# Derivatives with respect to the cosmological parameters - matter density and sigma_8
#
# Derivative with respect to matter density
def matter_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m , sig_8):
    """
    Function that calculates the derivative of C_l with respect to matter density
    -----------------
    Inputs:
    z_i : Lower limit of the redshift bin
    z_f : Upper limit of the redshift bin
    sig_z : photometric error
    z_bias : redshift bias
    bias: constant bias factor in a bin
    Omega_m: Omega matter
    sig_8: Sigma_8 parameter
    ---------------
    Returns:
    derivative w/r to matter of C_l in a bin i
    """
    alpha_m = Omega_m/10.0
    
    C_mat_1 = C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m+alpha_m , sig_8)[1]
    C_mat_2 = C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m-alpha_m , sig_8)[1]
    
    mat_der = (C_mat_1 - C_mat_2)/(2.0*alpha_m)
    return mat_der
#    
#===============================================================================================
#===============================================================================================
# Derivative with respect to sigma_8   
def sigma_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m , sig_8):
    """
    Function that calculates the derivative of C_l with respect to sigma_8 
    -----------------
    Inputs:
    z_i : Lower limit of the redshift bin
    z_f : Upper limit of the redshift bin
    sig_z : photometric error
    bias: constant bias factor in a bin
    Omega_m: Omega matter
    sig_8: Sigma_8 parameter
    ---------------
    Returns:
    derivative w/r to matter of C_l betwenn two bins, i and j
    """
    
    alpha_s = sig_8/10.0
    
    C_sig_1 = C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m, sig_8+alpha_s)[1]
    C_sig_2 = C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m , sig_8-alpha_s)[1]
    
    sig_der = (C_sig_1 - C_sig_2)/(2.0*alpha_s)
    return sig_der
#
# ==============================================================================================
# ==============================================================================================
## Derivative with respect to galaxy bias in a bin

def bias_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m , sig_8):
    """
    Function that calculates the derivative of C_l with respect to sigma_8 
    -----------------
    Inputs:
    z_i : Lower limit of the redshift bin
    z_f : Upper limit of the redshift bin
    sig_z : photometric error scatter
    z_bias : photometric error bias
    bias: constant bias factor in a bin
    Omega_m: Omega matter
    sig_8: Sigma_8 parameter
    ---------------
    Returns:
    derivative w/r to matter of C_l betwenn two bins, i and j
    """
    
    alpha_b = bias/10.0
    
    C_bias_1 = C_l_i(z_i, z_f, sig_z, z_bias, bias+alpha_b, Omega_m, sig_8)[1]
    C_bias_2 = C_l_i(z_i, z_f, sig_z, z_bias, bias-alpha_b, Omega_m, sig_8)[1]
    
    bias_der = (C_bias_1 - C_bias_2)/(2.0*alpha_b)
    
    return bias_der
#
# ============================================================================================
# ============================================================================================
# 
# Derivatives with respect 

def z_bias_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m , sig_8):
    
    alpha_z_b = 0.001
    
    C_z_bias_1 = C_l_i(z_i, z_f, sig_z, z_bias+alpha_z_b, bias, Omega_m, sig_8)[1]
    C_z_bias_2 = C_l_i(z_i, z_f, sig_z, z_bias-alpha_z_b, bias, Omega_m , sig_8)[1]
    
    z_bias_der = (C_z_bias_1 - C_z_bias_2)/(2.0*alpha_z_b)
    return z_bias_der
    
#=============================================================================================
def sig_z_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m , sig_8):
    
    alpha_sig = sig_z/10.0
    
    C_sig_z_1 = C_l_i(z_i, z_f, sig_z + alpha_sig, z_bias, bias, Omega_m, sig_8)[1]
    C_sig_z_2 = C_l_i(z_i, z_f, sig_z - alpha_sig, z_bias, bias, Omega_m , sig_8)[1]
    
    sig_z_der = (C_sig_z_1 - C_sig_z_2)/(2.0*alpha_sig)
    return sig_z_der
#
# ============================================================================================
# ============================================================================================



# ========================================================================================================

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
        for s in ragne(n_size):
            if (np.sign(search_array[s])<=0.0):
                ell_break = 0.5*(ells[s]+ells[s-1])
                break       
    #========================================================         
    
    return alpha, ell_break 

# ===========================================================================================================
# ===========================================================================================================


def Fish_single_bin(z_i, z_f, sig_z, z_bias, bias, f_sky, N_gal):
    """
    Calculates and returns the Fisher matrix for a single bin
    ------------------------------------------------
    Inputs:
    z_i : Lower limit of the redshift bin
    z_f : Upper limit of the redshift bin 
    sig_z : photometric error spread 
    z_bias : redshift bias
    bias : constant galaxy bias factor in the bin
    f_sky : fraction of the sky the survey covers
    N_gal : number of galaxies in the bin
    
    ------------------------------------------------
    Outputs:
    Fisher matrix for a single bin 
    """
    
    #Some constants 
    
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
    
    z_mean = 0.5*(z_i+z_f)
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
    
    C_ell_1 = C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m, sigma_8)[1] #C_ell's
    dC_ldOm_1 = matter_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m, sigma_8) #Matter derivative
    dC_ldsig8_1 = sigma_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m, sigma_8) #Sigma_8 derivative
    dC_ldbias_1 = bias_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m, sigma_8) # Galaxy bias derivative
    #dC_ldz_bias_1, dC_ldsig_z_1 = C_photo_der(z_i, z_f, sig_z, z_bias, bias) #Derivative w/r to the two cosmological parameters
    dC_ldz_bias_1  = z_bias_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m , sigma_8)
    dC_ldsig_z_1 = sig_z_der_C_l_i(z_i, z_f, sig_z, z_bias, bias, Omega_m , sigma_8) 
    #================================================================================
    
    #=================================================================================
    #=================================================================================
    #Find the breaking ells now 

    al_1, l_break_1 = breaking_ell(ell_lin, dC_ldOm_1)
    al_2, l_break_2 = breaking_ell(ell_lin, dC_ldz_bias_1)
    al_3, l_break_3 = breaking_ell(ell_lin, dC_ldsig_z_1)
            
        
    #==================================================================================
    ls = np.arange(10, 2000, dtype=np.float64)
    
    #Initialize
    
    C_ell = np.zeros(np.size(ls)) #C_ell's
    dC_ldOm = np.zeros(np.size(ls)) # matter derivative
    dC_ldsig8 = np.zeros(np.size(ls)) #sigma_8 derivative
    dC_ldbias = np.zeros(np.size(ls)) #galaxy bias derivative
    dC_ldz_bias = np.zeros(np.size(ls)) #z_bias derivtive
    dC_ldsig_z = np.zeros(np.size(ls)) #redshift error spread derivative
    
    #=================================================================================
    # Interpolate 
    
    C_l_matr_interp = UnivariateSpline(ell_lin, np.log10(C_ell_1+ 1.0e-20), s=0.0)
    C_om_mat_interp = UnivariateSpline(ell_lin, np.log10(abs(dC_ldOm_1+ 1.0e-20)), s=0.0)
    C_sig_interp = UnivariateSpline(ell_lin, np.log10(dC_ldsig8_1+1.0e-20), s=0.0)
    C_bias_interp = UnivariateSpline(ell_lin, np.log10(dC_ldbias_1+1.0e-20),s=0.0)
    C_zbias_interp = UnivariateSpline(ell_lin, np.log10(abs(dC_ldz_bias_1+1.0e-20)), s=0.0)
    C_sigz_interp = UnivariateSpline(ell_lin, np.log10(abs(dC_ldsig_z_1+1.0e-20)), s=0.0)
    
    #===============================================================================
    #populate
    
    for k, l in enumerate(ls):
    	ell = np.log10(float(l))
        C_ell[k]  = 10.0**(C_l_matr_interp(ell))
        dC_ldsig8[k] = 10.0**(C_sig_interp(ell))
        dC_ldbias[k] = 10.0**(C_bias_interp(ell))

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
                dC_ldz_bias[k] = -(10.0**C_zbias_interp(ell))
            else:
                dC_ldz_bias[k] = (10.0**C_zbias_interp(ell))
        else:
            if (ell < l_break_2):
                dC_ldz_bias[k] = (10.0**C_zbias_interp(ell))
            else:
                dC_ldz_bias[k] = -(10.0**C_zbias_interp(ell))
                
        if (al_3 == 0):
            if (ell < l_break_3):
                dC_ldsig_z[k] = -(10.0**C_sigz_interp(ell))
            else:
                dC_ldsig_z[k] = (10.0**C_sigz_interp(ell))
        else:
            if (ell < l_break_3):
                dC_ldsig_z[k] = (10.0**C_sigz_interp(ell))
            else:
                dC_ldsig_z[k] = -(10.0**C_sigz_interp(ell))

    # =================================================================================
    # =================================================================================    
    # =================================================================================
    ls = ls[:l_max-9]
    C_ell = C_ell[:l_max-9]
    dC_ldOm = dC_ldOm[:l_max-9]
    dC_ldsig8 = dC_ldsig8[:l_max-9]
    dC_ldbias = dC_ldbias[:l_max-9]
    dC_ldz_bias = dC_ldz_bias[:l_max-9]
    dC_ldsig_z = dC_ldsig_z[:l_max-9]
    
    #Create arrays with sigma^2
    sigma_sq = (2.0/(f_sky*(2.0*ls + 1.0)))*((C_ell + 1.0/n_bin)**2.0)
    #and its inverse - it will be useful 
    inv_sigma = 1.0/sigma_sq # inverse of sigma square - that's what we want
    
    #===============================================================================
    #===============================================================================
    # Calculation of the elements of the Fisher matrix
    
    Fish = np.zeros([5,5])
    # 0 = matter, 1 = sigma_8, 2 = bias, 3 = z_bias, 4 = sigma_z
    
    # Diagonal terms first
    Fish[0,0] = sum(inv_sigma*(dC_ldOm**2.0))
    Fish[1,1] = sum(inv_sigma*(dC_ldsig8**2.0))
    Fish[2,2] = sum(inv_sigma*(dC_ldbias**2.0))
    Fish[3,3] = sum(inv_sigma*(dC_ldz_bias**2.0))
    Fish[4,4] = sum(inv_sigma*(dC_ldsig_z**2.0))
    
    # Non-diagonal terms
    Fish[0,1] = Fish[1,0] = sum(inv_sigma*dC_ldOm*dC_ldsig8)
    Fish[0,2] = Fish[2,0] = sum(inv_sigma*dC_ldOm*dC_ldbias)
    Fish[0,3] = Fish[3,0] = sum(inv_sigma*dC_ldOm*dC_ldz_bias)
    Fish[0,4] = Fish[4,0] = sum(inv_sigma*dC_ldOm*dC_ldsig_z)
    
    Fish[1,2] = Fish[2,1] = sum(inv_sigma*dC_ldsig8*dC_ldbias)
    Fish[1,3] = Fish[3,1] = sum(inv_sigma*dC_ldsig8*dC_ldz_bias)
    Fish[1,4] = Fish[4,1] = sum(inv_sigma*dC_ldsig8*dC_ldsig_z)
    
    Fish[2,3] = Fish[3,2] = sum(inv_sigma*dC_ldbias*dC_ldz_bias)
    Fish[2,4] = Fish[4,2] = sum(inv_sigma*dC_ldbias*dC_ldsig_z)
    
    Fish[3,4] = Fish[4,3] = sum(inv_sigma*dC_ldz_bias*dC_ldsig_z)
    
    return Fish   




































