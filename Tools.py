#=============================================================================#
#=============================================================================#
#============== Tools.py - Tools for Fisher forecasts ========================#
#=============================================================================#
#=================  Author: Dimitrios Tanoglidis =============================#
#============================ September 2018 =================================#
#
#
# Here we put together some tools needed in order to play with Fisher matrices#
# ============================================================================#
#
#
# Import stuff

import numpy as np
import scipy as sp

#=============================================================================#

def marginalize(Fisher):
    """
    Function that gets as input a Fisher matrix of size 2N_bins + 3, where N_bins the number of bins,
    and returns the Fisher (sub)matrix for the set of cosmological parameters Omega_m and sigma_8.
    These have to be the first two parameters of the original matrix.
    ----------------------------------------------------------------
    Inputs:
    Fisher : (2N_bins + 3)*(2N_bins +3) Fisher matrix. Contains two cosmological parameters,
    one photo-z scatter parameter, N_bins photo-z bias parameters and N_bins galaxy bias parameters
    ----------------------------------------------------------------
    Returns:
    The Fisher matrix for Omega_m - sigma_8
    """
    
    #Covariance matrix - the inverse of the Fisher matrix
    
    Cov_mat = sp.linalg.inv(Fisher)
    
    #Now create a new, empty 2*2 cov sub-matrix for the cosmological parameters parameters only
    
    Cov_submatrix = np.zeros([2,2])
    
    Cov_submatrix[0,0] = Cov_mat[0,0]
    Cov_submatrix[1,1] = Cov_mat[1,1]
    Cov_submatrix[0,1] = Cov_submatrix[1,0] =Cov_mat[0,1] 
        
    # Now invert the submatrix to get the new Fisher matrix
    
    Fish_marg = sp.linalg.inv(Cov_submatrix)
    
    #Make sure that it is diagonal 
    Fish_marg[0,1] = Fish_marg[1,0]
    
    return Fish_marg   

#==========================================================================================#
#==========================================================================================#
#==========================================================================================#
#==========================================================================================#
#
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
# Function that calculates C_l,i for e specific ell

def C_l_specif(z_i,z_f, sig_z, z_bias, bias, Omega_m_var , sig_8_var, ell):
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
    
    w = np.ones(chis.shape)
    k=(ell+0.5)/chis
    w[:]=1
    w[k<1e-4]=0
    w[k>=kmax]=0
    c_ell = np.dot(dzarray, w*cosmo.PK.P(zarray, k, grid=False)*prefact)
    
    #===========================
    # Retrurn C_ell
    
    
    return  c_ell


