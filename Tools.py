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