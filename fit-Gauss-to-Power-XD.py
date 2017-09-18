import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys

import numpy as np
import matplotlib.pyplot as plt



import extreme_deconvolution as XD

def gen_init_mean_from_sample(Ndim, sample, K):
    """
    Ndim: Dimensionality
    sample [Nsample, Ndim]: Sample array
    K: Number of components
    """
    idxes = np.random.choice(range(sample.shape[0]), K, replace=False)
    init_mean = []
    for idx in idxes:
        init_mean.append([sample[idx]])
    return np.asarray(init_mean)


def gen_uni_init_amps(K):
    """
    K: Number of components
    """
    return np.asarray([1/float(K)]*K)


def gen_diag_init_covar(Ndim, var, K):
    """
    Genereate diagonal covariance matrices given var [Ndim]
    Ndim: Dimensionality
    K: Number of components
    """
    return np.asarray([np.diag(var)]*K)
    
def gen_diag_data_covar(Nsample, var):
    """
    Generate covar matrix for each of Nsample data points
    given diagonal variances of Ndim
    """
    return np.asarray([np.diag(var)]*Nsample)


# Sample from a power law distribution
def mock_pop_1(NSAMPLE, fmin):
    # Return a sample from the underlying mock population 1
    # There is no notion absolute number density here.
    # Single gaussian    
    scale_factor = 0.25
    mu = [0.55, 0.35]# mean
    cov = np.array([[0.05, 0.045],[0.045, 0.05]]) * scale_factor
    alpha = 3.5
    
    xrz, ygr = np.random.multivariate_normal(mu, cov, NSAMPLE).T
    gflux =  fmin * np.exp(-np.log(np.random.rand(NSAMPLE))/alpha)# 
    
    return xrz, ygr, gflux


mag_max = 24
mag_min = 23
xrz, ygr, gflux = mock_pop_1(int(1e4), mag2flux(mag_max))
gflux = gflux[gflux<mag2flux(mag_min)]


fbins = np.arange(mag2flux(mag_max), mag2flux(mag_min), 2.5e-3)
fbins_extended = np.arange(0.1, mag2flux(mag_min), 1e-3)
Ndim = 1 
Ntrial = 20 # Number of XD trials.
#Run the code 
ydata = gflux.reshape([gflux.size, 1])
ycovar = gen_diag_data_covar(gflux.size, var=[0])

for K in range(1, 27):
    lnL_best = -np.inf # Initially lnL is terrible
    init_mean = None
    for j in range(Ntrial):
        print "K, ntrial: %d, %d" % (K, j)
        var = [0.1**2]

        if init_mean is None:
            # Randomly pick K samples from the generated set.
            init_mean = gen_init_mean_from_sample(Ndim, gflux, K)
            init_amp = gen_uni_init_amps(K)
            init_covar = gen_diag_init_covar(Ndim, var, K)
            init_mean_tmp, init_amp_tmp, init_covar_tmp = init_mean, init_amp, init_covar
        else:
            # Randomly pick K samples from the generated set.
            init_mean_tmp = gen_init_mean_from_sample(Ndim, gflux, K)
            init_amp_tmp = gen_uni_init_amps(K)
            init_covar_tmp = gen_diag_init_covar(Ndim, var, K)

        fit_mean_tmp, fit_amp_tmp, fit_covar_tmp = np.copy(init_mean_tmp), np.copy(init_amp_tmp), np.copy(init_covar_tmp)
        #Set up your arrays: ydata has the data, ycovar the uncertainty covariances
        #initamp, initmean, and initcovar are initial guesses
        #get help on their shapes and other options using
        # ?XD.extreme_deconvolution

        lnL = XD.extreme_deconvolution(ydata, ycovar, fit_amp_tmp, fit_mean_tmp, fit_covar_tmp, w=0.01)
        
        if lnL > lnL_best:
            fit_mean, fit_amp, fit_covar = fit_mean_tmp, fit_amp_tmp, fit_covar_tmp
            init_mean, init_amp, init_covar = init_mean_tmp, init_amp_tmp, init_covar_tmp
    # After the search
    plt.hist(gflux, bins = fbins, histtype="step", normed=True, color="black") # plot histogram
    gauss_init = None
    for k in range(K):
        if gauss_init is None:
            gauss_fit = fit_amp[k]*stats.norm.pdf(fbins_extended, loc=fit_mean[k][0], scale=fit_covar[k][0,0])
            gauss_init = init_amp[k]*stats.norm.pdf(fbins_extended, loc=init_mean[k][0], scale=init_covar[k][0,0])        
            gauss0, gauss1 = gauss_init, gauss_fit
        else:
            gauss0 = init_amp[k]*stats.norm.pdf(fbins_extended, loc=init_mean[k][0], scale=init_covar[k][0,0]) 
            gauss1 = fit_amp[k]*stats.norm.pdf(fbins_extended, loc=fit_mean[k][0], scale=fit_covar[k][0,0])
            gauss_fit += gauss1
            gauss_init += gauss0
            
        plt.plot(fbins_extended, gauss1, lw=0.5, c="blue", alpha=0.75)
        plt.plot(fbins_extended, gauss0, lw=0.5, c="red", alpha=0.75)
    plt.plot(fbins_extended, gauss_fit, lw=1, c="blue")
    plt.plot(fbins_extended, gauss_init, lw=1, c="red")        
    plt.xlim([mag2flux(mag_max), mag2flux(mag_min)])
    plt.ylim([-2, 30])
    plt.savefig("fit-gauss-to-power-K%d.png"%K, dpi=200, bbox_inches="tight")
    plt.close()


