import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys

import numpy as np
import matplotlib.pyplot as plt


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

def lnL_gauss(mu, sigma, data_pts):
    """
    Given the mean and the sigma of a gaussian return the lnL.
    Note that I am computing negative lnL.
    """
    return np.sum((data_pts-mu)**2/float(2.*sigma))



# Generate sample
mag_min = 23
mag_max = 24
xrz, ygr, gflux = mock_pop_1(int(1e5), mag2flux(mag_max))



# dsig = 2.5/15.
# sig_array = 10**np.arange(-2.25, 0.25+dsig/2., dsig)
# Nsig = 16
# Ndim_plot = 4 
# mu_fit_array = np.zeros(Nsig)

# fmin, fmax = 0, 1
# df = 2.5e-3
# fbins = np.arange(fmin, fmax+df/2., df)

# dx = 1e-4
# xgrid = np.arange(0., 1.5*fmax+df/2., dx)

# fig, ax_list = plt.subplots(4, 4, figsize = (24, 24))

# for i, sig in enumerate(sig_array):
#     # For each sigma, compute the best mu and then plot.    
#     lnLs = []
#     for mu_tmp in fbins:
#         lnLs.append(lnL_gauss(mu_tmp, sig, data_pts=gflux))
#     lnLs = np.asarray(lnLs)
    
#     mu_min_lnL = fbins[np.argmin(lnLs)]    
    
#     # Axes
#     ax_num1 = i//Ndim_plot
#     ax_num2 = i-ax_num1*Ndim_plot
#     ax_current = ax_list[ax_num1, ax_num2]
    
#     ax_current.semilogy(fbins, lnLs, label="log(-lnL)", color="green")
#     ax = ax_current.twinx()
    
#     # Plot gaussian
#     gauss = stats.norm.pdf(xgrid, loc=mu_min_lnL, scale=sig)
#     ax.hist(gflux, bins = fbins, histtype="step", normed=True, color="black")
#     ax.plot(xgrid, gauss, lw=1, c="blue", label="MLE")
#     plt.xlim([fmin, fmax])
#     plt.axvline(x=mu_min_lnL, c="red", ls="--",lw=1)
#     plt.axhline(y=np.max(gauss), c="red", ls="--",lw=1)
#     plt.title(r"$\sigma$ = %.4f" % sig, fontsize=25)
#     ax.legend(loc = "upper right", fontsize=15)
# plt.savefig("fit-gauss-to-power-4by4-panel.png", dpi=200, bbox_inches="tight")
# plt.close()




sigma_array = 10**np.arange(-2, 0.5+0.025/2., 0.025)
fmin, fmax = 0, 1
df = 1e-2
fbins = np.arange(fmin, fmax+df/2., df)

lnLs = np.zeros((sigma_array.size, fbins.size))

for i in range(sigma_array.size):
    for j in range(fbins.size):        
        lnLs[i, j]= lnL_gauss(fbins[j], sigma_array[i], data_pts=gflux)
        
fmin, fmax = 0, 1
df = 1e-2
fbins = np.arange(fmin, fmax+df/2., df)

ft_size = 25
figure = plt.figure(figsize=(7, 7))
plt.imshow(np.log(lnLs), interpolation="none")
plt.xticks(np.arange(6)*20, fbins[np.arange(6)*20])
plt.yticks(np.arange(6)*20, np.round(np.log10(sigma_array)[np.arange(6)*20], decimals=1))
plt.colorbar()
plt.title(r"$\log(-\ln L)$", fontsize =ft_size)
plt.xlabel(r"$\mu$", fontsize =ft_size)
plt.ylabel(r"$\log_{10}(\sigma)$", fontsize=ft_size)
# plt.show()
plt.savefig("fit-gauss-to-power-2D-lnL.png", dpi=200, bbox_inches="tight")
plt.close()

# idx_min = np.argmin(lnLs)
# sig_idx = idx_min//fbins.size
# mu_idx = idx_min-(sig_idx*fbins.size)

# fig, ax2 = plt.subplots()
# # Plot gaussian
# dx = 1e-4
# xgrid = np.arange(0., 1.5*fmax+df/2., dx)
# gauss = stats.norm.pdf(xgrid, loc=fbins[mu_idx], scale=sigma_array[sig_idx])
# ax2.plot(xgrid, gauss, lw=1, c="blue")
# df = 2.5e-3
# fbins = np.arange(fmin, fmax+df/2., df)
# ax2.hist(gflux, bins = fbins, histtype="step", normed=True, color="black")
# plt.xlim([fmin, fmax])
# plt.axvline(x=mu_max_lnL, c="red", ls="--",lw=1)
# plt.axhline(y=np.max(gauss), c="red", ls="--",lw=1)
# plt.show()
# plt.close()