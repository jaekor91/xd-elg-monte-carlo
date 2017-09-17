import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys

import numpy as np
import matplotlib.pyplot as plt

def lnL_gauss(mu, sigma, data_pts):
    """
    Given the mean and the sigma of a gaussian return the lnL.
    Note that I am computing negative lnL.
    """
    return np.sum((data_pts-mu)**2/float(2.*sigma**2) + np.log(sigma))

# Generate sample
gflux = np.random.normal(size=int(1e5), loc=mag2flux(23.5), scale=mag2flux(25.5))
# plt.hist(gflux, bins=fbins)
# plt.show()
# plt.close()

dsig = 2.5/15.
sig_array = 10**np.arange(-2.25, 0.25+dsig/2., dsig)
Nsig = 16
Ndim_plot = 4 
mu_fit_array = np.zeros(Nsig)

fmin, fmax = 0, 1
df = 2.5e-3
fbins = np.arange(fmin, fmax+df/2., df)

dx = 1e-4
xgrid = np.arange(0., 1.5*fmax+df/2., dx)


# Avoid negative values
# lnL_pedastall = 1e7

fig, ax_list = plt.subplots(4, 4, figsize = (24, 24))

for i, sig in enumerate(sig_array):
    # For each sigma, compute the best mu and then plot.    
    lnLs = np.zeros(fbins.size)
    for j, mu_tmp in enumerate(fbins):
        lnLs[j] = lnL_gauss(mu_tmp, sig, data_pts=gflux)
    lnLs = np.asarray(lnLs)
    
    mu_min_lnL = fbins[np.argmin(lnLs)]    
    
    # Axes
    ax_num1 = i//Ndim_plot
    ax_num2 = i-ax_num1*Ndim_plot
    ax_current = ax_list[ax_num1, ax_num2]
    
    ax_current.plot(fbins, lnLs, label="lnL", color="green")
#     ax_current.set_ylim([1e6, 1e9])
    ax = ax_current.twinx()
    
    # Plot gaussian
    gauss = stats.norm.pdf(xgrid, loc=mu_min_lnL, scale=sig)
    ax.hist(gflux, bins = fbins, histtype="step", normed=True, color="black")
    ax.plot(xgrid, gauss, lw=1, c="blue", label="MLE")
    plt.xlim([fmin, fmax])
    plt.axvline(x=mu_min_lnL, c="red", ls="--",lw=1)
    plt.axhline(y=np.max(gauss), c="red", ls="--",lw=1)
    plt.title(r"$\sigma$ = %.4f" % sig, fontsize=25)
    ax.legend(loc = "upper right", fontsize=15)
plt.tight_layout()
plt.savefig("fit-gauss-to-gauss-4by4-panel.png", dpi=200, bbox_inches="tight")
plt.close()

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
plt.imshow(lnLs, interpolation="none", vmin=np.min(lnLs), vmax=np.percentile(lnLs, 55))
plt.xticks(np.arange(6)*20, fbins[np.arange(6)*20])
plt.yticks(np.arange(6)*20, np.round(np.log10(sigma_array)[np.arange(6)*20], decimals=1))
plt.colorbar()
plt.title(r"$\ln L$", fontsize =ft_size)
plt.xlabel(r"$\mu$", fontsize =ft_size)
plt.ylabel(r"$\log(\sigma)$", fontsize=ft_size)
# plt.show()
plt.savefig("fit-gauss-to-gauss-2D-lnL.png", dpi=200, bbox_inches="tight")
plt.close()