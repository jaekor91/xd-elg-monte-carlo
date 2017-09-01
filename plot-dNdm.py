import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys

import numpy as np
import matplotlib.pyplot as plt


def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)    


def load_tractor_DR3(fname):
    """
    Load select fields from combined Tractor (only) files.
    DR3 only. 
    """
    tbl = load_fits_table(fname)
    ra, dec = load_radec(tbl)
    bid = load_bid(tbl)
    bp = load_brick_primary(tbl)    
    r_dev, r_exp =load_shape(tbl)
    gflux_raw, rflux_raw, zflux_raw = load_grz_flux(tbl)
    gflux, rflux, zflux = load_grz_flux_dereddened(tbl)
    givar, rivar, zivar = load_grz_invar(tbl)
    g_allmask, r_allmask, z_allmask = load_grz_allmask(tbl)
    objtype = tbl["type"]
    tycho = tbl["TYCHOVETO"]
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask


def load_tractor_DR5(fname):
    """
    Load select columns
    """
    tbl = load_fits_table(fname)
    ra, dec = load_radec(tbl)
    bid = tbl["brickid"]
    bp = tbl["brick_primary"]
    r_dev, r_exp = tbl["shapedev_r"], tbl["shapeexp_r"]
    gflux, rflux, zflux = tbl["flux_g"], tbl["flux_r"], tbl["flux_z"]
    gflux_raw, rflux_raw, zflux_raw = gflux/tbl["mw_transmission_g"], rflux/tbl["mw_transmission_r"],zflux/tbl["mw_transmission_z"]
    givar, rivar, zivar = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    objtype = tbl["type"]
    tycho = tbl["TYCHOVETO"]    
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask


lw=3
lw2=2
mag_bins = np.arange(20, 25.05, 0.1)
gmag_nominal = 23.8
rmag_nominal = 23.4
zmag_nominal = 22.4
gmag_max = 25.

# All objects
gmag = []
rmag = []
zmag = []
for i, fnum in enumerate([2, 3, 4]):
    # DR5 data
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5("DR5-Tractor-D2f%d.fits"%fnum)
    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (tycho==0) & (gflux > mag2flux(gmag_max))
    gmag.append(flux2mag(gflux[ibool]))
    rmag.append(flux2mag(rflux[ibool]))
    zmag.append(flux2mag(zflux[ibool]))
    
# Plot all objects 
figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
for i in range(3):
    ax_list[i].hist(gmag[i], bins=mag_bins, histtype="step", color="green", label="DR5 g", alpha=1, lw=lw2)
    ax_list[i].hist(rmag[i], bins=mag_bins, histtype="step", color="red", label="DR5 r", alpha=1, lw=lw2)
    ax_list[i].hist(zmag[i], bins=mag_bins, histtype="step", color="purple", label="DR5 z", alpha=1, lw=lw2)
    # Nominal depth
    ax_list[i].axvline(x=gmag_nominal, c="green", lw=lw, ls="--")    
    ax_list[i].axvline(x=rmag_nominal, c="red", lw=lw, ls="--")
    ax_list[i].axvline(x=zmag_nominal, c="purple", lw=lw, ls="--")
    ax_list[i].set_xlim([20., 25.])
    ax_list[i].legend(loc="upper left")
    ax_list[i].set_xlabel("mag", fontsize=20)
    ax_list[i].set_title("Field %d"%(i+2), fontsize=20)
plt.suptitle("All objects mags", fontsize=30, y=1.05)
plt.savefig("dNdm-all-objects.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()



# With exponential cut
gmag = []
rmag = []
zmag = []
for i, fnum in enumerate([2, 3, 4]):

    # DR5 data
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5("DR5-Tractor-D2f%d.fits"%fnum)
    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (r_exp>0.35) & (r_exp<0.55)  & (tycho==0) & (gflux > mag2flux(gmag_max))
    gmag.append(flux2mag(gflux[ibool]))
    rmag.append(flux2mag(rflux[ibool]))
    zmag.append(flux2mag(zflux[ibool]))
    
mag_bins = np.arange(20, 25.05, 0.2)
figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
for i in range(3):
    ax_list[i].hist(gmag[i], bins=mag_bins, histtype="step", color="green", label="DR5 g", alpha=1, lw=lw2)
    ax_list[i].hist(rmag[i], bins=mag_bins, histtype="step", color="red", label="DR5 r", alpha=1, lw=lw2)
    ax_list[i].hist(zmag[i], bins=mag_bins, histtype="step", color="purple", label="DR5 z", alpha=1, lw=lw2)
    # Nominal depth
    ax_list[i].axvline(x=gmag_nominal, c="green", lw=lw, ls="--")    
    ax_list[i].axvline(x=rmag_nominal, c="red", lw=lw, ls="--")
    ax_list[i].axvline(x=zmag_nominal, c="purple", lw=lw, ls="--")    
    ax_list[i].set_xlim([20., 25.])
    ax_list[i].legend(loc="upper left")
    ax_list[i].set_xlabel("mag", fontsize=20)
    ax_list[i].set_title("Field %d"%(i+2), fontsize=20)
plt.suptitle("r_exp [0.35, 0.55] objects magss", fontsize=30, y=1.05)
plt.savefig("dNdm-rexp.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()