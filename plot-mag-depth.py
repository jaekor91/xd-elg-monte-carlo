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




lw = 3
lw2 = 1.5
mag_bins = np.arange(21, 26, 0.1)

# All objects
gdepths_DR3 = []
rdepths_DR3 = []
zdepths_DR3 = []
gdepths_DR3_med = []
rdepths_DR3_med = []
zdepths_DR3_med = []

gdepths_DR5 = []
rdepths_DR5 = []
zdepths_DR5 = []
gdepths_DR5_med = []
rdepths_DR5_med = []
zdepths_DR5_med = []
for i, fnum in enumerate([2, 3, 4]):
    # DR3 data
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR3("DR3-Tractor-D2f%d.fits"%fnum)
    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (tycho==0)
    gf_err, rf_err, zf_err = grz_flux_error([givar[ibool], rivar[ibool], zivar[ibool]])
    gdepths_DR3.append(mag_depth_Xsigma(gf_err))
    rdepths_DR3.append(mag_depth_Xsigma(rf_err))
    zdepths_DR3.append(mag_depth_Xsigma(zf_err))
    gdepths_DR3_med.append(np.median(gdepths_DR3[i][gdepths_DR3[i]>0]))
    rdepths_DR3_med.append(np.median(rdepths_DR3[i][rdepths_DR3[i]>0]))
    zdepths_DR3_med.append(np.median(zdepths_DR3[i][zdepths_DR3[i]>0]))
    

    # DR5 data
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5("DR5-Tractor-D2f%d.fits"%fnum)
    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (tycho==0)
    gf_err, rf_err, zf_err = grz_flux_error([givar[ibool], rivar[ibool], zivar[ibool]])
    gdepths_DR5.append(mag_depth_Xsigma(gf_err))
    rdepths_DR5.append(mag_depth_Xsigma(rf_err))
    zdepths_DR5.append(mag_depth_Xsigma(zf_err))
    gdepths_DR5_med.append(np.median(gdepths_DR5[i][gdepths_DR5[i]>0]))
    rdepths_DR5_med.append(np.median(rdepths_DR5[i][rdepths_DR5[i]>0]))
    zdepths_DR5_med.append(np.median(zdepths_DR5[i][zdepths_DR5[i]>0]))
    
    

# r_exp cut
figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
for i in range(3):
    ax_list[i].hist(gdepths_DR3[i], bins=mag_bins, histtype="step", color="green", label="DR3 g", normed=True, lw=lw2)
    ax_list[i].hist(gdepths_DR5[i], bins=mag_bins, histtype="stepfilled", color="green", label="DR5 g", normed=True, alpha=0.5, lw=lw2)
    ax_list[i].hist(rdepths_DR3[i], bins=mag_bins, histtype="step", color="red", label="DR3 r", normed=True, lw=lw2)
    ax_list[i].hist(rdepths_DR5[i], bins=mag_bins, histtype="stepfilled", color="red", label="DR5 r", normed=True, alpha=0.5, lw=lw2)
    ax_list[i].hist(zdepths_DR3[i], bins=mag_bins, histtype="step", color="purple", label="DR3 z", normed=True, lw=lw2)
    ax_list[i].hist(zdepths_DR5[i], bins=mag_bins, histtype="stepfilled", color="purple", label="DR5 z", normed=True, alpha=0.5, lw=lw2)
    ax_list[i].axvline(x=gdepths_DR3_med[i], c="green", ls="--", lw=lw)
    ax_list[i].axvline(x=gdepths_DR5_med[i], c="green", lw=lw)
    ax_list[i].axvline(x=rdepths_DR3_med[i], c="red", ls="--", lw=lw)
    ax_list[i].axvline(x=rdepths_DR5_med[i], c="red", lw=lw)
    ax_list[i].axvline(x=zdepths_DR3_med[i], c="purple", ls="--", lw=lw)
    ax_list[i].axvline(x=zdepths_DR5_med[i], c="purple", lw=lw)
    ax_list[i].set_xlim([21.5, 25.5])
    ax_list[i].legend(loc="upper left")
    ax_list[i].set_xlabel("mag", fontsize=20)
    ax_list[i].set_title("Field %d"%(i+2), fontsize=20)
plt.suptitle("All objects 5-sig depths", fontsize=30, y=1.05)
plt.savefig("depths-all-objects.png", dpi=400, bbox_inches="tight")
plt.close()





# With exponential cut
gdepths_DR3 = []
rdepths_DR3 = []
zdepths_DR3 = []
gdepths_DR3_med = []
rdepths_DR3_med = []
zdepths_DR3_med = []

gdepths_DR5 = []
rdepths_DR5 = []
zdepths_DR5 = []
gdepths_DR5_med = []
rdepths_DR5_med = []
zdepths_DR5_med = []
for i, fnum in enumerate([2, 3, 4]):
    # DR3 data
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR3("DR3-Tractor-D2f%d.fits"%fnum)
    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (r_exp>0.35) & (r_exp<0.55) & (tycho==0)
    gf_err, rf_err, zf_err = grz_flux_error([givar[ibool], rivar[ibool], zivar[ibool]])
    gdepths_DR3.append(mag_depth_Xsigma(gf_err))
    rdepths_DR3.append(mag_depth_Xsigma(rf_err))
    zdepths_DR3.append(mag_depth_Xsigma(zf_err))
    gdepths_DR3_med.append(np.median(gdepths_DR3[i][gdepths_DR3[i]>0]))
    rdepths_DR3_med.append(np.median(rdepths_DR3[i][rdepths_DR3[i]>0]))
    zdepths_DR3_med.append(np.median(zdepths_DR3[i][zdepths_DR3[i]>0]))
    

    # DR5 data
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5("DR5-Tractor-D2f%d.fits"%fnum)
    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (r_exp>0.35) & (r_exp<0.55)  & (tycho==0)
    gf_err, rf_err, zf_err = grz_flux_error([givar[ibool], rivar[ibool], zivar[ibool]])
    gdepths_DR5.append(mag_depth_Xsigma(gf_err))
    rdepths_DR5.append(mag_depth_Xsigma(rf_err))
    zdepths_DR5.append(mag_depth_Xsigma(zf_err))
    gdepths_DR5_med.append(np.median(gdepths_DR5[i][gdepths_DR5[i]>0]))
    rdepths_DR5_med.append(np.median(rdepths_DR5[i][rdepths_DR5[i]>0]))
    zdepths_DR5_med.append(np.median(zdepths_DR5[i][zdepths_DR5[i]>0]))
    

figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
for i in range(3):
    ax_list[i].hist(gdepths_DR3[i], bins=mag_bins, histtype="step", color="green", label="DR3 g", normed=True, lw=lw2)
    ax_list[i].hist(gdepths_DR5[i], bins=mag_bins, histtype="stepfilled", color="green", label="DR5 g", normed=True, alpha=0.5, lw=lw2)
    ax_list[i].hist(rdepths_DR3[i], bins=mag_bins, histtype="step", color="red", label="DR3 r", normed=True, lw=lw2)
    ax_list[i].hist(rdepths_DR5[i], bins=mag_bins, histtype="stepfilled", color="red", label="DR5 r", normed=True, alpha=0.5, lw=lw2)
    ax_list[i].hist(zdepths_DR3[i], bins=mag_bins, histtype="step", color="purple", label="DR3 z", normed=True, lw=lw2)
    ax_list[i].hist(zdepths_DR5[i], bins=mag_bins, histtype="stepfilled", color="purple", label="DR5 z", normed=True, alpha=0.5, lw=lw2)
    ax_list[i].axvline(x=gdepths_DR3_med[i], c="green", ls="--", lw=lw)
    ax_list[i].axvline(x=gdepths_DR5_med[i], c="green", lw=lw)
    ax_list[i].axvline(x=rdepths_DR3_med[i], c="red", ls="--", lw=lw)
    ax_list[i].axvline(x=rdepths_DR5_med[i], c="red", lw=lw)
    ax_list[i].axvline(x=zdepths_DR3_med[i], c="purple", ls="--", lw=lw)
    ax_list[i].axvline(x=zdepths_DR5_med[i], c="purple", lw=lw)
    ax_list[i].set_xlim([21.5, 25.5])
    ax_list[i].legend(loc="upper left")
    ax_list[i].set_xlabel("mag", fontsize=20)
    ax_list[i].set_title("Field %d"%(i+2), fontsize=20)
plt.suptitle("r_exp [0.35, 0.55] objects 5-sig depths", fontsize=30, y=1.05)
plt.savefig("depths-rexp.png", dpi=400, bbox_inches="tight")
plt.close()

