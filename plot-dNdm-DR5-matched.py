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



def load_tractor_DR5(fname):
    """
    Load select columns
    """
    tbl = load_fits_table(fname)
    ra, dec = load_radec(tbl)
    bid = tbl["brickid"]
    bp = tbl["brick_primary"]
    r_dev, r_exp = tbl["shapedev_r"], tbl["shapeexp_r"]
    gflux_raw, rflux_raw, zflux_raw = tbl["flux_g"], tbl["flux_r"], tbl["flux_z"]
    gflux, rflux, zflux = gflux_raw/tbl["mw_transmission_g"], rflux_raw/tbl["mw_transmission_r"],zflux_raw/tbl["mw_transmission_z"]
    givar, rivar, zivar = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    objtype = tbl["type"]
    tycho = tbl["TYCHOVETO"]
    D2matched = tbl["DEEP2_matched"]
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask, D2matched



areas = np.load("spec-area-DR5-matched.npy")
lw=3
lw2=2
mag_bins = np.arange(20, 25.05, 0.1)
gmag_nominal = 23.8
rmag_nominal = 23.4
zmag_nominal = 22.4
gmag_max = 24.25

for gmag_max in [24]:
    # All objects
    gmag = []
    rmag = []
    zmag = []
    for i, fnum in enumerate([2, 3, 4]):
        # DR5 data
        bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev,\
         r_exp, g_allmask, r_allmask, z_allmask, D2matched = load_tractor_DR5("DR5-matched-to-DEEP2-f%d-glim24p25.fits"%fnum)
        ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (tycho==0) & (gflux > mag2flux(gmag_max))
        gmag.append(flux2mag(gflux[ibool]))
        rmag.append(flux2mag(rflux[ibool]))
        zmag.append(flux2mag(zflux[ibool]))
        
    # Plot all objects 
    figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
    for i in range(3):
        numobjs = gmag[i].size
        ax_list[i].hist(gmag[i], bins=mag_bins, histtype="step", color="green", label="DR5 g", alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])
        ax_list[i].hist(rmag[i], bins=mag_bins, histtype="step", color="red", label="DR5 r", alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])
        ax_list[i].hist(zmag[i], bins=mag_bins, histtype="step", color="purple", label="DR5 z", alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])
        # Nominal depth
        ax_list[i].axvline(x=gmag_nominal, c="green", lw=lw, ls="--")    
        ax_list[i].axvline(x=rmag_nominal, c="red", lw=lw, ls="--")
        ax_list[i].axvline(x=zmag_nominal, c="purple", lw=lw, ls="--")
        ax_list[i].set_xlim([20., 25.])
        ax_list[i].legend(loc="upper left")
        ax_list[i].set_xlabel("mag", fontsize=20)
        ax_list[i].set_title("Field %d"%(i+2), fontsize=20)
    plt.suptitle("All objects g < 24", fontsize=30, y=1.05)
    plt.savefig("dNdm-all-objects-DR5-%d.png" % int(gmag_max), dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()


    mags_list = [gmag, rmag, zmag]
    mag_subtitles = ["g", "r", "z"]
    field_names = ["2", "3", "4"]
    nominal_depths = [gmag_nominal, rmag_nominal, zmag_nominal]
    colors = ["black", "red", "blue"]
    figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
    for i in range(3): # Index for filter
        for j in range(3): # Index for field
            m = mags_list[i][j]
            numobjs = m.size
            ax_list[i].hist(m, bins=mag_bins, histtype="step", color=colors[j], label="DR5 "+field_names[j], alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[j])
            # Nominal depth
        ax_list[i].axvline(x=nominal_depths[i], c="orange", lw=lw, ls="--")    
        ax_list[i].set_xlim([20., 25.])
        ax_list[i].legend(loc="upper left")
        ax_list[i].set_xlabel("mag", fontsize=20)
        ax_list[i].set_title(mag_subtitles[i], fontsize=20)
    plt.suptitle("All objects g < 24", fontsize=30, y=1.05)
    plt.savefig("dNdm-all-objects-DR5-by-field-%d.png" % int(gmag_max), dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()








    # ----- With exponential cut
    gmag = []
    rmag = []
    zmag = []
    for i, fnum in enumerate([2, 3, 4]):

        # DR5 data
        bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask, D2matched = load_tractor_DR5("DR5-matched-to-DEEP2-f%d-glim24p25.fits"%fnum)
        ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (r_exp>0.35) & (r_exp<0.55)  & (tycho==0) & (gflux > mag2flux(gmag_max))
        gmag.append(flux2mag(gflux[ibool]))
        rmag.append(flux2mag(rflux[ibool]))
        zmag.append(flux2mag(zflux[ibool]))
        
    # mag_bins = np.arange(20, 25.05, 0.1)
    figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
    for i in range(3):
        numobjs = gmag[i].size
        ax_list[i].hist(gmag[i], bins=mag_bins, histtype="step", color="green", label="DR5 g", alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])
        ax_list[i].hist(rmag[i], bins=mag_bins, histtype="step", color="red", label="DR5 r", alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])
        ax_list[i].hist(zmag[i], bins=mag_bins, histtype="step", color="purple", label="DR5 z", alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])
        # Nominal depth
        ax_list[i].axvline(x=gmag_nominal, c="green", lw=lw, ls="--")    
        ax_list[i].axvline(x=rmag_nominal, c="red", lw=lw, ls="--")
        ax_list[i].axvline(x=zmag_nominal, c="purple", lw=lw, ls="--")    
        ax_list[i].set_xlim([20., 25.])
        ax_list[i].legend(loc="upper left")
        ax_list[i].set_xlabel("mag", fontsize=20)
        ax_list[i].set_title("Field %d"%(i+2), fontsize=20)
    plt.suptitle("r_exp [0.35, 0.55] g < 24", fontsize=30, y=1.05)
    plt.savefig("dNdm-rexp-DR5-%d.png"% int(gmag_max), dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()




    mags_list = [gmag, rmag, zmag]
    mag_subtitles = ["g", "r", "z"]
    field_names = ["2", "3", "4"]
    nominal_depths = [gmag_nominal, rmag_nominal, zmag_nominal]
    colors = ["black", "red", "blue"]
    figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
    for i in range(3): # Index for filter
        for j in range(3): # Index for field
            m = mags_list[i][j]
            numobjs = m.size
            ax_list[i].hist(m, bins=mag_bins, histtype="step", color=colors[j], label="DR5 "+field_names[j], alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[j])
            # Nominal depth
        ax_list[i].axvline(x=nominal_depths[i], c="orange", lw=lw, ls="--")    
        ax_list[i].set_xlim([20., 25.])
        ax_list[i].legend(loc="upper left")
        ax_list[i].set_xlabel("mag", fontsize=20)
        ax_list[i].set_title(mag_subtitles[i], fontsize=20)
    plt.suptitle("r_exp [0.35, 0.55] objects g < 24", fontsize=30, y=1.05)
    plt.savefig("dNdm-rexp-DR5-by-field-%d.png" % int(gmag_max), dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()


        






