import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)


def load_tractor_DR5_matched_to_DEEP2(fname):
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
    B, R, I = tbl["BESTB"], tbl["BESTR"], tbl["BESTI"]
    cn = tbl["cn"]
    w = tbl["TARG_WEIGHT"]
    red_z, z_err = tbl["RED_Z"], tbl["Z_ERR"]
    oii, oii_err = tbl["OII_3727"], tbl["OII_3727_err"]
    D2matched = tbl["DEEP2_matched"]
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, cn, w, red_z, z_err, oii, oii_err, D2matched




def load_DEEP2(fname, ibool=None):
    tbl = load_fits_table(fname)
    if ibool is not None:
        tbl = tbl[ibool]

    ra, dec = tbl["RA_DEEP"], tbl["DEC_DEEP"]
    tycho = tbl["TYCHOVETO"]
    B, R, I = tbl["BESTB"], tbl["BESTR"], tbl["BESTI"]
    cn = tbl["cn"]
    w = tbl["TARG_WEIGHT"]
    BRI_cut = tbl["BRI_cut"]
    return ra, dec, tycho, B, R, I, cn, w, BRI_cut



d2_dir = "../data-repository/DEEP2/photo-redz-oii/"


for fnum in [2, 3, 4]:
    print("Field %d" % fnum)
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
    rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, cn, w, red_z, z_err, oii, oii_err, D2matched\
        = load_tractor_DR5_matched_to_DEEP2("DR5-matched-to-DEEP2-f%d-glim25.fits" % fnum)
    gmag, rmag, zmag = flux2mag(gflux), flux2mag(rflux), flux2mag(zflux)
    ygr = gmag - rmag
    xrz = rmag - zmag

    print "Anticipated uncertainty."
    ibroad = broad_cut(gmag, rmag, zmag) # & (gflux>0) & (rflux>0) & (zflux>0)

    mag_min, mag_max= 10, 25
    ibool = (gflux > mag2flux(mag_max)) & (gflux < mag2flux(mag_min))
    print "Fraction unmatched [%.1f, %.1f]  before/after design-space cut: %.2f/%.2f (percent)" % (mag_min, mag_max, (np.sum(D2matched[ibool]==0)/float(D2matched[ibool].size) * 100), (np.sum(D2matched[np.logical_and(ibool, ibroad)]==0)/float(D2matched[np.logical_and(ibool, ibroad)].size) * 100))

    mag_min, mag_max= 23, 25
    ibool = (gflux > mag2flux(mag_max)) & (gflux < mag2flux(mag_min))
    print "Fraction unmatched [%.1f, %.1f]  before/after design-space cut: %.2f/%.2f (percent)" % (mag_min, mag_max, (np.sum(D2matched[ibool]==0)/float(D2matched[ibool].size) * 100), (np.sum(D2matched[np.logical_and(ibool, ibroad)]==0)/float(D2matched[np.logical_and(ibool, ibroad)].size) * 100))

    mag_min, mag_max= 23, 24
    ibool = (gflux > mag2flux(mag_max)) & (gflux < mag2flux(mag_min))
    print "Fraction unmatched [%.1f, %.1f]  before/after design-space cut: %.2f/%.2f (percent)" % (mag_min, mag_max, (np.sum(D2matched[ibool]==0)/float(D2matched[ibool].size) * 100), (np.sum(D2matched[np.logical_and(ibool, ibroad)]==0)/float(D2matched[np.logical_and(ibool, ibroad)].size) * 100))
    print "For modeling purpose, categorize unmatched objects to be part of the non-ELG set."


    # Plot grz of unmatched objects
    dm = 0.1
    lw = 2
    mag_bins = np.arange(19, 26.+dm/2., dm)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3 , figsize=(15, 5))
    ibool = D2matched==0
    if np.sum(ibool) < 1000:
        dm2 = 0.25
    else:
        dm2 = 0.1

    mag_bins2 = np.arange(19, 26.+dm2/2., dm2)


    ax1.hist(gmag[ibool], bins=mag_bins2, histtype="stepfilled", alpha=0.5, lw=lw, label="g - unmatched", color="green", normed=True)
    ax1.hist(gmag, bins=mag_bins, histtype="step", lw=lw, label="g", color="green", normed=True)
    ax1.set_xlim([19, 26])
    ax1.legend(loc="upper left")
    ax1.set_xlabel("mag")

    ax2.hist(rmag[ibool], bins=mag_bins2, histtype="stepfilled", alpha=0.5, lw=lw, label="r - unmatched", color="red", normed=True)
    ax2.hist(rmag, bins=mag_bins, histtype="step", lw=lw, label="r", color="red", normed=True)
    ax2.set_xlim([19, 26])
    ax2.legend(loc="upper left")
    ax2.set_xlabel("mag")    

    ax3.hist(zmag[ibool], bins=mag_bins2, histtype="stepfilled", alpha=0.5, lw=lw, label="z- unmatched", color="purple", normed=True)
    ax3.hist(zmag, bins=mag_bins, histtype="step", lw=lw, label="z", color="purple", normed=True)
    ax3.set_xlim([19, 26])
    ax3.legend(loc="upper left")
    ax3.set_xlabel("mag")    

    # plt.show()
    plt.savefig("dNdm-f%d-DR5-grz-matched-vs-unmatched.png" %fnum, dpi=200, bbox_inches="tight")
    plt.close()



    # Plot BRI of matched objects 19, 26 
    dm = 0.1
    dm2 = 0.1
    lw = 2
    mag_bins = np.arange(19, 26.+dm/2., dm)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3 , figsize=(15, 5))
    ibool = D2matched==1
    ax1.hist(B[ibool], bins=mag_bins, histtype="stepfilled", alpha=0.5, lw=lw, label="B", color="green", normed=True)
    ax1.set_xlim([19, 26])
    ax1.legend(loc="upper left")
    ax1.set_xlabel("mag")

    ax2.hist(R[ibool], bins=mag_bins, histtype="stepfilled", alpha=0.5, lw=lw, label="R", color="red", normed=True)
    ax2.set_xlim([19, 26])
    ax2.legend(loc="upper left")
    ax2.set_xlabel("mag")

    ax3.hist(I[ibool], bins=mag_bins, histtype="stepfilled", alpha=0.5, lw=lw, label="I", color="purple", normed=True)
    ax3.set_xlim([19, 26])
    ax3.legend(loc="upper left")
    ax3.set_xlabel("mag")    

    # plt.show()
    plt.savefig("dNdm-f%d-DEEP2-BRI-matched-mag19to26.png" %fnum, dpi=200, bbox_inches="tight")    
    plt.close()


    # Plot BRI of matched objects [21, 28] 
    dm = 0.1
    dm2 = 0.1
    lw = 2
    mag_bins = np.arange(21, 28.+dm/2., dm)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3 , figsize=(15, 5))
    ibool = D2matched==1
    ax1.hist(B[ibool], bins=mag_bins, histtype="stepfilled", alpha=0.5, lw=lw, label="B", color="green")
    ax1.set_xlim([21, 28])
    ax1.legend(loc="upper left")
    ax1.set_xlabel("mag")

    ax2.hist(R[ibool], bins=mag_bins, histtype="stepfilled", alpha=0.5, lw=lw, label="R", color="red")
    ax2.set_xlim([21, 28])
    ax2.legend(loc="upper left")
    ax2.set_xlabel("mag")

    ax3.hist(I[ibool], bins=mag_bins, histtype="stepfilled", alpha=0.5, lw=lw, label="I", color="purple")
    ax3.set_xlim([21, 28])
    ax3.legend(loc="upper left")
    ax3.set_xlabel("mag")

    plt.savefig("dNdm-f%d-DEEP2-BRI-matched-mag21to28.png" %fnum, dpi=200, bbox_inches="tight")    
    # plt.show()
    plt.close()




    # Load DEEP2 data
    ra, dec, tycho, B, R, I, cn, w, BRI_cut= load_DEEP2("unmatched-deep2-f%d-photo-redz-oii.fits"%fnum)
    ra, dec, tycho, B_all, R_all, I_all, cn_all, w_all, BRI_cut_all = load_DEEP2(d2_dir+"deep2-f%d-photo-redz-oii.fits"%fnum)


    # Plot grz of unmatched objects
    dm = 0.1
    dm2 = 0.1
    lw = 2
    mag_bins = np.arange(21, 28.+dm/2., dm)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3 , figsize=(15, 5))

    ax1.hist(B_all, bins=mag_bins, histtype="step", lw=lw, label="B", color="green")
    ax1.hist(B, bins=mag_bins, histtype="stepfilled",alpha=0.5, lw=lw, label="B-", color="green")
    ax1.set_xlim([21, 28])
    ax1.legend(loc="upper right")
    ax1.set_xlabel("mag")

    ax2.hist(R_all, bins=mag_bins, histtype="step", lw=lw, label="R", color="red")
    ax2.hist(R, bins=mag_bins, histtype="stepfilled", alpha=0.5, lw=lw, label="R-", color="Red")
    ax2.set_xlim([21, 28])
    ax2.legend(loc="upper right")
    ax2.set_xlabel("mag")

    ax3.hist(I_all, bins=mag_bins, histtype="step", lw=lw, label="I", color="purple")
    ax3.hist(I, bins=mag_bins, histtype="stepfilled", alpha=0.5, lw=lw, label="I-", color="purple")
    ax3.set_xlim([21, 28])
    ax3.legend(loc="upper right")
    ax3.set_xlabel("mag")
    plt.savefig("dNdm-f%d-DEEP2-BRI-unmatched-mag21to28.png" %fnum, dpi=200, bbox_inches="tight")    
    # plt.show()
    plt.close()


