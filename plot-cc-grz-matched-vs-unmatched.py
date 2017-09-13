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


def load_tractor_DR5_matched_to_DEEP2(fname, ibool=None):
    """
    Load select columns
    """
    tbl = load_fits_table(fname)
    if ibool is not None:
        tbl = tbl[ibool]
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
    red_z, z_err, z_quality = tbl["RED_Z"], tbl["Z_ERR"], tbl["ZQUALITY"]
    oii, oii_err = tbl["OII_3727"]*1e17, tbl["OII_3727_err"]*1e17
    D2matched = tbl["DEEP2_matched"]
    BRI_cut = tbl["BRI_cut"].astype(int).astype(bool)
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn, w, red_z, z_err, z_quality, oii, oii_err, D2matched




areas = np.load("spec-area.npy")

for fnum in [2, 3, 4]:
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
    rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn, w, red_z, z_err, z_quality, oii, oii_err, D2matched\
        = load_tractor_DR5_matched_to_DEEP2("DR5-matched-to-DEEP2-f%d-glim25.fits" % fnum)

    # Magnitudes
    gmag, rmag, zmag = flux2mag(gflux), flux2mag(rflux), flux2mag(zflux)
    ygr = gmag - rmag
    xrz = rmag - zmag
    gf_err = np.sqrt(1./givar)
    rf_err = np.sqrt(1./rivar)
    zf_err = np.sqrt(1./zivar)

    # Signal to noise
    g_sn = gflux_raw/gf_err
    r_sn = rflux_raw/rf_err
    z_sn = zflux_raw/zf_err

    ibool = (D2matched==1) & (gflux>0)  & (rflux>0)  & (zflux>0) & (gmag<24)# matched objects
    ibool2 = (D2matched==0) & (gflux>0)  & (rflux>0)  & (zflux>0) & (gmag<24) # unmatched objects

    pt_size = 10
    fig = plt.figure(figsize=(7, 7))
    plt.scatter(xrz[ibool], ygr[ibool], c="black", s=pt_size, edgecolors="none", label="Matched")
    plt.scatter(xrz[ibool2], ygr[ibool2], c="red", s=pt_size, edgecolors="none", label="Unmatched")
    plt.axis("equal")
    plt.xlim([-.5, 2.5])
    plt.ylim([-.5, 2.5])
    plt.xlabel("$r-z$", fontsize=20)
    plt.ylabel("$g-r$", fontsize=20)
    plt.legend(loc="upper left")
    plt.title("DR5 Matched to DEEP2 (g<24)", fontsize=20)
    plt.savefig("cc-grz-DR5-to-DEEP2-f%d-matched-vs-unmatched-glim24.png"%fnum, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()



    ibool = (D2matched==1) & (gflux>0)  & (rflux>0)  & (zflux>0) & (gmag<25)# matched objects
    ibool2 = (D2matched==0) & (gflux>0)  & (rflux>0)  & (zflux>0) & (gmag<25) # unmatched objects

    pt_size = 10
    fig = plt.figure(figsize=(7, 7))
    plt.scatter(xrz[ibool], ygr[ibool], c="black", s=pt_size, edgecolors="none", label="Matched")
    plt.scatter(xrz[ibool2], ygr[ibool2], c="red", s=pt_size, edgecolors="none", label="Unmatched")
    plt.axis("equal")
    plt.xlim([-.5, 2.5])
    plt.ylim([-.5, 2.5])
    plt.xlabel("$r-z$", fontsize=20)
    plt.ylabel("$g-r$", fontsize=20)
    plt.legend(loc="upper left")
    plt.title("DR5 Matched to DEEP2 (g<25)", fontsize=20)
    plt.savefig("cc-grz-DR5-to-DEEP2-f%d-matched-vs-unmatched-glim25.png"%fnum, dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()