import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)


def load_tractor_DR5(fname, ibool=None):
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
    gflux, rflux, zflux = gflux_raw/tbl["mw_transmission_g"], rflux_raw/tbl["mw_transmission_r"], zflux_raw/tbl["mw_transmission_z"]
    givar_raw, rivar_raw, zivar_raw = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    mw_g, mw_r, mw_z = tbl["mw_transmission_g"], tbl["mw_transmission_r"], tbl["mw_transmission_z"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    objtype = tbl["type"]
    rex_expr, rex_expr_ivar = tbl["rex_shapeExp_r"], tbl["rex_shapeExp_r_ivar"]    
    tycho = tbl["TYCHOVETO"]
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar_raw,\
    rivar_raw, zivar_raw, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, rex_expr, rex_expr_ivar



areas_all = np.load("spec-area-all.npy")

for fnum in [2, 3, 4]: 

    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
    rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, rex_expr, rex_expr_ivar\
    = load_tractor_DR5("DR5-Tractor-D2f%d-all.fits"%fnum, ibool=None)

    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (tycho==0)\
    & (gflux > mag2flux(24.0)) & (gflux < mag2flux(21.0)) 

    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
    rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, rex_expr, rex_expr_ivar\
    = load_tractor_DR5("DR5-Tractor-D2f%d-all.fits"%fnum, ibool=ibool)


    r_exp_err = 1/np.sqrt(rex_expr_ivar)
    r_exp_sn = rex_expr/r_exp_err

    labels = ["PSF", "Others"]

    ipsf = (objtype == "PSF")
    # iREX = (objtype == "REX")
    iOthers = ~ipsf
    # iOthers = np.logical_or((objtype == "COMP"), (objtype == "EXP"), (objtype == "DEV"))
    colors = ["black", "red"]


    for i, ibool in enumerate([ipsf, iOthers]):
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(25, 7))    

        nobjs = ibool.sum()
        
        # Histogram of all r_exp
        if i == 1:
            rbins = np.arange(0, 1, 1e-2)
        else:
            rbins = np.arange(0, 0.2, 1e-3)            
        ax0.hist(rex_expr[ibool], bins=rbins, histtype="step", color=colors[i], lw=2, label=labels[i], weights=np.ones(nobjs)/float(areas_all[fnum-2]))

        # Histogram of error
        if i == 1:
            rbins = np.arange(0, 0.1, 1e-3)
        else:
            rbins = np.arange(0, 0.4, 5e-3)            
        ax1.hist(r_exp_err[ibool], bins=rbins, histtype="step", color=colors[i], lw=2, label=labels[i], weights=np.ones(nobjs)/float(areas_all[fnum-2]))

        # Histogram of singal to noise
        if i == 1:
            rbins = np.arange(0, 50, 0.5)
        else:
            rbins = np.arange(0, 5, 5e-2)            
        ax2.hist(r_exp_sn[ibool], bins=rbins, histtype="step", color=colors[i], lw=2, label=labels[i], weights=np.ones(nobjs)/float(areas_all[fnum-2]))
        ax2.axvline(x=3, ls="--", lw=1.5, c="blue")
        ax0.legend(loc="upper right", fontsize=25)
        ax1.legend(loc="upper right", fontsize=25)
        ax2.legend(loc="upper right", fontsize=25)
        ax0.set_xlabel(r"$r_{exp}$", fontsize=25)
        ax1.set_xlabel(r"$r_{exp}$ error", fontsize=25)
        ax2.set_xlabel(r"$r_{exp}$ S2N", fontsize=25)
        

        fraction  = ((r_exp_sn[ibool]<3).sum()*100/float(ibool.sum()))
        plt.suptitle("Field %d; Area %.2f; nobjs %d; Frac SN < 3: %.1f percent" % (fnum, areas_all[fnum-2], ibool.sum(), fraction), fontsize=30)
        plt.savefig("rex-expr-d2f%d-%s.png"% (fnum, labels[i]), dpi=200, bbox_inches="tight")
        plt.close()


for i in range(2):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(25, 7))    
    colors = ["black", "red", "blue"]
    for fnum in [2, 3, 4]: 
        bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, rex_expr, rex_expr_ivar\
        = load_tractor_DR5("DR5-Tractor-D2f%d-all.fits"%fnum, ibool=None)

        ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0) & (givar>0) & (rivar>0) & (zivar>0) & (tycho==0)\
        & (gflux > mag2flux(24.0)) & (gflux < mag2flux(21.0)) 

        bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, rex_expr, rex_expr_ivar\
        = load_tractor_DR5("DR5-Tractor-D2f%d-all.fits"%fnum, ibool=ibool)


        r_exp_err = 1/np.sqrt(rex_expr_ivar)
        r_exp_sn = rex_expr/r_exp_err

        labels = ["PSF", "Others"]

        ipsf = (objtype == "PSF")
        # iREX = (objtype == "REX")
        iOthers = ~ipsf
        # iOthers = np.logical_or((objtype == "COMP"), (objtype == "EXP"), (objtype == "DEV"))


        ibool = [ipsf, iOthers][i]

        nobjs = ibool.sum()

        # Histogram of all r_exp
        rbins = np.arange(0, 1, 1e-2)
        ax0.hist(rex_expr[ibool], bins=rbins, histtype="step", color=colors[fnum-2], lw=2, label=("%d: "%fnum)+labels[i], weights=np.ones(nobjs)/float(areas_all[fnum-2]))

        # Histogram of error
        rbins = np.arange(0, 0.1, 1e-3)
        ax1.hist(r_exp_err[ibool], bins=rbins, histtype="step", color=colors[fnum-2], lw=2, label=("%d: "%fnum)+labels[i], weights=np.ones(nobjs)/float(areas_all[fnum-2]))

        # Histogram of singal to noise
        rbins = np.arange(0, 50, 0.5)
        ax2.hist(r_exp_sn[ibool], bins=rbins, histtype="step", color=colors[fnum-2], lw=2, label=("%d: "%fnum)+labels[i], weights=np.ones(nobjs)/float(areas_all[fnum-2]))
        
    ax2.axvline(x=3, ls="--", lw=1.5, c="blue")
    ax0.legend(loc="upper right", fontsize=25)
    ax1.legend(loc="upper right", fontsize=25)
    ax2.legend(loc="upper right", fontsize=25)
    ax0.set_xlabel(r"$r_{exp}$", fontsize=25)
    ax1.set_xlabel(r"$r_{exp}$ error", fontsize=25)
    ax2.set_xlabel(r"$r_{exp}$ S2N", fontsize=25)

    plt.savefig("rex-expr-all-fields-%s.png"%labels[i], dpi=200, bbox_inches="tight")
    plt.close()
