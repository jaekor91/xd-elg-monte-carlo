import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys
import corner 

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
    mw_g, mw_r, mw_z = tbl["mw_transmission_g"], tbl["mw_transmission_r"], tbl["mw_transmission_z"]    
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
        rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn, w, red_z, z_err, z_quality, oii, oii_err, D2matched





areas = np.load("spec-area.npy") # Area of each field.
for fnum in [2, 3, 4]:
    print "/----- Field %d -----/" % fnum
    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
    rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn, w, red_z, z_err, z_quality, oii, oii_err, D2matched\
        = load_tractor_DR5_matched_to_DEEP2("DR5-matched-to-DEEP2-f%d-glim25.fits" % fnum)
        
    ibool = D2matched==1
    print "Fraction of unmatched objects: %.2f percent" % (100 * (ibool.size-ibool.sum())/float(ibool.size))
    print "We consider only the matched set. After fitting various densities, we scale the normalization by the amount we ignored in our fit due to unmatched set."


    bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
    rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn, w, red_z, z_err, z_quality, oii, oii_err, D2matched\
        = load_tractor_DR5_matched_to_DEEP2("DR5-matched-to-DEEP2-f%d-glim25.fits" % fnum, ibool)
    
    # OII signal to noise
    oii_sn = oii/oii_err

    # Proper weights for NonELG and color selected but unobserved classes. 
    w[cn==6] = 1
    w[cn==8] = 0

    # Magnitudes
    gmag, rmag, zmag = flux2mag(gflux), flux2mag(rflux), flux2mag(zflux)
    ygr = gmag - rmag
    xrz = rmag - zmag

    # error
    gf_err = np.sqrt(1./givar)/mw_g
    rf_err = np.sqrt(1./rivar)/mw_r
    zf_err = np.sqrt(1./zivar)/mw_z

    # Signal to noise
    g_sn = gflux/gf_err
    r_sn = rflux/rf_err
    z_sn = zflux/zf_err




    print "# ----- ELG group ----- #"
    # Define mutually exclusive sub-classes of objects.
    # Only DEEP2 color selected objects are concerned here.
    # - secure_z with proper error & secure oii with proper error
    iELG0 = (z_quality>=3) & (z_err>0) & (oii>0) & (oii_err>0) & BRI_cut
    # - secure_z with non-proper error & secure oii with proper error
    iELG1 = (z_quality>=3) & (z_err<=0) & (oii>0) & (oii_err>0) & BRI_cut
    # - secure_z with proper error & secure (negative) oii with proper error
    iELG2 = (z_quality>=3) & (z_err>0) & (oii<0) & (oii_err>0) & BRI_cut
    # - secure_z with proper error & secure (negative) oii with non-proper error
    iELG3 = (z_quality>=3) & (z_err>0) & (oii<0) & (oii_err<0) & BRI_cut

    # - secure_z with non-proper error & secure (negative) oii with non-proper error
    # iELG = (z_quality>=3) & (z_err<=0) & (oii<0) & (oii_err<0)
    # - secure_z with proper error & secure oii with non-proper error: None.
    # iELG = (z_quality>=3) & (z_err<=0) & (oii>0) & (oii_err<=0)
    # - secure_z with non-proper error & secure (negative) oii with proper error.
    # iELG = (z_quality>=3) & (z_err<=0) & (oii < 0) & (oii_err>0)
    # - secure_z with proper error & secure (negative) oii with non-proper error
    # iELG = (z_quality>=3) & (z_err<=0) & (oii<0) & (oii_err<0)
    # for e in [iELG0, iELG1, iELG2, iELG3]:
    #     print "%d" % w[e].sum()

    print "Among objects with secure redshift, only be concerned with those that have proper error, positive oii, and proper oii error. Ignore other objects as outliers."

    iELG = iELG0
    nELG_raw = iELG.sum()
    nELG_weighted = w[iELG].sum()
    print "ELG considered total # (raw/weighted): %d/%d" % (nELG_raw, nELG_weighted)



    print "# ----- Non-ELG group ----- # "
    # D2rejected group
    iNonELG0 = (cn==6)
    # Stars
    iNonELG1 = (z_quality == -1) & BRI_cut
    # iNonELG2 
    print "D2reject #: %d" % w[iNonELG0].sum()
    print "Stars #: %d" % w[iNonELG1].sum()
    iNonELG = np.logical_or(iNonELG0, iNonELG1)
    nNonELG_raw = iNonELG.sum()
    nNonELG_weighted =  w[iNonELG].sum()
    print "Non-ELG considered total # (raw/weighted): %d/%d" % (nNonELG_raw, nNonELG_weighted)

    print "# ----- NoZ ----- #"
    print "Objects with no secure redshift determination, but not a star (i.e., ZQAULITY==-1)"
    iNoZ = np.logical_or.reduce(((z_quality==-2) , (z_quality==0) , (z_quality==1) ,(z_quality==2)))  & (oii_err <=0) & BRI_cut
    nNoZ_raw = iNoZ.sum()
    nNoZ_weighted = w[iNoZ].sum()
    print "NoZ considered total # (raw/weighted): %d/%d" % (nNoZ_raw, nNoZ_weighted)

    print "# ----- Accounting ----- #"
    n_raw_total = nELG_raw + nNonELG_raw + nNoZ_raw + ((BRI_cut==1)& (z_quality<-10)).sum() 
    print "Raw total (including DEEP2 color selected but unobserved objects) / proportion of all objs considered: %d/%.2f" % (n_raw_total, n_raw_total/float(red_z.size)*100 )
    print "We ignore the ~2 percent of objects."
    print "\n"



    print "# ------ Exploratory analysis Marginal flux distributions ----- #"
    category = ["ELG", "NoZ", "NonELG"]
    fmin = -1.0 #np.min([rflux.min(), zflux.min()])
    fmax = mag2flux(22.)
    for mag_lim in [24, 25]:
        ifcut = gflux > mag2flux(mag_lim)
        print "# ----- Mag lim %.1f ----- #" % mag_lim
        print "# ----- ELG set exploration ----- #"
        ibool = iELG & ifcut
        ibool2 = ibool & (oii_sn == 1)

        # redz, z_err
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # red_z 
        dz = 0.025
        z_bins = np.arange(0.45, 1.75+dz/2., dz)
        lw = 1.
        ax1.hist(red_z[ibool2], bins=z_bins, histtype="step", lw=lw, color="red", weights=w[ibool2]/areas[fnum-2], label="OII SN = 1")
        ax1.hist(red_z[ibool], bins=z_bins, histtype="step", lw=lw, color="black", weights=w[ibool]/areas[fnum-2], label="All")
        ax1.set_xlabel("z", fontsize=20)
        ax1.legend(loc="upper right")
        # z_err
        z_err_bins = np.arange(-1e-5, 5e-4, 5e-6)
        ax2.hist(z_err[ibool2], bins = z_err_bins, histtype="step", lw=lw, color="red", weights=w[ibool2]/areas[fnum-2], label="OII SN = 1")
        ax2.hist(z_err[ibool], bins = z_err_bins, histtype="step", lw=lw, color="black", weights=w[ibool]/areas[fnum-2], label="All")
        ax2.set_xlabel("z_err", fontsize=20)
        ax2.legend(loc="upper right")
        plt.savefig("ELG-redz-zerr-D2f%d-glim%d.png"% (fnum, mag_lim), bbox_inches = "tight", dpi = 200)
        plt.close()


        # oii oii_err
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        # oii
        doii = .5
        oii_bins = np.arange(0, 50, doii)
        lw = 1.
        ax1.hist(oii[ibool2], bins=oii_bins, histtype="step", lw=lw, color="red", weights=w[ibool2], label="OII SN = 1")
        ax1.hist(oii[ibool], bins=oii_bins, histtype="step", lw=lw, color="black", weights=w[ibool], label="All")
        ax1.set_xlabel("oii", fontsize=20)
        ax1.legend(loc="upper right")
        # oii sn
        oii_sn_bins = np.arange(0, 40, 1) #  
        ax2.hist(oii_sn[ibool2], bins = oii_sn_bins,histtype="step", lw=lw, color="red", weights=w[ibool2], label="OII SN = 1")
        ax2.hist(oii_sn[ibool], bins = oii_sn_bins,histtype="step", lw=lw, color="black", weights=w[ibool], label="All")
        ax2.axvline(x=3, lw=1, c="red", ls="--")
        ax2.set_xlabel("oii_sn", fontsize=20)
        ax2.legend(loc="upper right")
        plt.savefig("ELG-oii-oiisn-D2f%d-glim%d.png"% (fnum, mag_lim), bbox_inches = "tight", dpi = 200)        
        plt.close()

        print "It appears that for certain low OII flux galaxies, their OII error was set to be equal to the flux level. For the purpose of modeling, let's ignore the uncertainty in reshift but retain uncertainty in OII."
        print "Fraction of ELGs with OII SN less than 3: %.2f" % (100-100*np.sum(oii_sn<3)/float(oii.size))
        print "\n"




        for i, ibool in enumerate([iELG, iNoZ, iNonELG]):
            print "Category %s" % category[i]
            ibool = ibool & ifcut

            # Flux distributions
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 7))
            # Unnormalized dNdf
            lw = 1.5

            df = (fmax-fmin)/50.
            fbins = np.arange(fmin, fmax, df)
            ax1.hist(gflux[ibool], bins=fbins, alpha=1, histtype="step", lw = lw, color = "green", label = "gflux", weights=w[ibool])
            ax1.hist(rflux[ibool], bins=fbins, alpha=1, histtype="step", lw = lw, color = "red", label = "rflux", weights=w[ibool])
            ax1.hist(zflux[ibool], bins=fbins, alpha=1, histtype="step", lw = lw, color = "purple", label = "zflux", weights=w[ibool])
            ax1.set_xlabel("flux", fontsize=20)
            ax1.set_xlim([-1., fmax])
            ax1.legend(loc="upper right", fontsize=20)
            ax1.set_title("dNdf [%.2f, %.2f]" % (fmin, fmax), fontsize=20)

            # Flux errrs 
            err_bins = np.arange(0.0, 0.3, 0.005)
            ax2.hist(gf_err[ibool], bins=err_bins, alpha=1, histtype="step", lw = lw, color = "green", label = "gf_err", normed=True, weights=w[ibool])
            ax2.hist(rf_err[ibool], bins=err_bins, alpha=1, histtype="step", lw = lw, color = "red", label = "rf_err", normed=True, weights=w[ibool])
            ax2.hist(zf_err[ibool], bins=err_bins, alpha=1, histtype="step", lw = lw, color = "purple", label = "zf_err", normed=True, weights=w[ibool])
            ax2.set_xlabel("flux", fontsize=20)
            ax2.legend(loc="upper right", fontsize=20)

            # Signal to noise
            sn_bins = np.arange(-4, 50, 0.5)
            ax3.hist(g_sn[ibool], bins=sn_bins, alpha=1, histtype="step", lw = lw, color = "green", label = "gf_SN", normed=True, weights=w[ibool])
            ax3.hist(r_sn[ibool], bins=sn_bins, alpha=1, histtype="step", lw = lw, color = "red", label = "rf_SN", normed=True, weights=w[ibool])
            ax3.hist(z_sn[ibool], bins=sn_bins, alpha=1, histtype="step", lw = lw, color = "purple", label = "zf_SN", normed=True, weights=w[ibool])
            ax3.set_xlabel("SN", fontsize=20)
            ax3.legend(loc="upper right", fontsize=20)
            plt.savefig("%s-flux-ferr-fsn-D2f%d-glim%d.png"% (category[i], fnum, mag_lim), bbox_inches = "tight", dpi = 200)        
            plt.close()

            nobs = g_sn[ibool].size
            for sig_cut in [3, 4, 5]:
                itau = gflux[ibool] < mag2flux(22)
                nobs_cut = itau.sum()
                print "Fraction of objects with S/N less than %d:" % sig_cut
                print "g: %.5f/%.5f" % ((g_sn[ibool]<sig_cut).sum()/float(nobs) * 100, (g_sn[ibool][itau]<sig_cut).sum()/float(nobs_cut) * 100)
                print "r: %.5f/%.5f" % ((r_sn[ibool]<sig_cut).sum()/float(nobs) * 100, (r_sn[ibool][itau]<sig_cut).sum()/float(nobs_cut) * 100)
                print "z: %.5f/%.5f" % ((z_sn[ibool]<sig_cut).sum()/float(nobs) * 100, (z_sn[ibool][itau]<sig_cut).sum()/float(nobs_cut) * 100)

            print "\n"

        print "Error, SN, and Flux distributions seem reasonable. z-band data has the highest photometric noise."