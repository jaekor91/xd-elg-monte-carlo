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



def load_DEEP2(fname):
    tbl = load_fits_table(fname)
    ra, dec = tbl["RA_DEEP"], tbl["DEC_DEEP"]
    tycho = tbl["TYCHOVETO"]
    B, R, I = tbl["BESTB"], tbl["BESTR"], tbl["BESTI"]
    cn = tbl["cn"]
    return ra, dec, tycho, B, R, I, cn


# Import area
areas = np.load("spec-area.npy")

lw=3
lw2=2
mag_bins0= np.arange(20, 24.1, 0.05)
mag_bins_small = np.arange(20, 24.1, 0.2)
Rmag_nominal = 24.1
# Bmag_nominal = 23.4
# Imag_nominal = 22.4
Rmag_max = 25.

# All objects
Rmag = []
Bmag = []
Imag = []
for i, fnum in enumerate([2, 3, 4]):
    # DEEP2 data
    d2_dir = "../data-repository/DEEP2/photo-redz-oii/"
    ra, dec, tycho, B, R, I, cn = load_DEEP2(d2_dir+"deep2-f%d-photo-redz-oii.fits"%fnum)
    ibool = (tycho==0) & (R<24.1)
    Rmag.append(R[ibool])
    Bmag.append(B[ibool])
    Imag.append(I[ibool])
    
# Plot all objects 
figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
for i in range(3):
    numobjs = Bmag[i].size
    if numobjs < 1000:
        mag_bins = mag_bins_small
    else:
        mag_bins = mag_bins0
    ax_list[i].hist(Bmag[i], bins=mag_bins, histtype="step", color="green", label="DEEP2 B", alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])    
    ax_list[i].hist(Rmag[i], bins=mag_bins, histtype="step", color="red", label="DEEP2 R", alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])
    ax_list[i].hist(Imag[i], bins=mag_bins, histtype="step", color="purple", label="DEEP2 I", alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])
    # Nominal depth
#     ax_list[i].axvline(x=Rmag_nominal, c="red", lw=lw, ls="--")
    ax_list[i].set_xlim([20., 24.1])
    ax_list[i].legend(loc="upper left")
    ax_list[i].set_xlabel("mag", fontsize=20)
    ax_list[i].set_title("Field %d"%(i+2), fontsize=20)
plt.suptitle("DEEP2 pcats all objects by field", fontsize=30, y=1.05)
plt.savefig("dNdm-DEEP2-per-sqdeg-by-field-Rlim24p1.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()


# Plot all objects 
figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
BRI_tags = ["B", "R", "I"]
mags = [Bmag, Rmag, Imag]
field_colors = ["black", "red", "blue"]
for i in range(3):
    for j in range(3):
        numobjs = mags[i][j].size
        if numobjs < 1000:
            mag_bins = mag_bins_small
        else:
            mag_bins = mag_bins0
        ax_list[i].hist(mags[i][j], bins=mag_bins, histtype="step", color=field_colors[j], label=("Field %d"%(j+2)), alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])    
    # Nominal depth
#     ax_list[i].axvline(x=Rmag_nominal, c="green", lw=lw, ls="--")
    ax_list[i].set_xlim([20., 24.1])
    ax_list[i].legend(loc="upper left")
    ax_list[i].set_xlabel("mag", fontsize=20)
    ax_list[i].set_title(BRI_tags[i], fontsize=20)
plt.suptitle("DEEP2 pcats all objects by BRI", fontsize=30, y=1.05)
plt.savefig("dNdm-DEEP2-per-sqdeg-by-BRI-Rlim24p1.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()




# Plot objects by respective class
BRI_tags = ["B", "R", "I"]
mags = [Bmag, Rmag, Imag]
field_colors = ["black", "red", "blue"]
cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]

for l in [0, 1, 2, 3, 4, 5, 6, 8]:
    # All objects
    Rmag = []
    Bmag = []
    Imag = []
    for i, fnum in enumerate([2, 3, 4]):
        # DEEP2 data
        d2_dir = "../data-repository/DEEP2/photo-redz-oii/"
        ra, dec, tycho, B, R, I, cn = load_DEEP2(d2_dir+"deep2-f%d-photo-redz-oii.fits"%fnum)
        ibool = (tycho==0) & (cn==l) & (R<24.1)
        Rmag.append(R[ibool])
        Bmag.append(B[ibool])
        Imag.append(I[ibool])

    mags = [Bmag, Rmag, Imag]
    figure, ax_list = plt.subplots(1, 3, figsize=(25,6))
    for i in range(3):
        for j in range(3):
            numobjs = mags[i][j].size
            if numobjs < 2000:
                mag_bins = mag_bins_small
            else:
                mag_bins = mag_bins0
            ax_list[i].hist(mags[i][j], bins=mag_bins, histtype="step", color=field_colors[j], label=("Field %d"%(j+2)), alpha=1, lw=lw2, weights=np.ones(numobjs)/areas[i])    
        # Nominal depth
#         ax_list[i].axvline(x=Rmag_nominal, c="green", lw=lw, ls="--")
        ax_list[i].set_xlim([20., 24.1])
        ax_list[i].legend(loc="upper left")
        ax_list[i].set_xlabel("mag", fontsize=20)
        ax_list[i].set_title(BRI_tags[i], fontsize=20)
    plt.suptitle("DEEP2 pcats by BRI: %s" % cnames[l], fontsize=30, y=1.05)
    plt.savefig("dNdm-DEEP2-per-sqdeg-by-BRI-%d%s-Rlim24p1.png"%(l,cnames[l]), dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()
