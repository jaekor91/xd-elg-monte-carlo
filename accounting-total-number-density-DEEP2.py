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
    w = tbl["TARG_WEIGHT"]
    return ra, dec, tycho, B, R, I, cn, w


areas = np.load("spec-area.npy")

print("Before cut")
for i, fnum in enumerate([2, 3, 4]):
    d2_dir    = "../data-repository/DEEP2/photo-redz-oii/"
    ra, dec, tycho, B, R, I, cn, w = load_DEEP2(d2_dir+"deep2-f%d-photo-redz-oii.fits" % fnum)
    ibool = np.ones(ra.size, dtype=bool)
    weighted_total = np.sum(w[(cn<6) & ibool]) + np.sum((cn==6)&ibool) # Color-selected. Observed.
    raw_total = np.sum(ibool)
    print "Field %d" % fnum
    print "Raw/Weigthed densities: %d, %d" % (raw_total/areas[i], weighted_total/areas[i])
    print "Relative difference from raw: %.2f" % ((weighted_total-raw_total)/raw_total * 100)
print "\n"


print("After R<24.1 cut")
for i, fnum in enumerate([2, 3, 4]):
    d2_dir    = "../data-repository/DEEP2/photo-redz-oii/"
    ra, dec, tycho, B, R, I, cn, w = load_DEEP2(d2_dir+"deep2-f%d-photo-redz-oii.fits" % fnum)
    ibool =  (R<24.1)
    weighted_total = np.sum(w[(cn<6) & ibool]) + np.sum((cn==6)&ibool) # Color-selected. Observed.
    raw_total = np.sum(ibool)
    print "Field %d" % fnum
    print "Raw/Weigthed densities: %d, %d" % (raw_total/areas[i], weighted_total/areas[i])
    print "Relative difference from raw: %.2f" % ((weighted_total-raw_total)/raw_total * 100)
print "\n"

print("After tycho cut")
for i, fnum in enumerate([2, 3, 4]):
    d2_dir    = "../data-repository/DEEP2/photo-redz-oii/"
    ra, dec, tycho, B, R, I, cn, w = load_DEEP2(d2_dir+"deep2-f%d-photo-redz-oii.fits" % fnum)
    ibool =  (tycho==0)
    weighted_total = np.sum(w[(cn<6) & ibool]) + np.sum((cn==6)&ibool) # Color-selected. Observed.
    raw_total = np.sum(ibool)
    print "Field %d" % fnum
    print "Raw/Weigthed densities: %d, %d" % (raw_total/areas[i], weighted_total/areas[i])
    print "Relative difference from raw: %.2f" % ((weighted_total-raw_total)/raw_total * 100)
print "\n"


print("After R<24.1 AND tycho mask cuts")
for i, fnum in enumerate([2, 3, 4]):
    d2_dir    = "../data-repository/DEEP2/photo-redz-oii/"
    ra, dec, tycho, B, R, I, cn, w = load_DEEP2(d2_dir+"deep2-f%d-photo-redz-oii.fits" % fnum)
    ibool = (tycho==0) & (R<24.1)
    weighted_total = np.sum(w[(cn<6) & ibool]) + np.sum((cn==6)&ibool) # Color-selected. Observed.
    raw_total = np.sum(ibool)
    print "Field %d" % fnum
    print "Raw/Weigthed densities: %d, %d" % (raw_total/areas[i], weighted_total/areas[i])
    print "Relative difference from raw: %.2f" % ((weighted_total-raw_total)/raw_total * 100)
print "\n"





    