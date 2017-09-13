# Loading modules
import numpy as np
from os import listdir
from os.path import isfile, join
from astropy.io import ascii, fits
from astropy.wcs import WCS
import numpy.lib.recfunctions as rec
from xd_elg_utils import *
import sys

large_random_constant = -999119283571
deg2arcsec=3600

data_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/DR5/tractor-incompleteness/f4/"
tycho_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/"

##############################################################################	
print("1. Combining the tractor files.")
trac = combine_tractor_nocut(data_directory)
print("Completed.")


print("2. Append Tycho2 stark mask field.")
trac = apply_tycho(trac, tycho_directory+"tycho2.fits", galtype="ELG")
print("Completed.")


##############################################################################	
print("4. Save the trimmed catalogs.")
cols = fits.ColDefs(trac)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto("DR5-Tractor-D2f4-all.fits", clobber=True)