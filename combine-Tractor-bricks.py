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

dr_v = "3" # Data release version
data_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/DR"+dr_v+"/"
tycho_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/"

##############################################################################	
print("Combine all Tractor files by field, append Tycho-2 stellar mask column and mask objects using DEEP2 window funtions.")
print("1. Combining the tractor files: Impose mask conditions (brick_primary==True and flux inverse variance positive).")
# Field 2
tracf2 = combine_tractor_nocut(data_directory+"f2/")
# Field 3
tracf3 = combine_tractor_nocut(data_directory+"f3/")
# Field 4
tracf4 = combine_tractor_nocut(data_directory+"f4/")
print("Completed.")


print("2. Append Tycho2 stark mask field.")
# Field 2 
tracf2 = apply_tycho(tracf2, tycho_directory+"tycho2.fits", galtype="ELG")
# Field 3
tracf3 = apply_tycho(tracf3, tycho_directory+"tycho2.fits", galtype="ELG")
# Field 4 
tracf4 = apply_tycho(tracf4, tycho_directory+"tycho2.fits", galtype="ELG")
print("Completed.")




##############################################################################	
print("3. Save the trimmed catalogs.")
# Field 2
cols = fits.ColDefs(tracf2)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto("DR"+dr_v+"-Tractor-D2f2.fits", clobber=True)

# Field 3
cols = fits.ColDefs(tracf3)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto("DR"+dr_v+"-Tractor-D2f3.fits", clobber=True)

# Field 4
cols = fits.ColDefs(tracf4)
tbhdu = fits.BinTableHDU.from_columns(cols)
tbhdu.writeto("DR"+dr_v+"-Tractor-D2f4.fits", clobber=True)
print("Completed.")

