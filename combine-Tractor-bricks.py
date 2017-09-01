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

dr_v = "5" # Data release version
data_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/DR"+dr_v+"/"
tycho_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/"
windowf_directory= "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/DEEP2/windowf/"

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


print("2. Impose DEEP2 window functions.")
# Field 2
idx = np.logical_or(window_mask(tracf2["ra"], tracf2["dec"], windowf_directory+"windowf.21.fits"), window_mask(tracf2["ra"], tracf2["dec"], windowf_directory+"windowf.22.fits"))
tracf2_trimmed = tracf2[idx]

# Field 3
idx = np.logical_or.reduce((window_mask(tracf3["ra"], tracf3["dec"], windowf_directory+"windowf.31.fits"), window_mask(tracf3["ra"], tracf3["dec"], windowf_directory+"windowf.32.fits"),window_mask(tracf3["ra"], tracf3["dec"], windowf_directory+"windowf.33.fits")))
tracf3_trimmed = tracf3[idx]

# Field 4
idx = np.logical_or(window_mask(tracf4["ra"], tracf4["dec"], windowf_directory+"windowf.41.fits"), window_mask(tracf4["ra"], tracf4["dec"], windowf_directory+"windowf.42.fits"))
tracf4_trimmed = np.copy(tracf4[idx])
print("Completed.")


print("3. Append Tycho2 stark mask field.")
# Field 2 
tracf2 = apply_tycho(tracf2_trimmed, tycho_directory+"tycho2.fits", galtype="ELG")
# Field 3
tracf3 = apply_tycho(tracf3_trimmed, tycho_directory+"tycho2.fits", galtype="ELG")
# Field 4 
tracf4 = apply_tycho(tracf4_trimmed, tycho_directory+"tycho2.fits", galtype="ELG")
print("Completed.")




##############################################################################	
print("4. Save the trimmed catalogs.")
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

