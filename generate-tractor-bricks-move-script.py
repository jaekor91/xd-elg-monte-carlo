# Use this script to move tractor brick files from a designated directory to another.

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

from_directory = "/global/cscratch1/sd/desiproc/DR5_out/tractor/"
to_directory = "/global/homes/j/jaehyeon/tmp-tractor-storage/"
data_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/DR5/"
brick_fname = "survey-bricks.fits"


##############################################################################
print("\
\n\
Field 2\n\
RA bounds: [251.3, 253.7]\n\
DEC bounds: [34.6, 35.3]\n\
\n\
Field 3\n\
RA bounds: [351.25, 353.8]\n\
DEC bounds: [-.2, .5]\n\
\n\
Field 4\n\
RA bounds: [36.4, 38]\n\
DEC bounds: [.3, 1.0]\n\
")

fits_bricks = fits.open(data_directory+brick_fname)[1].data
ra = fits_bricks['ra'][:]
dec = fits_bricks['dec'][:]
br_name = fits_bricks['brickname'][:]

# Getting the brick names near the ranges specified below.
tol = 0.25
f2_bricks = return_bricknames(ra, dec, br_name,[251.3, 253.7],[34.6, 35.3],tol)
f3_bricks = return_bricknames(ra, dec, br_name,[351.25, 353.8],[-.2, .5],tol)
f4_bricks = return_bricknames(ra, dec, br_name,[36.4,38.],[.3, 1.0],tol)
bricks = [f2_bricks, f3_bricks, f4_bricks]

postfix = ".fits"
prefix = "cp "
for i in range(3):
    f = open("tractor-move-d2f%d.sh"%(i+2),"w")
    for brick in bricks[i]:
        tractor_directory = brick[:3]
        brick_address = tractor_directory+"/tractor-"+brick+postfix
        mv_command = prefix + from_directory + brick_address + " " + to_directory + "\n"
        f.write(mv_command)
    f.close()
print("Completed")

