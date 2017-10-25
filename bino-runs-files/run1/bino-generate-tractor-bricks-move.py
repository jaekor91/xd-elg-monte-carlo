# Use this script to move tractor brick files from a designated directory to another.
# bricks-file is not used.

# Loading modules
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
from astropy.io import fits


large_random_constant = -999119283571
deg2arcsec=3600


from_directory = "/global/project/projectdirs/cosmo/data/legacysurvey/dr5/tractor/"
to_directory = "/global/homes/j/jaehyeon/"


##############################################################################
# run 1
# bricks = ["0118p010", "0393p002", "1202p275"]
center_ra = [11.8, 39.3, 120.2]
center_dec = [1.0, 0.2, 27.5]

# Using survye-bricks file to get all bricks near the center bricks specified above
data = fits.open("../../../data-repository/DR5/survey-bricks-dr5.fits")[1].data
ra_bricks = data["RA"]
dec_bricks = data["DEC"]
names_bricks = data["BRICKNAME"]

tol = 0.5
bricks = []
for i in range(3):
    ra_c = center_ra[i]
    dec_c = center_dec[i]
    ibool = (ra_bricks < ra_c+tol) & (ra_bricks > ra_c - tol) & (dec_bricks > dec_c -tol) & (dec_bricks < dec_c+tol)
    bricks.append(list(names_bricks[ibool]))
    
bricks = [item for sublist in bricks for item in sublist]
# print bricks

postfix = ".fits"
prefix = "cp "


f = open("tractor-move-binospec-test.sh","w")
for brick in bricks:
	if "p" in brick:
		tmp = brick.split("p")
	else:
		tmp = brick.split("m")		
	tractor_directory = tmp[0][-4:-1]
	brick_address = tractor_directory+"/"+"tractor-"+brick
	mv_command = "cp "+from_directory + brick_address + ".fits " + to_directory + "\n"
	f.write(mv_command)
f.close()
print("Completed")
