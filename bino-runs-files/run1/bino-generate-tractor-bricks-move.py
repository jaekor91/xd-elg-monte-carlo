# Use this script to move tractor brick files from a designated directory to another.
# bricks-file is not used.

# Loading modules
import numpy as np
from os import listdir
from os.path import isfile, join
import sys

large_random_constant = -999119283571
deg2arcsec=3600


from_directory = "/global/project/projectdirs/cosmo/data/legacysurvey/dr5/tractor/"
to_directory = "/global/homes/j/jaehyeon/"


##############################################################################
# run 1
# bricks = ['0178p010', '0118p010', '0093p002', '0101p007', '0143p010', '0116p007', '0093p007', '0076p002', '0123p010', '0438p007', '0386p002', '0431p012', '0428p002', '0393p002', '0426p002', '0381p002', '0423p005', '0431p007', '1342p262', '1327p257', '1202p275', '1182p285', '1269p270', '1313p285', '1157p270', '1151p275', '1149p265']

# run test
bname_selected = np.load("bname_selected.npy")
bricks = [item for sublist in bname_selected for item in sublist]

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
