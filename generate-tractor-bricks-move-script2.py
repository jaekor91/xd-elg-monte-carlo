# Use this script to move tractor brick files from a designated directory to another.
# bricks-file is not used.

# Loading modules
from os import listdir
from os.path import isfile, join
import sys

large_random_constant = -999119283571
deg2arcsec=3600

# from_directory = "/global/cscratch1/sd/desiproc/DR5_out/tractor/"
from_directory = "/global/projecta/projectdirs/cosmo/work/dr5/DR5_out/"
to_directory = "/global/homes/j/jaehyeon/tmp-tractor-storage/f"
# data_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/DR5/"


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


def search_bricknames(fits_directory, ra_search_range, dec_search_range, tol=0.25, dir_names = None):
	bnames = []
	ra_min, ra_max = ra_search_range
	dec_min, dec_max = dec_search_range

	if dir_names is None:
		onlyfiles = [f for f in listdir(fits_directory) if isfile(join(fits_directory, f))]
		for i,e in enumerate(onlyfiles, start=0):
			# If the file ends with "fits"
			if e[-4:] == "fits":
				if e.startswith("tractor-"):
				#                 print(e)
					if "p" in e:
						tmp = e.split("p")
					else:
						tmp = e.split("m")
					ra = int(tmp[0][-4:])/10.
					dec = int(tmp[1][:3])/10.
					if (ra<ra_max+tol) & (ra>ra_min-tol) &  (dec<dec_max+tol) & (dec>dec_min-tol):
						bnames.append(e)
	else:
		for j, nm in enumerate(dir_names):
			if type(nm) is int:
				search_directory = fits_directory+("%d/"%nm)
			else:
				search_directory = fits_directory+nm+"/"
			print(search_directory)
			onlyfiles = [f for f in listdir(search_directory) if isfile(join(search_directory, f))]
			for i,e in enumerate(onlyfiles, start=0):
				# If the file ends with "fits"
				if e[-4:] == "fits":
					if e.startswith("tractor-"):
						# print(e)
						if "p" in e:
							tmp = e.split("p")
						else:
							tmp = e.split("m")
						ra = int(tmp[0][-4:])/10.
						dec = int(tmp[1][:3])/10.
						if (ra<ra_max+tol) & (ra>ra_min-tol) &  (dec<dec_max+tol) & (dec>dec_min-tol):
							print(e)
							bnames.append(e)

	return bnames


# Getting the brick names near the ranges specified below.
tol = .5
f2_bricks, f3_bricks, f4_bricks = [], [], []
f2_bricks = search_bricknames(from_directory, [251.3, 253.7], [0, 0.1], tol, dir_names=range(245, 256))
f3_bricks = search_bricknames(from_directory, [351.25, 353.8], [-.2, .5], tol, dir_names=[351, 352, 353, 354])
# f4_bricks = search_bricknames(from_directory, [36.4,38.],[.3, 1.0], tol, dir_names=["030", "031", "032", "033", "034", "035", "036", "037", "038", "039", "040", "041", "042", "043", "044", "045"])
bricks = [f2_bricks, f3_bricks, f4_bricks]


postfix = ".fits"
prefix = "cp "
for i in range(3):
	field_num = i+2
	f = open(("tractor-move-d2f%d.sh"%field_num),"w")
	for brick in bricks[i]:
		if "p" in brick:
			tmp = brick.split("p")
		else:
			tmp = brick.split("m")		
		tractor_directory = tmp[0][-4:-1]
		brick_address = tractor_directory+"/"+brick
		mv_command = prefix + from_directory + brick_address + " " + to_directory+ ("%d\n"%field_num)
		f.write(mv_command)
	f.close()
print("Completed")
