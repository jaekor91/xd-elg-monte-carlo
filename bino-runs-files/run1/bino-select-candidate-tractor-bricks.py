import numpy as np
import matplotlib.pylab as plt
import astropy.io.fits as fits 
from xd_elg_utils import *


def hr2deg(hr):
    return hr * 15

tycho_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/"


print "Load ccdtosky output file."
# Load in hpix file
data = np.load("/Users/jaehyeon/Documents/Research/DESI-angular-clustering/ccdtosky/outputs/DR5/decals_Nside11/output_arr_chunk0thru50331648.npy")
ra, dec = data["hpix_ra"], data["hpix_dec"]
Ng, Nr, Nz = data["g_Nexp_sum"], data["r_Nexp_sum"], data["z_Nexp_sum"]
N_pix = ra.size# 
iNexp_cut = (Ng >=2) & (Nr >=2) & (Nz >=2)

# Apply further tycho constraint    
print "Generate tycho mask."
iTycho = apply_tycho_radec(ra, dec, tycho_directory+"tycho2.fits", galtype="ELG") == 0
print "\n"

print "Load survey brick file."
# Load survey bricks file
data_bricks = fits.open("survey-bricks-dr5.fits")[1].data
ra_b, dec_b = data_bricks["ra"], data_bricks["dec"]
bname = data_bricks["brickname"]

print "Spcify regions to target."
# Field cuts (corresponding to Run 1)
# 1hr. Stripe 82
ifield_cut1 = np.logical_and((ra_b < hr2deg(1.5)), (ra_b > hr2deg(0.5))) & (dec_b < 1.5) & (dec_b > 0)
# 3 hr. Stripe 82
ifield_cut2 = np.logical_and((ra_b < hr2deg(3.5)), (ra_b > hr2deg(2.5))) & (dec_b < 1.5) & (dec_b > 0)
# 8h+30.
ifield_cut3 = np.logical_and((ra_b < hr2deg(9)), (ra_b > hr2deg(7))) & (dec_b > 25)





# For each field, randomly select a pixel until the following criteria area met.
# - hpix pixels that lie within 0.25 x 0.25 brick has a high pass rate.
print "Find candidates for visual inspection."
coverage_threshold = 0.90
tycho_threashold = 0.95
num_candidates = 4

bname_selected = []
coverage_fracs = []
for i, ibool in enumerate([ifield_cut1, ifield_cut2, ifield_cut3]):
    print i, ibool.sum()
    bname_tmp = bname[ibool]
    ra_b_tmp = ra_b[ibool]
    dec_b_tmp = dec_b[ibool]
    
    list_tmp1 = []
    list_tmp2 = []            
    for j in range(num_candidates):
        print j

        while True:
            while True: # Pick a brick that has been choosen yet
                idx = np.random.randint(bname_tmp.size)
                if bname_tmp[idx] not in list_tmp1:
                    break
            ra_selected = ra_b_tmp[idx]
            dec_selected = dec_b_tmp[idx]

            # Collect all hpix that lie within the brick centered at the tractor brick center
            tol = 0.25/2.
            ibool2 = (ra<(ra_selected+tol)) & (ra>(ra_selected-tol)) &  (dec<(dec_selected+tol)) & (dec>(dec_selected-tol)) 

            # Tells what fraction of pixels have Nexp_grz >=2
            Nexp_cut_tally = iNexp_cut[ibool2]
            Npix_inside_brick = Nexp_cut_tally.size
            coverage_fraction = Nexp_cut_tally.sum()/float(Npix_inside_brick)

            # Tells what fraction of pixels pass Tycho 2 cut
            tycho_fraction = iTycho[ibool2].sum()/float(Npix_inside_brick)
            
            print Npix_inside_brick, coverage_fraction, tycho_fraction

            if (coverage_fraction > coverage_threshold) and (tycho_fraction > tycho_threashold):
                list_tmp1.append(bname_tmp[idx])
                list_tmp2.append(coverage_fraction)
                break
        print "\n"
        
    bname_selected.append(list_tmp1)
    coverage_fracs.append(list_tmp2)
        
    print "\n"



for j in range(3):
    print bname_selected[j]
    print coverage_fracs[j]


print "List of candidates"
flat_list = [item for sublist in bname_selected for item in sublist]
print flat_list

print "List of candidates per region"
print bname_selected
np.save("bname_selected.npy", bname_selected)