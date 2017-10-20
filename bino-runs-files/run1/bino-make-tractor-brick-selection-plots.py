import numpy as np
import matplotlib.pylab as plt
import astropy.io.fits as fits 
from xd_elg_utils import *
from selection_script import *

def hr2deg(hr):
    return hr * 15

bname_selected = np.load("bname_selected.npy")

tycho_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/"
field_names = ["St82-1h", "St82-3h", "8h+30"]
num_cols = int(np.sqrt(num_candidates))
num_candidates = 2

for j in range(3):
    fig, ax_list = plt.subplots(num_cols, num_cols, figsize=(25,25))
    for i, bn in enumerate(bname_selected[j]):
        fname = "../data-repository/Tractor_binospec_test/run1-Nov2017/tractor-%s.fits" % bn

        # load the data and apply tycho
        data = load_fits_table(fname)
        data = apply_tycho(data, tycho_directory+"tycho2.fits", galtype="ELG")    
        ra, dec = data["ra"], data["dec"]

        tycho = data["TYCHOVETO"]
        gflux = data["flux_g"]/data["mw_transmission_g"]
        igmag = flux2mag(gflux) < 24
        N_gmag = igmag.sum()
        

        # apply selection
        iselected = apply_selection(fname) & (tycho==0)

        # Number selected
        N_selected = iselected.sum()

#         assert ra.size == n_total

        # plotting the ra/dec of the selection    
        ax_list[i//num_cols, i%num_cols].scatter(ra[(tycho==0)], dec[(tycho==0)], c = "black", s=5, edgecolors="none")
        ax_list[i//num_cols, i%num_cols].scatter(ra[iselected], dec[iselected], c = "red", s=30, edgecolors="none")            
        ax_list[i//num_cols, i%num_cols].set_xlabel("RA", fontsize=20)
        ax_list[i//num_cols, i%num_cols].set_ylabel("DEC", fontsize=20) 
        ax_list[i//num_cols, i%num_cols].axis("equal")    
        ax_list[i//num_cols, i%num_cols].set_title("%d: %d/%d (%s/%.2f)" % (i, N_selected, N_gmag, bn, coverage_fracs[j][i]), fontsize =20)      

    plt.suptitle("%s Candidate : N_NDM / N_g<24 (Tractor fname/Nexp_grz>=2 coverage frac)" % field_names[j], fontsize=25)
    plt.savefig("%d-%s-candidates-test.png" % (j, field_names[j]), bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()