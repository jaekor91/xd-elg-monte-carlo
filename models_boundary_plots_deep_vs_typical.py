import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
from model_class import *
import sys
import matplotlib.pyplot as plt
import time


def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)

category = ["NonELG", "NoZ", "ELG"]

# 0: Full F34 data
# 1: F3 data only
# 2: F4 data only
# 3-7: CV1-CV5: Sub-sample F34 into five-fold CV sets.
# 8-10: Magnitude changes. For power law use full data. Not used. 
# g in [22.5, 23.5], [22.75, 23.75], [23, 24]. 

sub_sample_name = ["Full", "F3", "F4", "CV1", "CV2", "CV3", "CV4", "CV5", "Mag1", "Mag2", "Mag3"] # No need to touch this
NK_list = [1]#, 3, 4, 5, 6, 7]
Niter = 1

MC_AREA = 1000 # In sq. deg.

# Correspond to 1,000 sq. deg.
# NonELG sample number: 16,896,579
# NoZ sample number: 4,963,537
# ELG sample number: 7,925,135
# 15 seconds


j = 0
print "/----- %s -----/" % sub_sample_name[j]

print("# ----- Model3 ----- #")
instance_model = model3(j)        


print "Fit MoGs"
instance_model.fit_MoG(NK_list, "model3", sub_sample_name[j], cache=True, Niter=Niter)
print "\n"
print "Fit Pow"
instance_model.fit_dNdf("model3", "Full", cache=True, Niter=Niter)
print "\n"


print "/---- Plotting boundary ----/"
instance_model.set_area_MC(MC_AREA)            
instance_model.gen_sample_intrinsic()

print "Typcial depths"
# Convolve error to the intrinsic sample.
start = time.time()
instance_model.set_err_lims(23.8, 23.4, 22.4, 8) 
instance_model.gen_err_conv_sample()
print "Time for convolving error sample: %.2f seconds" % (time.time() - start)

# Create the selection.
start = time.time()            
eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
print "Time for generating selection volume: %.2f seconds" % (time.time() - start)


print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred


# for i in [2, 1, 0]:
#     instance_model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag="Full-typical")

print "Remember the selected cells for the first option"
cell_centers_typical = instance_model.cell_select_centers()

print "\n\n"





print "Deep depths"
# Convolve error to the intrinsic sample.
start = time.time()
instance_model.set_err_lims(23.8+1, 23.4+1, 22.4+1, 8) 
instance_model.gen_err_conv_sample()
print "Time for convolving error sample: %.2f seconds" % (time.time() - start)

# Create the selection.
start = time.time()            
eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
print "Time for generating selection volume: %.2f seconds" % (time.time() - start)


print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred

# for i in [2, 1, 0]:
#     instance_model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag="Full-deep")


print "\n\n"


print "Plotting them together"
for i in [2, 1, 0]:
    instance_model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag="Full-deep-vs-typical",\
        var_x_ext = cell_centers_typical[:, 0], var_y_ext = cell_centers_typical[:, 1], gmag_ext = cell_centers_typical[:, 2],\
        use_parameterized_ext=True, plot_ext =True, alpha_ext=0.3, guide=True)
