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
# 11: F2 data only

# Should not be altered.
sub_sample_name = ["Full", "F3", "F4", "CV1", "CV2", "CV3", "CV4", "CV5", "Mag1", "Mag2", "Mag3", "F2"] 
NK_list = [1]#, 3, 4, 5, 6, 7]
Niter = 1


# Series of operations to apply
plot_data = False
make_fits = True
# *************** Never turn it False! *********************** #
use_cached = True # Use cached models?
# *************** Never turn it False! *********************** #
visualize_fit = False
validate = True
plot_boundary = False 
MC_AREA = 1000  # In sq. deg.

# Correspond to 1,000 sq. deg.
# NonELG sample number: 16,896,579
# NoZ sample number: 4,963,537
# ELG sample number: 7,925,135
# 15 seconds

# Goal: Check the effect of model specification based on predicted distribution. 
# Two scenarios:
# - 1: Data is generated with different depths but the fixed selection is applied. 
#       Compare to adapted selection performance.
# - 2: Data is fixed to the fiducial but different selection is applied.

# Results: 
# - 1: Show how the efficiency varies as data distribution changes (fixed vs. adapted selection)
# - 2: " " " as the selection is varied (and data is fixed)








j=0 # Only use the full data model.

print("# ----- Model3 ----- #")
instance_model = model3(j)        

print "Load fitted model parameters"
print "Fit MoGs"
instance_model.fit_MoG(NK_list, "model3", sub_sample_name[j], cache=True, Niter=Niter)
print "Fit Pow"
instance_model.fit_dNdf("model3", "Full", cache=True, Niter=Niter)

print "\n\n"



print("# ----- Scenario 1 ----- #")
print("Data is generated with different depths but the fixed selection is applied. Compare to the adapated selection case.")
# Strategy:
# - Use a model3 instance to generate the fiducial depth data and selection at g=23.8, r=23.4, z=22.4.
# - Save the cell number of the fiducial selection.
# - Use the same model3 instance to:
#     - For each band, generate selection by changing the depth by pm dm
#     - Record the efficiency and other quantities IF the selection was adapated to the newly generated data. 
#     - Apply the fiducial selection, and record efficiency and other quantities.
#     - Plot boundary different between typical vs. adapted for chosen slices for dm = +- 0.25 cases.

bands = ["g", "r", "z"]
bands_fiducial = [23.8, 23.4, 22.4]

print("Generating the fiducial selection boundary.")
start = time.time()
instance_model.set_area_MC(MC_AREA)            
instance_model.gen_sample_intrinsic()
instance_model.set_err_lims(23.8, 23.4, 22.4, 8)
instance_model.gen_err_conv_sample()
eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
print "Time for generating selection volume: %.2f seconds" % (time.time() - start)



# # dm_list = np.arange(-0.5, 0.54, 0.05)
# dm_list = np.arange(-0.05, 0.054, 0.05)
# for i, b in enumerate(bands):
#     for j, dm in enumerate(dm_list):
#         print "Band %s: dm = %.2f" % (b, dm)

#     instance_model.set_area_MC(MC_AREA)            
#     instance_model.gen_sample_intrinsic()

#     print "Typcial depths"
#     # Convolve error to the intrinsic sample.
#     start = time.time()
#     instance_model.set_err_lims(23.8, 23.4, 22.4, 8) 
#     instance_model.gen_err_conv_sample()
#     print "Time for convolving error sample: %.2f seconds" % (time.time() - start)

#     # Create the selection.
#     start = time.time()            
#     eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#         N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
#     print "Time for generating selection volume: %.2f seconds" % (time.time() - start)


#     print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
#     print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#         N_ELG_NonDESI_pred

#     for i in [2, 1, 0]:
#             instance_model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag="Full-typical", guide=True)
#     print "\n"


# print "/---- Plotting boundary ----/"
# if j ==0:
#     instance_model.set_area_MC(MC_AREA)            
#     instance_model.gen_sample_intrinsic()

#     print "Typcial depths"
#     # Convolve error to the intrinsic sample.
#     start = time.time()
#     instance_model.set_err_lims(23.8, 23.4, 22.4, 8) 
#     instance_model.gen_err_conv_sample()
#     print "Time for convolving error sample: %.2f seconds" % (time.time() - start)

#     # Create the selection.
#     start = time.time()            
#     eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#         N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
#     print "Time for generating selection volume: %.2f seconds" % (time.time() - start)


#     print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
#     print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#         N_ELG_NonDESI_pred

#     for i in [2, 1, 0]:
#             instance_model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag="Full-typical", guide=True)
#     print "\n"

# Part of validation scheme 
# np.save("validation_set_model3_Full_PowerLaw", np.asarray(nums_list))
# 
