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

sub_sample_name = ["Full", "F3", "F4", "CV1", "CV2", "CV3", "CV4", "CV5", "Mag1", "Mag2", "Mag3", "F2"] # No need to touch this
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
MC_AREA = 2000  # In sq. deg.

# Correspond to 1,000 sq. deg.
# NonELG sample number: 16,896,579
# NoZ sample number: 4,963,537
# ELG sample number: 7,925,135
# 15 seconds

nums_list = [] # Keeps trac of validation information. 1st and 2nd levels -- cv and field.
# for j in range(12):
for j in [0]: # , 1, 2, 11]:
    print "/----- %s -----/" % sub_sample_name[j]

    print("# ----- Model3 ----- #")
    instance_model = model3(j)
    instance_model.frac_regular = 0.025 # If using 2000 sq. deg.
    if plot_data:
        print "Plot original data points"
        instance_model.plot_data("model3", sub_sample_name[j], guide=True) # Plot all together
        print "\n"

    if make_fits:
        print "Fit MoGs"
        instance_model.fit_MoG(NK_list, "model3", sub_sample_name[j], cache=True, Niter=Niter)
        print "\n"
        print "Fit Pow"
        instance_model.fit_dNdm_broken_pow("model3", "Full", cache=True, Niter=Niter)                    
        # if (j < 8) or (j == 11):
        #     instance_model.fit_dNdf("model3", sub_sample_name[j], cache=True, Niter=Niter)
        # else:
        #     instance_model.fit_dNdf("model3", "Full", cache=True, Niter=Niter)            
        print "\n"

#     # if visualize_fit:
#     #     print "MC sample draw from the intrinsic distribution"
#     #     instance_model.set_err_lims(25., 24.5, 23.5, 8) # Training data is deep!        
#     #     for K in NK_list:
#     #         K_selected = [K] * 3
#     #         instance_model.gen_sample_intrinsic(K_selected)
#     #         instance_model.gen_err_conv_sample()
#     #         print "\n"

#     #         print "Visualize the fits"
#     #         for i in range(3):
#     #             print "Plotting %s with K %d" % (category[i], K_selected[i])
#     #             instance_model.visualize_fit("model3", sub_sample_name[j], cat=i, K=K_selected[i], cum_contour=True, MC=True)  
#     #         print "\n"

    if validate:
        print "Generate Nsample from intrinsic density proportional to area: %.1f" % MC_AREA
        instance_model.set_area_MC(MC_AREA)
        start = time.time()
        instance_model.gen_sample_intrinsic_mag()
        print "Time for generating samples: %.1f seconds" % (time.time()-start)
        print "Validate on the DEEP2 sample by field"
        nums_list_2nd = []
        for fnum in [2, 3, 4]:
            print "Validating on field %d" % fnum
            # nums = instance_model.validate_on_DEEP2(fnum)
            returned  = instance_model.validate_on_DEEP2(fnum, model_tag="model3", cv_tag=sub_sample_name[j], plot_validation=False)
            # np.save("radec-XD-F%d-model3-kernel"%fnum, np.asarray(returned[-2:])) # radec
            nums = returned[:-2] # nums_list
            nums_list_2nd.append(np.asarray(nums))
            print "\n"
        nums_list.append(np.asarray(nums_list_2nd))

    # if plot_boundary:
    #     print "/---- Plotting boundary ----/"
    #     if j ==0:
    #         instance_model.set_area_MC(MC_AREA)            
    #         instance_model.gen_sample_intrinsic()

    #         print "Typcial depths"
    #         # Convolve error to the intrinsic sample.
    #         start = time.time()
    #         instance_model.set_err_lims(23.8, 23.4, 22.4, 8) 
    #         instance_model.gen_err_conv_sample()
    #         print "Time for convolving error sample: %.2f seconds" % (time.time() - start)

    #         # Create the selection.
    #         start = time.time()            
    #         eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    #             N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
    #         print "Time for generating selection volume: %.2f seconds" % (time.time() - start)


    #         print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
    #         print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    #             N_ELG_NonDESI_pred

    #         for i in [2, 1, 0]:
    #                 instance_model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag="Full-typical", guide=True)
    #         print "\n"

    # Part of validation scheme 
    # np.save("validation_set_model3_Full_PowerLaw", np.asarray(nums_list))
    # np.save("validation_set_model3_Sub_PowerLaw", np.asarray(nums_list))

