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
if True:
    print("# ----- Scenario 1 ----- #")
    print("Data is generated with different depths but the fixed selection is applied. Compare to the adapated selection case.")
    # Strategy:
    # - Use a model3 instance to generate the fiducial depth data and selection at g=23.8, r=23.4, z=22.4, OII=8.
    # - Save the cell number of the fiducial selection.
    # - Use the same model3 instance to:
    #     - For each band, generate selection by changing the depth by pm dm
    #     - Record the efficiency and other quantities IF the selection was adapated to the newly generated data. 
    #     - Apply the fiducial selection, and record efficiency and other quantities.
    #     - Plot boundary different between typical vs. adapted for chosen slices for dm = +- 0.25 cases.

    # Note:
    # - There is no need to generate intrinsic sample each time, only new error convolution is necessary.


    instance_model = model3(j)        

    print "Load fitted model parameters"
    print "Fit MoGs"
    instance_model.fit_MoG(NK_list, "model3", sub_sample_name[j], cache=True, Niter=Niter)
    print "Fit Pow"
    instance_model.fit_dNdf("model3", "Full", cache=True, Niter=Niter)
    print "\n\n"

    print "Adjust the box sizes so we don't run into limits."
    instance_model.var_x_limits = [-.15, 2.75]
    instance_model.var_y_limits = [-0.5, 1.2]
    instance_model.gmag_limits = [21.0, 24.]
    instance_model.num_bins = [290, 170, 300]



    bands = ["g", "r", "z", "OII"]
    bands_fiducial = [23.8, 23.4, 22.4, 8]

    print("Generating the fiducial selection boundary.")
    start = time.time()
    instance_model.set_area_MC(1000)            
    instance_model.gen_sample_intrinsic()
    instance_model.set_err_lims(23.8, 23.4, 22.4, 8)
    instance_model.gen_err_conv_sample()
    eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI,\
        N_ELG_NonDESI = instance_model.gen_selection_volume_scipy()
    print "Time for generating the fiducial selection volume: %.2f seconds" % (time.time() - start)

    print "Eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI"
    print "%.3f, %d, %d, %d, %d, %d, %d" % (eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI)

    print "Save the fiducial selection."
    cell_select_fiducial = instance_model.cell_select

    # Place holder for the result
    error_misspec1_list = []

    # dm_list = np.arange(-0.25 , 0., 0.5) # Debug
    for i, b in enumerate(bands):
        if i < 3: 
            dm_list = np.arange(-0.5, 0.51, 0.05)
        else:
            dm_list = np.arange(-2, 0.51, 0.25)

        for j, dm in enumerate(dm_list):
            print "Band %s: dm = %.2f" % (b, dm)
            bands_fiducial_tmp = np.copy(bands_fiducial)
            bands_fiducial_tmp[i] = bands_fiducial_tmp[i] + dm
            gtmp, rtmp, ztmp, OIItmp = bands_fiducial_tmp
            print "New depths g/r/z/OII: %.2f / %.2f / %.2f / %.2f" % (gtmp, rtmp, ztmp, OIItmp)

            print "Generate the selection"
            start = time.time()
            instance_model.set_err_lims(gtmp, rtmp, ztmp, OIItmp) 
            instance_model.gen_err_conv_sample()

            eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI,\
            eff_ext, Ntotal_ext, Ngood_ext, N_NonELG_ext, N_NoZ_ext, N_ELG_DESI_ext, N_ELG_NonDESI_ext,\
            = instance_model.gen_selection_volume_scipy(selection_ext = cell_select_fiducial)
            print "Time for generating the new selection volume: %.2f seconds" % (time.time() - start)

            print "Adapted (fixed)"        
            print "Eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI"
            print "%.3f (%.3f), %d (%d), %d (%d), %d (%d), %d (%d), %d (%d), %d (%d)" \
            % (eff, eff_ext, Ntotal, Ntotal_ext, Ngood, Ngood_ext, N_NonELG, N_NonELG_ext, N_NoZ, N_NoZ_ext,\
             N_ELG_DESI,  N_ELG_DESI_ext, N_ELG_NonDESI, N_ELG_NonDESI_ext)

            # for i in [2, 1, 0]:
                # instance_model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag="Full-typical", guide=True)

            # Save the result into an array
            # 0-3: Error model
            # 4-9: Adpated result
            # 10-15: Fixed result
            error_misspec1_list.append([gtmp, rtmp, ztmp, OIItmp, \
                eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI,\
                eff_ext, Ntotal_ext, Ngood_ext, N_NonELG_ext, N_NoZ_ext, N_ELG_DESI_ext, N_ELG_NonDESI_ext])

            print "\n"

    np.save("model3_error_misspec1", np.asarray(error_misspec1_list))






if True:
    print("# ----- Scenario 2 ----- #")
    print("2: Data is fixed to the fiducial but different selection is applied.")
    # Strategy:
    # - Use a different model3 instance to: 
    #     - For each band, generate selection by changing the depth by pm dm.
    #     - Use a model3 instance to generate the fiducial depth data and selection at g=23.8, r=23.4, z=22.4, OII=8.    
    #     - Apply the new selection to the typical depth data and record efficiency and other quantities.
    #     - Plot boundary different between typical vs. adapted for chosen slices for dm = +- 0.25 cases.

    instance_model = model3(j)        
    instance_model2 = model3(j)        

    print "Load fitted model parameters"
    print "Fit MoGs"
    instance_model.fit_MoG(NK_list, "model3", sub_sample_name[j], cache=True, Niter=Niter)
    instance_model2.fit_MoG(NK_list, "model3", sub_sample_name[j], cache=True, Niter=Niter)
    print "Fit Pow"
    instance_model.fit_dNdf("model3", "Full", cache=True, Niter=Niter)
    instance_model2.fit_dNdf("model3", "Full", cache=True, Niter=Niter)
    print "\n\n"

    print "Adjust the box sizes so we don't run into limits."
    instance_model.var_x_limits = [-.15, 2.75]
    instance_model.var_y_limits = [-0.5, 1.2]
    instance_model.gmag_limits = [21.0, 24.]
    instance_model.num_bins = [290, 170, 300]


    print "Adjust the box sizes so we don't run into limits."
    instance_model2.var_x_limits = [-.15, 2.75]
    instance_model2.var_y_limits = [-0.5, 1.2]
    instance_model2.gmag_limits = [21.0, 24.]
    instance_model2.num_bins = [290, 170, 300]


    print "Generate the intrinsic samples for both."
    instance_model.set_area_MC(1000)                        
    instance_model.gen_sample_intrinsic()
    instance_model2.set_area_MC(1000)                        
    instance_model2.gen_sample_intrinsic()

    print "Perform error convolutino for the first instance which will remain fixed."
    instance_model.set_err_lims(23.8, 23.4, 22.4, 8)
    instance_model.gen_err_conv_sample()

    bands = ["g", "r", "z", "OII"]
    bands_fiducial = [23.8, 23.4, 22.4, 8]

    # Place holder for the result
    error_misspec2_list = []

    # dm_list = np.arange(-0.25 , 0., 0.5) # Debug
    for i, b in enumerate(bands):
        if i < 3: 
            dm_list = np.arange(-0.5, 0.51, 0.05)
        else:
            dm_list = np.arange(-2, 0.51, 0.25)

        for j, dm in enumerate(dm_list):
            print "Band %s: dm = %.2f" % (b, dm)
            bands_fiducial_tmp = np.copy(bands_fiducial)
            bands_fiducial_tmp[i] = bands_fiducial_tmp[i] + dm
            gtmp, rtmp, ztmp, OIItmp = bands_fiducial_tmp
            print "New depths g/r/z/OII: %.2f / %.2f / %.2f / %.2f" % (gtmp, rtmp, ztmp, OIItmp)

            print "Generate the selection"
            start = time.time()
            instance_model2.set_err_lims(gtmp, rtmp, ztmp, OIItmp) 
            instance_model2.gen_err_conv_sample()
            eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI,\
                N_ELG_NonDESI = instance_model2.gen_selection_volume_scipy()
            print "Time for generating the new selection volume: %.2f seconds" % (time.time() - start)

            # Save
            cell_select_new = instance_model2.cell_select

            print "Apply to the fiducial generated data."
            start = time.time()            
            eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI,\
            eff_ext, Ntotal_ext, Ngood_ext, N_NonELG_ext, N_NoZ_ext, N_ELG_DESI_ext, N_ELG_NonDESI_ext,\
            = instance_model.gen_selection_volume_scipy(selection_ext = cell_select_new)
            print "Time for generating the new selection volume: %.2f seconds" % (time.time() - start)

            print "Baseline (Mis-specified)"        
            print "Eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI"
            print "%.3f (%.3f), %d (%d), %d (%d), %d (%d), %d (%d), %d (%d), %d (%d)" \
            % (eff, eff_ext, Ntotal, Ntotal_ext, Ngood, Ngood_ext, N_NonELG, N_NonELG_ext, N_NoZ, N_NoZ_ext,\
             N_ELG_DESI,  N_ELG_DESI_ext, N_ELG_NonDESI, N_ELG_NonDESI_ext)

            # for i in [2, 1, 0]:
                # instance_model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag="Full-typical", guide=True)

            # Save the result into an array
            # 0-3: Error model
            # 4-10: Baseline
            # 11-17: Mis-specification of error model
            error_misspec2_list.append([gtmp, rtmp, ztmp, OIItmp, \
                eff, Ntotal, Ngood, N_NonELG, N_NoZ, N_ELG_DESI, N_ELG_NonDESI,\
                eff_ext, Ntotal_ext, Ngood_ext, N_NonELG_ext, N_NoZ_ext, N_ELG_DESI_ext, N_ELG_NonDESI_ext])

            print "\n"

    np.save("model3_error_misspec2", np.asarray(error_misspec2_list))

