import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
from model_class import *
import sys
import matplotlib.pyplot as plt
import time


category = ["NonELG", "NoZ", "ELG"]

sub_sample_name = ["Full"] 
NK_list = []
Niter = 0 


j = 0
print "Generate model 3 object."
model = model3(j)       



print "Fit MoGs"
model.fit_MoG(NK_list, "model3", sub_sample_name[j], cache=True, Niter=Niter)
print "\n"
print "Fit Pow"
model.fit_dNdm_broken_pow("model3", "Full", cache=True, Niter=Niter)
print "\n"





print "Setting the parameters that won't be changed"
# Flux range to draw the sample from. Slightly larger than the range we are interested.
model.fmin_MC = mag2flux(24.5) # Note that around 23.8, the power law starts to break down.
model.fmax_MC = mag2flux(19.5)
model.fcut = mag2flux(24.) # After noise addition, we make a cut at 24.

# Mag Power law from which to generate importance samples.
model.alpha_q = [9, 20, 20]
model.A_q = [1, 1, 1] # This information is not needed.

# For MoG
model.sigma_proposal = 1.5 # sigma factor for the proposal        

# Regularization number when computing utility
model.frac_regular = 0.05

# Fraction of NoZ objects that we expect to be good
model.f_NoZ = 0.25

# FoM values for individual NoZ and NonELG objects.
model.FoM_NonELG = 0.0

# Selection grid limits and number of bins 
# var_x, var_y, gmag. Width (0.01, 0.01, 0.01)
model.var_x_limits = [-.50, 3.75] # g-z
model.var_y_limits = [-0.85, 1.75] # g-r
model.gmag_limits = [19.5, 24.]
model.num_bins = [425, 260, 450]

# Number of pixels width to be used during Gaussian smoothing.
model.sigma_smoothing = [5., 5., 5.]
model.sigma_smoothing_limit = 4.

# Area
MC_AREA = 2000 # In sq. deg.
print "\n"


print "Generating intrinsic sample proportional to simulation area of %d" % MC_AREA
model.set_area_MC(MC_AREA)
start = time.time()
model.gen_sample_intrinsic_mag()
print "Time for generating sample: %.2f seconds" % (time.time() - start)
print "\n"


# Calibration data should be load after the binsize has been adjusted.
print "Time to load the calibration data"
start = time.time()
model.load_calibration_data()
dt = time.time() - start
print "Time taken: %.2f seconds" % dt











print "/----- Case 1:"
print "NDM Typical depths, N_tot = 2400, Flat FoM with external calibration"
# Depths of the field
model.set_err_lims(23.8, 23.4, 22.4, 8) # Use fiducial depths
model.num_desired = 2400
model.FoM_NoZ = 0.25
model.FoM_option = "flat"
# model.FoM_option = "Quadratic_redz"


# Convolve error to the intrinsic sample.
start = time.time()
model.gen_err_conv_sample()
print "Time for convolving error sample: %.2f seconds" % (time.time() - start)
print "\n"

# Create the selection.
start = time.time()            
eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred = model.gen_selection_volume_ext_cal(gaussian_smoothing=True)
print "Total time for generating selection volume: %.2f seconds" % (time.time() - start)
print "\n"

print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred


print "Save the selection"
tag = "ext-cal-case1-nominal"
np.save(tag+"-cell_select.npy", model.cell_select)
print "\n"
    
print "Plotting boundary"
for i in range(3):
    model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag=tag,\
        guide=True, output_sparse=True, increment=25)
print "\n"

print "Completed"
print "\n"












print "/----- Case 2:"
print "NDM deep depths, N_tot = 2400, Flat FoM with external calibration"
# Depths of the field
model.set_err_lims(23.8+0.5, 23.4+0.5, 22.4+0.5, 8) # Use fiducial depths
model.num_desired = 2400
model.FoM_NoZ = 0.25
model.FoM_option = "flat"
# model.FoM_option = "Quadratic_redz"


# Convolve error to the intrinsic sample.
start = time.time()
model.gen_err_conv_sample()
print "Time for convolving error sample: %.2f seconds" % (time.time() - start)
print "\n"

# Create the selection.
start = time.time()            
eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred = model.gen_selection_volume_ext_cal(gaussian_smoothing=True)
print "Total time for generating selection volume: %.2f seconds" % (time.time() - start)
print "\n"

print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred


print "Save the selection"
tag = "ext-cal-case2-deep"
np.save(tag+"-cell_select.npy", model.cell_select)
print "\n"
    
print "Plotting boundary"
for i in range(3):
    model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag=tag,\
        guide=True, output_sparse=True, increment=25)
print "\n"

print "Completed"
print "\n"










print "/----- Case 3:"
print "NDM nominal depths, N_tot = 2400, No OII FoM with external calibration"
# Depths of the field
model.set_err_lims(23.8, 23.4, 22.4, 8) # Use fiducial depths
model.num_desired = 2400
model.FoM_NoZ = 0.25
model.FoM_option = "NoOII"
# model.FoM_option = "Quadratic_redz"


# Convolve error to the intrinsic sample.
start = time.time()
model.gen_err_conv_sample()
print "Time for convolving error sample: %.2f seconds" % (time.time() - start)
print "\n"

# Create the selection.
start = time.time()            
eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred = model.gen_selection_volume_ext_cal(gaussian_smoothing=True)
print "Total time for generating selection volume: %.2f seconds" % (time.time() - start)
print "\n"

print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred


print "Save the selection"
tag = "ext-cal-case3-NoOII"
np.save(tag+"-cell_select.npy", model.cell_select)
print "\n"
    
print "Plotting boundary"
for i in range(3):
    model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag=tag,\
        guide=True, output_sparse=True, increment=25)
print "\n"

print "Completed"
print "\n"











print "/----- Case 4:"
print "NDM nominal depths, N_tot = 2400, quadratic z with external calibration."
# Depths of the field
model.set_err_lims(23.8, 23.4, 22.4, 8) # Use fiducial depths
model.num_desired = 2400
model.FoM_NoZ = 0.25
model.FoM_option = "Quadratic_redz"


# Convolve error to the intrinsic sample.
start = time.time()
model.gen_err_conv_sample()
print "Time for convolving error sample: %.2f seconds" % (time.time() - start)
print "\n"

# Create the selection.
start = time.time()            
eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred = model.gen_selection_volume_ext_cal(gaussian_smoothing=True)
print "Total time for generating selection volume: %.2f seconds" % (time.time() - start)
print "\n"

print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred


print "Save the selection"
tag = "ext-cal-case4-QuadraticZ"
np.save(tag+"-cell_select.npy", model.cell_select)
print "\n"
    
print "Plotting boundary"
for i in range(3):
    model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag=tag,\
        guide=True, output_sparse=True, increment=25)
print "\n"

print "Completed"
print "\n"










print "/----- Case 5:"
print "NDM nominal depths, N_tot = 2400, flat FoM, larger fNoZ with external calibration."
# Depths of the field
model.set_err_lims(23.8, 23.4, 22.4, 8) # Use fiducial depths
model.num_desired = 2400
model.FoM_NoZ = 0.5
model.FoM_option = "flat"


# Convolve error to the intrinsic sample.
start = time.time()
model.gen_err_conv_sample()
print "Time for convolving error sample: %.2f seconds" % (time.time() - start)
print "\n"

# Create the selection.
start = time.time()            
eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred = model.gen_selection_volume_ext_cal(gaussian_smoothing=True)
print "Total time for generating selection volume: %.2f seconds" % (time.time() - start)
print "\n"

print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
    N_ELG_NonDESI_pred


print "Save the selection"
tag = "ext-cal-case5-fNoZ50"
np.save(tag+"-cell_select.npy", model.cell_select)
print "\n"
    
print "Plotting boundary"
for i in range(3):
    model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag=tag,\
        guide=True, output_sparse=True, increment=25)
print "\n"

print "Completed"
print "\n"



# ------ print outs ------- # 
# Generate model 3 object.
# model_class.py:66: RuntimeWarning: divide by zero encountered in divide
#   w1_err, w2_err = np.sqrt(1./w1ivar)/tbl["mw_transmission_w1"], np.sqrt(1./w2ivar)/tbl["mw_transmission_w2"]
# Fraction of unmatched objects with g [17.0, 24.2]: 6.97 percent
# We multiply the correction to the weights before training.
# Fit MoGs
# Cached result will be used for MODELS-NonELG-model3-Full.
# Cached result will be used for MODELS-NoZ-model3-Full.
# Cached result will be used for MODELS-ELG-model3-Full.


# Fit Pow
# Cached result will be used for MODELS-NonELG-model3-Full-mag-broken-pow.
# Cached result will be used for MODELS-NoZ-model3-Full-mag-broken-pow.
# Cached result will be used for MODELS-ELG-model3-Full-mag-broken-pow.


# Setting the parameters that won't be changed


# Generating intrinsic sample proportional to simulation area of 2000
# NonELG sample number: 51737244
# NoZ sample number: 12783742
# ELG sample number: 20473307
# Time for generating sample: 78.68 seconds


# Time to load the calibration data
# Time taken: 5.67 seconds






# /----- Case 1:
# NDM Typical depths, N_tot = 2400, Flat FoM with external calibration
# Convolving error and re-parametrizing
# NonELG
# NoZ
# ELG
# Time for convolving error sample: 27.91 seconds


# Start of computing selection region.
# Constructing histograms.
# Time taken: 73.17 seconds
# Applying gaussian smoothing.
# Time taken: 9.50 seconds
# Computing magnitude dependent regularization.
# Time taken: 4.55 seconds
# Computing utility and sorting.
# Time taken: 15.23 seconds
# Flattening the MD histograms.
# Time taken: 11.00 seconds
# Sorting the flattened arrays.
# Time taken: 19.25 seconds
# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 603.6, 1077.2, 1699.5
# NonDESI ELGs: 116.5, 207.4, 379.7
# NoZ: 92.3, 186.9, 357.1
# NonELG: 156.2, 172.7, 257.3
# Poorly characterized objects (not included in density modeling, no prediction): 615.0, 129.1, NA
# ----------
# Total based on individual parts: NA, 1773.4, NA
# Total number: 1583.7, 1773.4, 2773.5
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.634, 0.645



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 755.7, 1368.8, 1699.5
# NonDESI ELGs: 153.0, 273.7, 379.7
# NoZ: 102.4, 203.6, 357.1
# NonELG: 76.5, 82.7, 257.3
# Poorly characterized objects (not included in density modeling, no prediction): 723.9, 97.6, NA
# ----------
# Total based on individual parts: NA, 2026.4, NA
# Total number: 1811.5, 2026.4, 2773.5
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.701, 0.645



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 1124.0, 2037.7, 1699.5
# NonDESI ELGs: 272.7, 490.8, 379.7
# NoZ: 132.9, 275.5, 357.1
# NonELG: 110.1, 117.8, 257.3
# Poorly characterized objects (not included in density modeling, no prediction): 790.1, 149.7, NA
# ----------
# Total based on individual parts: NA, 3071.4, NA
# Total number: 2429.8, 3071.4, 2773.5
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.686, 0.645



# Total time for generating selection volume: 136.21 seconds


# Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred
# 0.644954887252 2773.53604421 1788.80562668 257.258160197 357.075875877 1699.53665771 379.734115289
# Save the selection


# Plotting boundary
# mu_gz
# 0 $\mu_g - \mu_z$ [-0.500, -0.490]
# 25 $\mu_g - \mu_z$ [-0.250, -0.240]
# 50 $\mu_g - \mu_z$ [0.000, 0.010]
# 75 $\mu_g - \mu_z$ [0.250, 0.260]
# 100 $\mu_g - \mu_z$ [0.500, 0.510]
# 125 $\mu_g - \mu_z$ [0.750, 0.760]
# 150 $\mu_g - \mu_z$ [1.000, 1.010]
# 175 $\mu_g - \mu_z$ [1.250, 1.260]
# 200 $\mu_g - \mu_z$ [1.500, 1.510]
# 225 $\mu_g - \mu_z$ [1.750, 1.760]
# 250 $\mu_g - \mu_z$ [2.000, 2.010]
# 275 $\mu_g - \mu_z$ [2.250, 2.260]
# 300 $\mu_g - \mu_z$ [2.500, 2.510]
# 325 $\mu_g - \mu_z$ [2.750, 2.760]
# 350 $\mu_g - \mu_z$ [3.000, 3.010]
# 375 $\mu_g - \mu_z$ [3.250, 3.260]
# 400 $\mu_g - \mu_z$ [3.500, 3.510]
# mu_gr
# 0 $\mu_g - \mu_r$ [-0.850, -0.840]
# 25 $\mu_g - \mu_r$ [-0.600, -0.590]
# 50 $\mu_g - \mu_r$ [-0.350, -0.340]
# 75 $\mu_g - \mu_r$ [-0.100, -0.090]
# 100 $\mu_g - \mu_r$ [0.150, 0.160]
# 125 $\mu_g - \mu_r$ [0.400, 0.410]
# 150 $\mu_g - \mu_r$ [0.650, 0.660]
# 175 $\mu_g - \mu_r$ [0.900, 0.910]
# 200 $\mu_g - \mu_r$ [1.150, 1.160]
# 225 $\mu_g - \mu_r$ [1.400, 1.410]
# 250 $\mu_g - \mu_r$ [1.650, 1.660]
# gmag
# 0 $g$ [19.500, 19.510]
# 25 $g$ [19.750, 19.760]
# 50 $g$ [20.000, 20.010]
# 75 $g$ [20.250, 20.260]
# 100 $g$ [20.500, 20.510]
# 125 $g$ [20.750, 20.760]
# 150 $g$ [21.000, 21.010]
# 175 $g$ [21.250, 21.260]
# 200 $g$ [21.500, 21.510]
# 225 $g$ [21.750, 21.760]
# 250 $g$ [22.000, 22.010]
# 275 $g$ [22.250, 22.260]
# 300 $g$ [22.500, 22.510]
# 325 $g$ [22.750, 22.760]
# 350 $g$ [23.000, 23.010]
# 375 $g$ [23.250, 23.260]
# 400 $g$ [23.500, 23.510]
# 425 $g$ [23.750, 23.760]


# Completed













# /----- Case 2:
# NDM deep depths, N_tot = 2400, Flat FoM with external calibration
# Convolving error and re-parametrizing
# NonELG
# NoZ
# ELG
# Time for convolving error sample: 46.90 seconds


# Start of computing selection region.
# Constructing histograms.
# Time taken: 76.04 seconds
# Applying gaussian smoothing.
# Time taken: 9.10 seconds
# Computing magnitude dependent regularization.
# Time taken: 2.59 seconds
# Computing utility and sorting.
# Time taken: 13.08 seconds
# Flattening the MD histograms.
# Time taken: 7.49 seconds
# Sorting the flattened arrays.
# Time taken: 13.28 seconds
# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 607.9, 1084.6, 1843.4
# NonDESI ELGs: 96.6, 171.6, 333.0
# NoZ: 92.3, 181.0, 294.1
# NonELG: 149.1, 164.1, 86.1
# Poorly characterized objects (not included in density modeling, no prediction): 596.5, 121.6, NA
# ----------
# Total based on individual parts: NA, 1722.9, NA
# Total number: 1542.5, 1722.9, 2613.9
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.656, 0.733



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 780.4, 1410.7, 1843.4
# NonDESI ELGs: 136.5, 246.3, 333.0
# NoZ: 104.8, 206.8, 294.1
# NonELG: 82.4, 89.0, 86.1
# Poorly characterized objects (not included in density modeling, no prediction): 763.9, 115.5, NA
# ----------
# Total based on individual parts: NA, 2068.2, NA
# Total number: 1868.0, 2068.2, 2613.9
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.707, 0.733



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 1188.7, 2151.3, 1843.4
# NonDESI ELGs: 251.7, 453.8, 333.0
# NoZ: 143.3, 294.1, 294.1
# NonELG: 122.4, 130.9, 86.1
# Poorly characterized objects (not included in density modeling, no prediction): 840.8, 160.4, NA
# ----------
# Total based on individual parts: NA, 3190.5, NA
# Total number: 2546.9, 3190.5, 2613.9
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.697, 0.733



# Total time for generating selection volume: 124.38 seconds


# Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred
# 0.733360942255 2613.89441161 1916.92806865 86.0824699216 294.143875767 1843.39209971 332.973874747
# Save the selection


# Plotting boundary
# mu_gz
# 0 $\mu_g - \mu_z$ [-0.500, -0.490]
# 25 $\mu_g - \mu_z$ [-0.250, -0.240]
# 50 $\mu_g - \mu_z$ [0.000, 0.010]
# 75 $\mu_g - \mu_z$ [0.250, 0.260]
# 100 $\mu_g - \mu_z$ [0.500, 0.510]
# 125 $\mu_g - \mu_z$ [0.750, 0.760]
# 150 $\mu_g - \mu_z$ [1.000, 1.010]
# 175 $\mu_g - \mu_z$ [1.250, 1.260]
# 200 $\mu_g - \mu_z$ [1.500, 1.510]
# 225 $\mu_g - \mu_z$ [1.750, 1.760]
# 250 $\mu_g - \mu_z$ [2.000, 2.010]
# 275 $\mu_g - \mu_z$ [2.250, 2.260]
# 300 $\mu_g - \mu_z$ [2.500, 2.510]
# 325 $\mu_g - \mu_z$ [2.750, 2.760]
# 350 $\mu_g - \mu_z$ [3.000, 3.010]
# 375 $\mu_g - \mu_z$ [3.250, 3.260]
# 400 $\mu_g - \mu_z$ [3.500, 3.510]
# mu_gr
# 0 $\mu_g - \mu_r$ [-0.850, -0.840]
# 25 $\mu_g - \mu_r$ [-0.600, -0.590]
# 50 $\mu_g - \mu_r$ [-0.350, -0.340]
# 75 $\mu_g - \mu_r$ [-0.100, -0.090]
# 100 $\mu_g - \mu_r$ [0.150, 0.160]
# 125 $\mu_g - \mu_r$ [0.400, 0.410]
# 150 $\mu_g - \mu_r$ [0.650, 0.660]
# 175 $\mu_g - \mu_r$ [0.900, 0.910]
# 200 $\mu_g - \mu_r$ [1.150, 1.160]
# 225 $\mu_g - \mu_r$ [1.400, 1.410]
# 250 $\mu_g - \mu_r$ [1.650, 1.660]
# gmag
# 0 $g$ [19.500, 19.510]
# 25 $g$ [19.750, 19.760]
# 50 $g$ [20.000, 20.010]
# 75 $g$ [20.250, 20.260]
# 100 $g$ [20.500, 20.510]
# 125 $g$ [20.750, 20.760]
# 150 $g$ [21.000, 21.010]
# 175 $g$ [21.250, 21.260]
# 200 $g$ [21.500, 21.510]
# 225 $g$ [21.750, 21.760]
# 250 $g$ [22.000, 22.010]
# 275 $g$ [22.250, 22.260]
# 300 $g$ [22.500, 22.510]
# 325 $g$ [22.750, 22.760]
# 350 $g$ [23.000, 23.010]
# 375 $g$ [23.250, 23.260]
# 400 $g$ [23.500, 23.510]
# 425 $g$ [23.750, 23.760]


# Completed













# /----- Case 3:
# NDM nominal depths, N_tot = 2400, No OII FoM with external calibration
# Convolving error and re-parametrizing
# NonELG
# NoZ
# ELG
# Time for convolving error sample: 50.18 seconds


# Start of computing selection region.
# Constructing histograms.
# Time taken: 76.74 seconds
# Applying gaussian smoothing.
# Time taken: 9.28 seconds
# Computing magnitude dependent regularization.
# Time taken: 2.69 seconds
# Computing utility and sorting.
# Time taken: 13.88 seconds
# Flattening the MD histograms.
# Time taken: 10.56 seconds
# Sorting the flattened arrays.
# Time taken: 20.67 seconds
# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 578.1, 1026.6, 1530.1
# NonDESI ELGs: 237.2, 419.7, 806.4
# NoZ: 82.4, 155.6, 251.1
# NonELG: 129.3, 143.2, 183.5
# Poorly characterized objects (not included in density modeling, no prediction): 615.0, 94.3, NA
# ----------
# Total based on individual parts: NA, 1839.4, NA
# Total number: 1641.9, 1839.4, 2858.5
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.579, 0.557



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 688.6, 1225.6, 1530.1
# NonDESI ELGs: 339.0, 601.0, 806.4
# NoZ: 80.0, 144.4, 251.1
# NonELG: 51.8, 57.0, 183.5
# Poorly characterized objects (not included in density modeling, no prediction): 711.0, 44.0, NA
# ----------
# Total based on individual parts: NA, 2072.0, NA
# Total number: 1870.4, 2072.0, 2858.5
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.609, 0.557



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 875.8, 1560.9, 1530.1
# NonDESI ELGs: 499.9, 889.8, 806.4
# NoZ: 113.6, 210.7, 251.1
# NonELG: 61.2, 66.7, 183.5
# Poorly characterized objects (not included in density modeling, no prediction): 725.4, 92.6, NA
# ----------
# Total based on individual parts: NA, 2820.6, NA
# Total number: 2276.0, 2820.6, 2858.5
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.572, 0.557



# Total time for generating selection volume: 136.66 seconds


# Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred
# 0.557255348243 2858.45486349 1592.88926039 183.529920073 251.053983498 1530.12576452 806.358448201
# Save the selection


# Plotting boundary
# mu_gz
# 0 $\mu_g - \mu_z$ [-0.500, -0.490]
# 25 $\mu_g - \mu_z$ [-0.250, -0.240]
# 50 $\mu_g - \mu_z$ [0.000, 0.010]
# 75 $\mu_g - \mu_z$ [0.250, 0.260]
# 100 $\mu_g - \mu_z$ [0.500, 0.510]
# 125 $\mu_g - \mu_z$ [0.750, 0.760]
# 150 $\mu_g - \mu_z$ [1.000, 1.010]
# 175 $\mu_g - \mu_z$ [1.250, 1.260]
# 200 $\mu_g - \mu_z$ [1.500, 1.510]
# 225 $\mu_g - \mu_z$ [1.750, 1.760]
# 250 $\mu_g - \mu_z$ [2.000, 2.010]
# 275 $\mu_g - \mu_z$ [2.250, 2.260]
# 300 $\mu_g - \mu_z$ [2.500, 2.510]
# 325 $\mu_g - \mu_z$ [2.750, 2.760]
# 350 $\mu_g - \mu_z$ [3.000, 3.010]
# 375 $\mu_g - \mu_z$ [3.250, 3.260]
# 400 $\mu_g - \mu_z$ [3.500, 3.510]
# mu_gr
# 0 $\mu_g - \mu_r$ [-0.850, -0.840]
# 25 $\mu_g - \mu_r$ [-0.600, -0.590]
# 50 $\mu_g - \mu_r$ [-0.350, -0.340]
# 75 $\mu_g - \mu_r$ [-0.100, -0.090]
# 100 $\mu_g - \mu_r$ [0.150, 0.160]
# 125 $\mu_g - \mu_r$ [0.400, 0.410]
# 150 $\mu_g - \mu_r$ [0.650, 0.660]
# 175 $\mu_g - \mu_r$ [0.900, 0.910]
# 200 $\mu_g - \mu_r$ [1.150, 1.160]
# 225 $\mu_g - \mu_r$ [1.400, 1.410]
# 250 $\mu_g - \mu_r$ [1.650, 1.660]
# gmag
# 0 $g$ [19.500, 19.510]
# 25 $g$ [19.750, 19.760]
# 50 $g$ [20.000, 20.010]
# 75 $g$ [20.250, 20.260]
# 100 $g$ [20.500, 20.510]
# 125 $g$ [20.750, 20.760]
# 150 $g$ [21.000, 21.010]
# 175 $g$ [21.250, 21.260]
# 200 $g$ [21.500, 21.510]
# 225 $g$ [21.750, 21.760]
# 250 $g$ [22.000, 22.010]
# 275 $g$ [22.250, 22.260]
# 300 $g$ [22.500, 22.510]
# 325 $g$ [22.750, 22.760]
# 350 $g$ [23.000, 23.010]
# 375 $g$ [23.250, 23.260]
# 400 $g$ [23.500, 23.510]
# 425 $g$ [23.750, 23.760]


# Completed










# /----- Case 4:
# NDM nominal depths, N_tot = 2400, quadratic z with external calibration.
# Convolving error and re-parametrizing
# NonELG
# NoZ
# ELG
# Time for convolving error sample: 51.20 seconds


# Start of computing selection region.
# Constructing histograms.
# Time taken: 77.24 seconds
# Applying gaussian smoothing.
# Time taken: 10.60 seconds
# Computing magnitude dependent regularization.
# Time taken: 2.87 seconds
# Computing utility and sorting.
# Time taken: 12.83 seconds
# Flattening the MD histograms.
# Time taken: 9.90 seconds
# Sorting the flattened arrays.
# Time taken: 18.08 seconds
# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 573.8, 1029.7, 1698.8
# NonDESI ELGs: 106.5, 190.0, 392.2
# NoZ: 115.0, 234.9, 453.1
# NonELG: 154.8, 166.6, 269.6
# Poorly characterized objects (not included in density modeling, no prediction): 589.4, 102.8, NA
# ----------
# Total based on individual parts: NA, 1724.0, NA
# Total number: 1539.7, 1724.0, 2900.3
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.631, 0.625



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 729.8, 1331.0, 1698.8
# NonDESI ELGs: 124.8, 226.4, 392.2
# NoZ: 133.0, 277.7, 453.1
# NonELG: 74.2, 81.1, 269.6
# Poorly characterized objects (not included in density modeling, no prediction): 735.7, 93.9, NA
# ----------
# Total based on individual parts: NA, 2010.2, NA
# Total number: 1797.4, 2010.2, 2900.3
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.697, 0.625



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 1064.6, 1937.5, 1698.8
# NonDESI ELGs: 237.7, 429.3, 392.2
# NoZ: 167.8, 361.7, 453.1
# NonELG: 96.1, 102.8, 269.6
# Poorly characterized objects (not included in density modeling, no prediction): 790.1, 135.5, NA
# ----------
# Total based on individual parts: NA, 2966.8, NA
# Total number: 2356.4, 2966.8, 2900.3
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.684, 0.625



# Total time for generating selection volume: 134.40 seconds


# Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred
# 0.62478056857 2900.33906344 1812.0754891 269.600511968 453.12306082 1698.7947239 392.185350147
# Save the selection


# Plotting boundary
# mu_gz
# 0 $\mu_g - \mu_z$ [-0.500, -0.490]
# 25 $\mu_g - \mu_z$ [-0.250, -0.240]
# 50 $\mu_g - \mu_z$ [0.000, 0.010]
# 75 $\mu_g - \mu_z$ [0.250, 0.260]
# 100 $\mu_g - \mu_z$ [0.500, 0.510]
# 125 $\mu_g - \mu_z$ [0.750, 0.760]
# 150 $\mu_g - \mu_z$ [1.000, 1.010]
# 175 $\mu_g - \mu_z$ [1.250, 1.260]
# 200 $\mu_g - \mu_z$ [1.500, 1.510]
# 225 $\mu_g - \mu_z$ [1.750, 1.760]
# 250 $\mu_g - \mu_z$ [2.000, 2.010]
# 275 $\mu_g - \mu_z$ [2.250, 2.260]
# 300 $\mu_g - \mu_z$ [2.500, 2.510]
# 325 $\mu_g - \mu_z$ [2.750, 2.760]
# 350 $\mu_g - \mu_z$ [3.000, 3.010]
# 375 $\mu_g - \mu_z$ [3.250, 3.260]
# 400 $\mu_g - \mu_z$ [3.500, 3.510]
# mu_gr
# 0 $\mu_g - \mu_r$ [-0.850, -0.840]
# 25 $\mu_g - \mu_r$ [-0.600, -0.590]
# 50 $\mu_g - \mu_r$ [-0.350, -0.340]
# 75 $\mu_g - \mu_r$ [-0.100, -0.090]
# 100 $\mu_g - \mu_r$ [0.150, 0.160]
# 125 $\mu_g - \mu_r$ [0.400, 0.410]
# 150 $\mu_g - \mu_r$ [0.650, 0.660]
# 175 $\mu_g - \mu_r$ [0.900, 0.910]
# 200 $\mu_g - \mu_r$ [1.150, 1.160]
# 225 $\mu_g - \mu_r$ [1.400, 1.410]
# 250 $\mu_g - \mu_r$ [1.650, 1.660]
# gmag
# 0 $g$ [19.500, 19.510]
# 25 $g$ [19.750, 19.760]
# 50 $g$ [20.000, 20.010]
# 75 $g$ [20.250, 20.260]
# 100 $g$ [20.500, 20.510]
# 125 $g$ [20.750, 20.760]
# 150 $g$ [21.000, 21.010]
# 175 $g$ [21.250, 21.260]
# 200 $g$ [21.500, 21.510]
# 225 $g$ [21.750, 21.760]
# 250 $g$ [22.000, 22.010]
# 275 $g$ [22.250, 22.260]
# 300 $g$ [22.500, 22.510]
# 325 $g$ [22.750, 22.760]
# 350 $g$ [23.000, 23.010]
# 375 $g$ [23.250, 23.260]
# 400 $g$ [23.500, 23.510]
# 425 $g$ [23.750, 23.760]


# Completed












# /----- Case 5:
# NDM nominal depths, N_tot = 2400, flat FoM, larger fNoZ with external calibration.
# Convolving error and re-parametrizing
# NonELG
# NoZ
# ELG
# Time for convolving error sample: 50.23 seconds


# Start of computing selection region.
# Constructing histograms.
# Time taken: 80.45 seconds
# Applying gaussian smoothing.
# Time taken: 11.36 seconds
# Computing magnitude dependent regularization.
# Time taken: 3.52 seconds
# Computing utility and sorting.
# Time taken: 13.05 seconds
# Flattening the MD histograms.
# Time taken: 9.17 seconds
# Sorting the flattened arrays.
# Time taken: 15.78 seconds
# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 579.5, 1039.8, 1676.5
# NonDESI ELGs: 107.9, 195.9, 344.5
# NoZ: 119.3, 254.1, 505.8
# NonELG: 150.6, 165.6, 248.7
# Poorly characterized objects (not included in density modeling, no prediction): 589.4, 119.0, NA
# ----------
# Total based on individual parts: NA, 1774.5, NA
# Total number: 1546.8, 1774.5, 2858.2
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.622, 0.631



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 722.7, 1324.5, 1676.5
# NonDESI ELGs: 133.0, 241.3, 344.5
# NoZ: 134.2, 289.2, 505.8
# NonELG: 71.8, 77.6, 248.7
# Poorly characterized objects (not included in density modeling, no prediction): 703.9, 94.5, NA
# ----------
# Total based on individual parts: NA, 2027.1, NA
# Total number: 1765.6, 2027.1, 2858.2
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.689, 0.631



# Raw/Weigthed/Predicted number of selection
# ----------
# DESI ELGs: 1066.3, 1940.9, 1676.5
# NonDESI ELGs: 227.2, 411.9, 344.5
# NoZ: 173.1, 392.4, 505.8
# NonELG: 99.6, 106.6, 248.7
# Poorly characterized objects (not included in density modeling, no prediction): 783.1, 138.1, NA
# ----------
# Total based on individual parts: NA, 2989.9, NA
# Total number: 2349.4, 2989.9, 2858.2
# ----------
# Efficiency, weighted vs. prediction (DESI/Ntotal): 0.682, 0.631



# Total time for generating selection volume: 136.21 seconds


# Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred
# 0.630781871099 2858.23668897 1802.92388671 248.735635298 505.848173105 1676.46184344 344.495626746
# Save the selection


# Plotting boundary
# mu_gz
# 0 $\mu_g - \mu_z$ [-0.500, -0.490]
# 25 $\mu_g - \mu_z$ [-0.250, -0.240]
# 50 $\mu_g - \mu_z$ [0.000, 0.010]
# 75 $\mu_g - \mu_z$ [0.250, 0.260]
# 100 $\mu_g - \mu_z$ [0.500, 0.510]
# 125 $\mu_g - \mu_z$ [0.750, 0.760]
# 150 $\mu_g - \mu_z$ [1.000, 1.010]
# 175 $\mu_g - \mu_z$ [1.250, 1.260]
# 200 $\mu_g - \mu_z$ [1.500, 1.510]
# 225 $\mu_g - \mu_z$ [1.750, 1.760]
# 250 $\mu_g - \mu_z$ [2.000, 2.010]
# 275 $\mu_g - \mu_z$ [2.250, 2.260]
# 300 $\mu_g - \mu_z$ [2.500, 2.510]
# 325 $\mu_g - \mu_z$ [2.750, 2.760]
# 350 $\mu_g - \mu_z$ [3.000, 3.010]
# 375 $\mu_g - \mu_z$ [3.250, 3.260]
# 400 $\mu_g - \mu_z$ [3.500, 3.510]
# mu_gr
# 0 $\mu_g - \mu_r$ [-0.850, -0.840]
# 25 $\mu_g - \mu_r$ [-0.600, -0.590]
# 50 $\mu_g - \mu_r$ [-0.350, -0.340]
# 75 $\mu_g - \mu_r$ [-0.100, -0.090]
# 100 $\mu_g - \mu_r$ [0.150, 0.160]
# 125 $\mu_g - \mu_r$ [0.400, 0.410]
# 150 $\mu_g - \mu_r$ [0.650, 0.660]
# 175 $\mu_g - \mu_r$ [0.900, 0.910]
# 200 $\mu_g - \mu_r$ [1.150, 1.160]
# 225 $\mu_g - \mu_r$ [1.400, 1.410]
# 250 $\mu_g - \mu_r$ [1.650, 1.660]
# gmag
# 0 $g$ [19.500, 19.510]
# 25 $g$ [19.750, 19.760]
# 50 $g$ [20.000, 20.010]
# 75 $g$ [20.250, 20.260]
# 100 $g$ [20.500, 20.510]
# 125 $g$ [20.750, 20.760]
# 150 $g$ [21.000, 21.010]
# 175 $g$ [21.250, 21.260]
# 200 $g$ [21.500, 21.510]
# 225 $g$ [21.750, 21.760]
# 250 $g$ [22.000, 22.010]
# 275 $g$ [22.250, 22.260]
# 300 $g$ [22.500, 22.510]
# 325 $g$ [22.750, 22.760]
# 350 $g$ [23.000, 23.010]
# 375 $g$ [23.250, 23.260]
# 400 $g$ [23.500, 23.510]
# 425 $g$ [23.750, 23.760]


# Completed
