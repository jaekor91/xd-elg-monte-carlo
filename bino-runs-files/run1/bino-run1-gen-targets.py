# This is a record of the script that was used to produce targets for run1.

import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
from model_class import *
import sys
import matplotlib.pyplot as plt
import time

def load_tractor_DR5(fname, ibool=None):
    """
    Load select columns
    """
    tbl = load_fits_table(fname)    
    if ibool is not None:
        tbl = tbl[ibool]

    ra, dec = tbl["ra"], tbl["dec"]
    bid = tbl["brickid"]
    bp = tbl["brick_primary"]
    r_dev, r_exp = tbl["shapedev_r"], tbl["shapeexp_r"]
    gflux_raw, rflux_raw, zflux_raw = tbl["flux_g"], tbl["flux_r"], tbl["flux_z"]
    mw_g, mw_r, mw_z = tbl["mw_transmission_g"], tbl["mw_transmission_r"], tbl["mw_transmission_z"]
    gflux, rflux, zflux = gflux_raw/mw_g, rflux_raw/mw_r,zflux_raw/mw_z
    givar, rivar, zivar = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    tycho = tbl["TYCHOVETO"]
    objtype = tbl["type"]
    
    # error
    gf_err = np.sqrt(1./givar)/mw_g
    rf_err = np.sqrt(1./rivar)/mw_r
    zf_err = np.sqrt(1./zivar)/mw_z
    
    return bid, objtype, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask, gf_err, rf_err, zf_err, tycho 

def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)

category = ["NonELG", "NoZ", "ELG"]


# # Focus only on the full data set.
# # 0: Full F34 data
# # 1: F3 data only
# # 2: F4 data only
# # 3-7: CV1-CV5: Sub-sample F34 into five-fold CV sets.
# # 8-10: Magnitude changes. For power law use full data. Not used. 
# # g in [22.5, 23.5], [22.75, 23.75], [23, 24]. 
# sub_sample_name = ["Full", "F3", "F4", "CV1", "CV2", "CV3", "CV4", "CV5", "Mag1", "Mag2", "Mag3"] # No need to touch this
# NK_list = [1]#, 3, 4, 5, 6, 7]
# Niter = 1

# # Monte Carlo area. 
# MC_AREA = 1000 # In sq. deg.

# # Correspond to 1,000 sq. deg.
# # NonELG sample number: 16,896,579
# # NoZ sample number: 4,963,537
# # ELG sample number: 7,925,135
# # 15 seconds

# print "# ----- NDM selections considered ----- #"
# print "- Bit 4: NDM Typical depths, N_tot = 3050, Flat FoM"
# print "- Bit 5: NDM Local depths, N_tot = 3050, Flat FoM"
# print "- Bit 6: NDM Typical depths, N_tot = 3050, Redshift dependent FoM. More precisely. Quadratic dependency."
# print "- Bit 7: NDM Typical depths, N_tot = 3050, Flat FoM, f_NoZ = 1 (rather than 0.25)"
# print "\n\n"




# print "# ----- Model3 ----- #"

# j = 0 
# print "/----- %s -----/" % sub_sample_name[j]

# print "Loading the fitted models"
# instance_model = model3(j)        
# print "Fit MoGs"
# instance_model.fit_MoG(NK_list, "model3", sub_sample_name[j], cache=True, Niter=Niter)
# print "\n"
# print "Fit Pow"
# instance_model.fit_dNdf("model3", "Full", cache=True, Niter=Niter)
# print "\n"

# print "Global grid properties"
# # Selection grid limits and number of bins 
# # var_x, var_y, gmag. Width (0.01, 0.01, 0.01)
# instance_model.var_x_limits = [0.25, 2.45]
# instance_model.var_y_limits = [-0.25, 1.05]
# instance_model.gmag_limits = [21.5, 24.]
# instance_model.num_bins = [220, 130, 250]
# instance_model.N_regular = 1e4

# # Sigma widths to be used in kernel approximation.
# instance_model.sigmas = [5., 5., 2.5]



# print "# ----- Generate global selections and save. ----- #"
# print "Set desired number of objects to 3050 in all cases"
# instance_model.set_num_desired(3050)

# print "- Bit 4: NDM Typical depths, N_tot = 3050, Flat FoM"
# print "Generate intrinsic sample using default (flat) FoM option."
# instance_model.set_area_MC(MC_AREA)             
# instance_model.gen_sample_intrinsic()

# # print "Convolve error to the intrinsic sample."
# start = time.time()
# instance_model.set_err_lims(23.8, 23.4, 22.4, 8) 
# instance_model.gen_err_conv_sample()
# print "Time for convolving error sample: %.2f seconds" % (time.time() - start)

# # Create the selection.
# start = time.time()            
# eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#     N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
# print "Time for generating selection volume: %.2f seconds" % (time.time() - start)

# print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
# print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#     N_ELG_NonDESI_pred

# print "Saving the selection"
# np.save("NDM_cell_select_bit4.npy", instance_model.cell_select)

# print "\n\n"




# print "- Bit 6: NDM Typical depths, N_tot = 3050, Redshift dependent FoM. More precisely. Quadratic dependency."
# print "Generate intrinsic sample using quadratic option."
# instance_model.set_area_MC(MC_AREA)
# instance_model.set_FoM_option("Quadratic_redz")
# instance_model.gen_sample_intrinsic()

# # print "Convolve error to the intrinsic sample."
# start = time.time()
# instance_model.set_err_lims(23.8, 23.4, 22.4, 8) 
# instance_model.gen_err_conv_sample()
# print "Time for convolving error sample: %.2f seconds" % (time.time() - start)

# # Create the selection.
# start = time.time()            
# eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#     N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
# print "Time for generating selection volume: %.2f seconds" % (time.time() - start)

# print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
# print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#     N_ELG_NonDESI_pred

# print "Saving the selection"
# np.save("NDM_cell_select_bit6.npy", instance_model.cell_select)

# print "\n\n"


# print "- Bit 7: NDM Typical depths, N_tot = 3050, Flat FoM, f_NoZ = 0.5 (rather than 0.25)"
# print "Generate intrinsic sample using default (flat) FoM option."
# instance_model.set_FoM_option("flat")
# instance_model.set_f_NoZ(0.5)
# instance_model.set_area_MC(MC_AREA)             
# instance_model.gen_sample_intrinsic()

# # print "Convolve error to the intrinsic sample."
# start = time.time()
# instance_model.set_err_lims(23.8, 23.4, 22.4, 8) 
# instance_model.gen_err_conv_sample()
# print "Time for convolving error sample: %.2f seconds" % (time.time() - start)

# # Create the selection.
# start = time.time()            
# eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#     N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
# print "Time for generating selection volume: %.2f seconds" % (time.time() - start)

# print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
# print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#     N_ELG_NonDESI_pred

# print "Saving the selection"
# np.save("NDM_cell_select_bit7.npy", instance_model.cell_select)

# print "\n\n"



# plot_boundary = True
# if plot_boundary:
#     print "# ----- Plot the boundaries ----- #"
#     print "Note that green dots are the new selection region and the red is fiducial (typical depths, flat FoM and num desired 2400)"

#     print "Create the selection for the default"
#     instance_model.set_num_desired(2400)
#     instance_model.set_FoM_option("flat")
#     instance_model.set_f_NoZ(0.25)
#     instance_model.set_area_MC(MC_AREA)             
#     instance_model.gen_sample_intrinsic()

#     # print "Convolve error to the intrinsic sample."
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

#     print "Saving the selection"
#     np.save("NDM_cell_select_default.npy", instance_model.cell_select)
#     cell_centers_default = instance_model.cell_select_centers()
    
#     print "For selection 4, 6 and 7, draw comparison plots"

#     for bit_num in [4, 6, 7]:       
#         # Use the model instance to generate cell centers for the plotting purpose.
#         instance_model.cell_select = np.load("NDM_cell_select_bit%d.npy" % bit_num)

#         print "/---- Plotting boundary ----/"
#         for i in [0, 1, 2]:
#             instance_model.gen_select_boundary_slices(slice_dir = i, model_tag="model3", cv_tag="NDM-default-vs-bit%d" % bit_num,\
#             var_x_ext = cell_centers_default[:, 0], var_y_ext = cell_centers_default[:, 1], gmag_ext = cell_centers_default[:, 2],\
#             use_parameterized_ext=True, plot_ext =True, alpha_ext=0.3, guide=True)







print "# ----- Target Tractor files used ----- #"
print "Stripe 82 - 1hr: 0118p010"
print "Stripe 82 - 3hr: 0393p002"
print "8h+30: 1202p275"
print "DEEP2: TBD"
# region name, ra/dec centers
region_names = ["St82-1hr", "St82-3hr", "8h+30"]
center_ra = [11.8, 39.3, 120.2]
center_dec = [1.0, 0.2, 27.5]


# File address for the test catalogs
fdir1 = "/Users/jaehyeon/Documents/Research/ELG_target_selection/XD-paper2/bino-runs-files/run1/"

# File address for Tycho star
tycho_directory = "/Users/jaehyeon/Documents/Research/ELG_target_selection/data-repository/"


print "\n"




# def load_RF_targets(fname):
#     data = load_fits_table(fname)

#     return data["ra"], data["dec"], data["brickid"], data["objid"], data["Priority_RF"]

# fname = fdir1 + "List_RF_NGC.fits"
# ra_RF_north, dec_RF_north, brickid_RF_north, objid_RF_north, priority_RF_north = load_RF_targets(fname)

# fname = fdir1 + "List_RF_SGC.fits"
# ra_RF_south, dec_RF_south, brickid_RF_south, objid_RF_south, priority_RF_south = load_RF_targets(fname)

# print "Total # RF targets in North: %d" % ra_RF_north.size
# print "Total # RF targets in South: %d" % ra_RF_south.size

# print "Priority: North/South"
# for i in range(1, 4):
#     print i, ":", (priority_RF_north==i).sum(), "/", (priority_RF_south==i).sum()
    
# ra_RF = np.concatenate((ra_RF_north, ra_RF_south))
# dec_RF = np.concatenate((dec_RF_north, dec_RF_south))
# priority_RF = np.concatenate((priority_RF_north, priority_RF_south))






# tol = 0.5

# Load Tractor test catalog
bid, objtype, bp, ra_before_cut, dec_before_cut, gflux_raw, rflux_raw, zflux_raw, \
gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, \
r_allmask, z_allmask, gf_err, rf_err, zf_err, tycho = load_tractor_DR5(fdir1+"DR5-Tractor-bino-run1.fits")

print "Number of objects before any cut: %d" % ra_before_cut.size

# Apply the selection including tycho mask.
# Also, restrict to three areas
# iregion = np.zeros(ra_before_cut.size, dtype=bool)

# for i, name in enumerate(region_names):
#     # Selecting only objects within the regions of interest
#     ra_c = center_ra[i]
#     dec_c = center_dec[i]
#     ibool = (ra_before_cut < ra_c+tol) & (ra_before_cut > ra_c-tol) & (dec_before_cut > dec_c-tol) & (dec_before_cut < dec_c+tol)
    
#     iregion = np.logical_or(ibool, iregion)

# ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0)\
# & (givar>0) & (rivar>0) & (zivar>0)\
# & (tycho==0) & iregion

# # Re-load the file, looking at only interesting objects.
# bid, objtype, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, \
# gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, \
# r_allmask, z_allmask, gf_err, rf_err, zf_err, tycho = load_tractor_DR5(fdir1+"DR5-Tractor-bino-run1.fits", ibool=ibool)

# print "Number of objects after quality cut: %d" % ra.size





# print "# ---- Produce target files ----- #"
# # - For each target area, select objects from the test catalog within 0.5 range from the center
# # - Create a selection bit vector for all objects.
# # - Apply global selections and adjust the bit vector.
# # - Apply the local selection and save the local selection with the field name and the bit descriptor.
# # - Load RF targets for each field and cross-match and assign proper bits.

# tol = 0.5 # Region to select from 
    
# for i, name in enumerate(region_names):
#     print "Field %d: %s " % (i, name)
    
#     # Selecting only objects within the regions of interest
#     ra_c = center_ra[i]
#     dec_c = center_dec[i]
#     iregion = (ra < ra_c+tol) & (ra > ra_c-tol) & (dec > dec_c-tol) & (dec < dec_c+tol)
    
#     ra_tmp = ra[iregion]
#     dec_tmp = dec[iregion] 
#     gflux_tmp = gflux[iregion]  
#     rflux_tmp = rflux[iregion] 
#     zflux_tmp = zflux[iregion] 


#     # Create a bit mask for the current tractor catalog.
#     bit_mask = np.zeros(ra_tmp.size, dtype=int)

#     # Apply the selection and update the bit mask
# #     print "Applying selection bit num"
#     for bit_num in [4, 5, 6, 7]: # There are four models to be applied
# #     for bit_num in [4, 6, 7]: # Don't include 5 while testing.
#         if bit_num != 5:  # If the selection has been already computed.
#             instance_model.cell_select = np.load("NDM_cell_select_bit%d.npy" % bit_num)
#             iselected = instance_model.apply_selection(gflux_tmp, rflux_tmp, zflux_tmp)
#             bit_mask[iselected] += 2**bit_num
#     #             print bit_num, "selected: %d" % iselected.sum()
#         else:  # If the selection to be applied needs to be computed, then do the following.
#             print "Generate intrinsic sample using default (flat) FoM option but using local depths."
#             instance_model.set_FoM_option("flat")
#             instance_model.set_f_NoZ(0.25)
#             instance_model.set_num_desired(3050)
#             instance_model.set_area_MC(MC_AREA)             
#             instance_model.gen_sample_intrinsic()


#             # Compute local depths
#             glim_err = median_mag_depth(gf_err[iregion])
#             rlim_err = median_mag_depth(rf_err[iregion])
#             zlim_err = median_mag_depth(zf_err[iregion])
#             oii_lim_err = 8


#             # print "Convolve error to the intrinsic sample."
#             start = time.time()
#             instance_model.set_err_lims(glim_err, rlim_err, zlim_err, oii_lim_err) # Training data is deep!

#             instance_model.gen_err_conv_sample()
#             print "Time for convolving error sample: %.2f seconds" % (time.time() - start)

#             # Create the selection.
#             start = time.time()            
#             eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#                 N_ELG_NonDESI_pred = instance_model.gen_selection_volume_scipy()
#             print "Time for generating selection volume: %.2f seconds" % (time.time() - start)

#             print "Eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred, N_ELG_NonDESI_pred"
#             print eff_pred, Ntotal_pred, Ngood_pred, N_NonELG_pred, N_NoZ_pred, N_ELG_DESI_pred,\
#                 N_ELG_NonDESI_pred

#             print "Saving the selection"
#             np.save("NDM_cell_select_bit5_%s.npy", instance_model.cell_select)

#             iselected = instance_model.apply_selection(gflux_tmp, rflux_tmp, zflux_tmp)
#             bit_mask[iselected] += 2**bit_num

#             print "\n"


#     # Load RF samples and assign the proper bit numbers
#     #     - Bit 8: RF tuned for 2400 deg
#     #     - Bit 9: RF tuned for 3000 deg
#     #     - Bit 10: RF tuned for 3000 deg with an additional gmag cut.
#     idx1, idx2 = crossmatch_cat1_to_cat2(ra_tmp, dec_tmp, ra_RF, dec_RF, tol=1e-2/(deg2arcsec+1e-12))    
#     RF_priority_tmp = priority_RF[idx2]

#     mask_update_tmp = np.zeros(idx1.size, dtype=int)
#     for k, bit_num in zip(range(1, 5), [8, 9, 10, 11]):
#         mask_update_tmp[RF_priority_tmp==k] += 2**bit_num
#     bit_mask[idx1] += mask_update_tmp

#     # Tally how many were selected in each category    
#     print "Tally of selections"
#     for bit_num in range(4, 12):
#         print bit_num, (np.bitwise_and(bit_mask, 2**bit_num)>0).sum()

#     # Finding the union number.
#     iNDM = np.bitwise_and(bit_mask, 2**4 + 2**5 +  2**6 +  2**7) > 0 # NDM targets
#     NDM_total = (iNDM).sum()

#     # Finding the union number.
#     iRF = np.bitwise_and(bit_mask, 2**8 + 2**9 +  2**10 +  2**11) > 0 # RF targets
#     RF_total = (iRF).sum()

#     # Union
#     iUnion = np.logical_or(iRF, iNDM)
#     Union_total = (iUnion).sum()

#     # Intersection
#     iIntersection  = np.logical_and(iRF, iNDM)
#     Intersection_total = (iIntersection).sum()

#     print "Number in the union of NDM selections: %d" % NDM_total
#     print "Number in the union of RF selections: %d" % RF_total
#     print "Number in the total union: %d" % (Union_total)
#     print "Number in the intersection: %d" % (Intersection_total)
    

#     # Save the targets
#     ra_targets = ra_tmp[iUnion]
#     dec_targets = dec_tmp[iUnion]
#     g_targets = flux2mag(gflux_tmp)[iUnion]
#     mask_targets = bit_mask[iUnion]


#     f = open("%d-%s-targets.txt" % (i, name), "w")
#     f.write("name,ra,dec,magnitude,priority\n")
#     for i in range(ra_targets.size):
#         line = ",".join([str(x) for x in [mask_targets[i], ra_targets[i], dec_targets[i], g_targets[i], 2]])
#         line += "\n"
#         f.write(line)
#     f.close()    

#     print "\n\n"
    
    

from astropy.io import ascii

# boss.r110_130.d25_30.txt		standard_bino.r110_130.d25_30.txt
# boss.stripe82.txt			standard_bino.stripe82.txt
# Columns
# 1: RA
# 2: DEC
# 4: g

boss_8hr = ascii.read(fdir1+"boss.r110_130.d25_30.txt")
boss_St82 = ascii.read(fdir1+"boss.stripe82.txt")
standard_8hr = ascii.read(fdir1+"standard_bino.r110_130.d25_30.txt")
standard_St82 = ascii.read(fdir1+"standard_bino.stripe82.txt")

# Import targets
targets1 = ascii.read("0-St82-1hr-targets.txt")
targets2 = ascii.read("1-St82-3hr-targets.txt")
targets3 = ascii.read("2-8h+30-targets.txt")


# tol = 0.45
# fig, ax_list = plt.subplots(1, 3, figsize=(30,7))
# for i, targets in enumerate([targets1, targets2, targets3]):
#     ra_tmp, dec_tmp, bit_mask = targets["ra"].data, targets["dec"].data, targets["name"].data
    
#     # Finding the union number.
#     iNDM = np.bitwise_and(bit_mask, 2**4 + 2**5 +  2**6 +  2**7) > 0 # NDM targets
#     NDM_total = (iNDM).sum()

#     # Finding the union number.
#     iRF = np.bitwise_and(bit_mask, 2**8 + 2**9 +  2**10 +  2**11) > 0 # RF targets
#     RF_total = (iRF).sum()

#     # Union
#     iUnion = np.logical_or(iRF, iNDM)
#     Union_total = (iUnion).sum()

#     # Intersection
#     iIntersection  = np.logical_and(iRF, iNDM)
#     Intersection_total = (iIntersection).sum()
    
    
#     assert (iNDM & ~iRF).sum() == NDM_total-Intersection_total
    
#     if i <= 1:
#         stars = standard_St82
#         galaxies = boss_St82
#         # Stars
#         ra_stars, dec_stars, gmag_stars = stars["ra"].data, stars["dec"].data, stars["psfMag_g"].data
#         ra_c = center_ra[i]
#         dec_c = center_dec[i]
#         ibool = (ra_stars < ra_c+tol) & (ra_stars > ra_c-tol) & (dec_stars > dec_c-tol) & (dec_stars < dec_c+tol)
#         # Gals
#         ra_gal, dec_gal, gmag_gal = galaxies["col1"].data, galaxies["col2"].data, galaxies["col4"].data
#         ibool2 = (ra_gal < ra_c+tol) & (ra_gal > ra_c-tol) & (dec_gal > dec_c-tol) & (dec_gal < dec_c+tol)        
#     else:
#         stars = standard_8hr
#         galaxies = boss_8hr
#         # Stars 
#         ra_stars, dec_stars, gmag_stars = stars["col1"].data, stars["col2"].data, stars["col4"].data
#         ra_c = center_ra[i]
#         dec_c = center_dec[i]
#         ibool = (ra_stars < ra_c+tol) & (ra_stars > ra_c-tol) & (dec_stars > dec_c-tol) & (dec_stars < dec_c+tol)
#         # gal
#         # Gals
#         ra_gal, dec_gal, gmag_gal = galaxies["col1"].data, galaxies["col2"].data, galaxies["col4"].data
#         ibool2 = (ra_gal < ra_c+tol) & (ra_gal > ra_c-tol) & (dec_gal > dec_c-tol) & (dec_gal < dec_c+tol)        
        
        

#     # Plotting stars
#     ax_list[i].scatter(ra_stars[ibool], dec_stars[ibool], edgecolors="none", c="green", s=50, alpha=0.75, label = "stars: %d" % ibool.sum())    
    
#     # Plotting galaxies
#     ax_list[i].scatter(ra_gal[ibool2], dec_gal[ibool2], edgecolors="none", c="orange", s=50, label = "LRG: %d" % ibool2.sum())    
    
#     # Plotting targets
#     ax_list[i].scatter(ra_tmp[iNDM & ~iRF], dec_tmp[iNDM & ~iRF], edgecolors="none", c="blue", s=5, label = "NDM only: %d" % (NDM_total-Intersection_total))
#     ax_list[i].scatter(ra_tmp[iRF & ~iNDM], dec_tmp[iRF & ~iNDM], edgecolors="none", c="red", s=5, label = "RF only: %d" % (RF_total-Intersection_total))
#     ax_list[i].scatter(ra_tmp[iIntersection], dec_tmp[iIntersection], edgecolors="none", c="black", s=5, label = "AND: %d" % Intersection_total)    
    
#     ax_list[i].set_title("%s" % (region_names[i]), fontsize=25)
#     ax_list[i].set_xlabel("RA", fontsize=25)    
#     ax_list[i].set_ylabel("DEC", fontsize=25)            
#     ax_list[i].legend(loc="lower right", fontsize=15)            
#     ax_list[i].axis("equal")        


# plt.savefig("bino-run1-targets-by-field.png", dpi=400, bbox_inches="tight")
# plt.show()
# plt.close()



def med_x1_minus_x2(x1, x2):
    """
    Computer median difference
    """
    return np.median(x1-x2)


tol = 0.45

fig, ax_list = plt.subplots(1, 3, figsize=(30,7))  # For plotting astrometric difference
for i, targets in enumerate([targets1, targets2, targets3]):
    ra_tmp, dec_tmp, bit_mask, g_tmp =\
    targets["ra"].data, targets["dec"].data, targets["name"].data, targets["magnitude"].data
    
    # Finding the union number.
    iNDM = np.bitwise_and(bit_mask, 2**4 + 2**5 +  2**6 +  2**7) > 0 # NDM targets
    NDM_total = (iNDM).sum()

    # Finding the union number.
    iRF = np.bitwise_and(bit_mask, 2**8 + 2**9 +  2**10 +  2**11) > 0 # RF targets
    RF_total = (iRF).sum()

    # Union
    iUnion = np.logical_or(iRF, iNDM)
    Union_total = (iUnion).sum()

    # Intersection
    iIntersection  = np.logical_and(iRF, iNDM)
    Intersection_total = (iIntersection).sum()
    
    
    assert (iNDM & ~iRF).sum() == NDM_total-Intersection_total
    
    
    
    if i <= 1:
        stars = standard_St82
        galaxies = boss_St82
        # Stars
        ra_stars, dec_stars, gmag_stars = stars["ra"].data, stars["dec"].data, stars["psfMag_g"].data
        # Gals
        ra_gal, dec_gal, gmag_gal = galaxies["col1"].data, galaxies["col2"].data, galaxies["col4"].data

    else:
        stars = standard_8hr
        galaxies = boss_8hr
        # Stars 
        ra_stars, dec_stars, gmag_stars = stars["col1"].data, stars["col2"].data, stars["col4"].data
        # gal
        ra_gal, dec_gal, gmag_gal = galaxies["col1"].data, galaxies["col2"].data, galaxies["col4"].data
        
    # Concetrate on objects in the region of interest
    ra_c = center_ra[i]
    dec_c = center_dec[i]        
    ibool = (ra_stars < ra_c+tol) & (ra_stars > ra_c-tol) & (dec_stars > dec_c-tol) & (dec_stars < dec_c+tol)
    ibool2 = (ra_gal < ra_c+tol) & (ra_gal > ra_c-tol) & (dec_gal > dec_c-tol) & (dec_gal < dec_c+tol)
    ra_stars, dec_stars, gmag_stars = ra_stars[ibool], dec_stars[ibool], gmag_stars[ibool] 
    ra_gal, dec_gal, gmag_gal = ra_gal[ibool2], dec_gal[ibool2], gmag_gal[ibool2]
    
    # Make astrometric correction for each set based on targets
    # stars
    idx1, idx2 = crossmatch_cat1_to_cat2(ra_stars, dec_stars, ra_before_cut, dec_before_cut, tol=0.5/(deg2arcsec+1e-12))    
    ra_med_diff = med_x1_minus_x2(ra_stars[idx1], ra_before_cut[idx2])
    dec_med_diff = med_x1_minus_x2(dec_stars[idx1], dec_before_cut[idx2])
    ra_stars -= ra_med_diff
    dec_stars -= dec_med_diff    
    # Plotting stars radec difference
    ax_list[i].scatter(3600*ra_stars[idx1]-3600*ra_before_cut[idx2], 3600*dec_stars[idx1]-3600*dec_before_cut[idx2], edgecolors="none",\
                       c="green", s=50, alpha=0.75, label = "star")    

    # gal
    idx1, idx2 = crossmatch_cat1_to_cat2(ra_gal, dec_gal, ra_before_cut, dec_before_cut, tol=0.5/(deg2arcsec+1e-12))    
    ra_med_diff = med_x1_minus_x2(ra_gal[idx1], ra_before_cut[idx2])
    dec_med_diff = med_x1_minus_x2(dec_gal[idx1], dec_before_cut[idx2])
    # Plotting galaxies radec differences 
    ra_gal -= ra_med_diff
    dec_gal -= dec_med_diff        
    ax_list[i].scatter(3600*ra_gal[idx1]-3600*ra_before_cut[idx2], 3600*dec_gal[idx1]-3600*dec_before_cut[idx2],\
                       edgecolors="none", c="orange", s=50, label="LRG")
    
    ax_list[i].axhline(y=0, c="black")
    ax_list[i].axvline(x=0, c="black")
    ax_list[i].set_title("%s" % (region_names[i]), fontsize=25)
    ax_list[i].set_xlabel("RA", fontsize=25)    
    ax_list[i].set_ylabel("DEC", fontsize=25)            
    ax_list[i].legend(loc="lower right", fontsize=15)            
    ax_list[i].axis("equal")    
    tol_astrometry = 1.
    ax_list[i].set_xlim([-tol_astrometry, tol_astrometry])
    ax_list[i].set_ylim([-tol_astrometry, tol_astrometry])    
    
    # Create a file saving targets as well objects
    f = open("%d-%s-bino-input-catalog.txt" % (i, region_names[i]), "w")
    f.write("name,ra,dec,magnitude,priority,type\n")
    # Targets
    for j in range(ra_tmp.size):
        line = ",".join([str(x) for x in [bit_mask[j], ra_tmp[j], dec_tmp[j], g_tmp[j], 2, 1]])
        line += "\n"
        f.write(line)
    
    # Add stars
    for j in range(ra_stars.size):
        line = ",".join([str(x) for x in [2**1, ra_stars[j], dec_stars[j], gmag_stars[j], 1, 3]])
        line += "\n"
        f.write(line)        
        
    
    # Add galaxies
    for j in range(ra_gal.size):
        line = ",".join([str(x) for x in [2**2, ra_gal[j], dec_gal[j], gmag_gal[j], 1, 1]])
        line += "\n"
        f.write(line)        
    
    f.close()    

plt.savefig("bino-run1-astrometric-correction-by-field.png")
plt.show()
plt.close()




# Slit dot mask

data = load_fits_table("DR5-matched-to-DEEP2-f4-glim30.fits")
ibool = (data["RED_Z"] > 0.7) & (data["ZQUALITY"] >=3)
ra_before_cut, dec_before_cut = data["RA"], data["DEC"]
data = data[ibool]
ra, dec, gmag, redz = data["RA"], data["DEC"], flux2mag(data["flux_g"]/data["mw_transmission_g"]), data["RED_Z"]



# Astrometric difference and targets
i = 0
fig, ax_list = plt.subplots(1, 2, figsize=(20,7))  

stars = standard_St82
galaxies = boss_St82
# Stars
ra_stars, dec_stars, gmag_stars = stars["ra"].data, stars["dec"].data, stars["psfMag_g"].data
# Gals
ra_gal, dec_gal, gmag_gal = galaxies["col1"].data, galaxies["col2"].data, galaxies["col4"].data
        
# Concetrate on objects in the region of interest
tol = 0.75
ra_c = 37
dec_c = .5        
ibool = (ra_stars < ra_c+tol) & (ra_stars > ra_c-tol) & (dec_stars > dec_c-tol) & (dec_stars < dec_c+tol)
ibool2 = (ra_gal < ra_c+tol) & (ra_gal > ra_c-tol) & (dec_gal > dec_c-tol) & (dec_gal < dec_c+tol)
ra_stars, dec_stars, gmag_stars = ra_stars[ibool], dec_stars[ibool], gmag_stars[ibool] 
ra_gal, dec_gal, gmag_gal = ra_gal[ibool2], dec_gal[ibool2], gmag_gal[ibool2]

# Make astrometric correction for each set based on targets
# stars
idx1, idx2 = crossmatch_cat1_to_cat2(ra_stars, dec_stars, ra_before_cut, dec_before_cut, tol=0.5/(deg2arcsec+1e-12))    
ra_med_diff = med_x1_minus_x2(ra_stars[idx1], ra_before_cut[idx2])
dec_med_diff = med_x1_minus_x2(dec_stars[idx1], dec_before_cut[idx2])
ra_stars -= ra_med_diff
dec_stars -= dec_med_diff    
# Plotting stars radec difference
ax_list[i].scatter(3600*ra_stars[idx1]-3600*ra_before_cut[idx2], 3600*dec_stars[idx1]-3600*dec_before_cut[idx2], edgecolors="none",\
                   c="green", s=25, alpha=0.75, label = "star")    

# gal
idx1, idx2 = crossmatch_cat1_to_cat2(ra_gal, dec_gal, ra_before_cut, dec_before_cut, tol=0.5/(deg2arcsec+1e-12))    
ra_med_diff = med_x1_minus_x2(ra_gal[idx1], ra_before_cut[idx2])
dec_med_diff = med_x1_minus_x2(dec_gal[idx1], dec_before_cut[idx2])
# Plotting galaxies radec differences 
ra_gal -= ra_med_diff
dec_gal -= dec_med_diff        
ax_list[i].scatter(3600*ra_gal[idx1]-3600*ra_before_cut[idx2], 3600*dec_gal[idx1]-3600*dec_before_cut[idx2],\
                   edgecolors="none", c="orange", s=25, label="LRG")

ax_list[i].axhline(y=0, c="black")
ax_list[i].axvline(x=0, c="black")
ax_list[i].set_title("Diff after astrometric correction", fontsize=25)
ax_list[i].set_xlabel("RA", fontsize=25)    
ax_list[i].set_ylabel("DEC", fontsize=25)            
ax_list[i].legend(loc="lower right", fontsize=15)            
ax_list[i].axis("equal")    
tol_astrometry = 1.
ax_list[i].set_xlim([-tol_astrometry, tol_astrometry])
ax_list[i].set_ylim([-tol_astrometry, tol_astrometry])    


# Plotting the targets and stars/galaxies
i=1
ax_list[i].scatter(ra, dec, c="black", s=2.5, label="ELG")
ax_list[i].scatter(ra_stars, dec_stars, c="green", s=25, label="stars", alpha=0.75, edgecolors="None")
ax_list[i].scatter(ra_gal, dec_gal, c="orange", s=25, label="LRG", alpha=0.75, edgecolors="None")
ax_list[i].set_xlabel("RA", fontsize=25)    
ax_list[i].set_ylabel("DEC", fontsize=25)            
ax_list[i].legend(loc="lower right", fontsize=15)            
ax_list[i].axis("equal")    



plt.savefig("bino-run1-slitdot.png")
plt.show()
plt.close()


# Create a file saving targets as well objects
f = open("slitdot-bino-input-catalog.txt", "w")
f.write("name,ra,dec,magnitude,priority,type\n")
# Targets
for j in range(ra.size):
    line = ",".join([str(x) for x in ["ELG-" + str(redz[j]), ra[j], dec[j], gmag[j], 2, 1]])
    line += "\n"
    f.write(line)

# Add stars
for j in range(ra_stars.size):
    line = ",".join([str(x) for x in [2**1, ra_stars[j], dec_stars[j], gmag_stars[j], 1, 3]])
    line += "\n"
    f.write(line)        


# Add galaxies
for j in range(ra_gal.size):
    line = ",".join([str(x) for x in [2**2, ra_gal[j], dec_gal[j], gmag_gal[j], 1, 1]])
    line += "\n"
    f.write(line)        

f.close()    