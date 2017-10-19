# This script is used to generate target samples for observation.
# The bulk of the code is dedicated to generating NDM samples
# given Tractor or Sweep file names.

# The following analysis is performed field by field. To select fields, 
# 1) Use the CCD files and determine regions of the sky (in HEALPix) where the 
# Nexp is at least twice in all three bands for almost all of the pixels.
# 2) Find all such pixels and rank order them by coverage.
# 3) According to observational constraints, select pixels that are appropriate to target.
# 4) If there are enough such fields, make an appropriate coverage cut (say, 95%)
# and select randomly (say, 30) fields. 
# 5) Sort the fields according to RA/DEC and name each field by the sorted order + center coordinate in
# tractor file style.
# 6) For each field, find the relevant Tractor of Sweeps file, identify a block (say, 0.25 x 0.25 deg sq)
# around the center, and save the resulting objects in the selection in a fits file of the same name 
# modified in fron by the sorted order.
# 7) Apply consistent quality cuts to all objects before saving.

# # Define the bit mask.
# - Bit 1: NDM Typical depths, N_tot = 3000, Flat FoM
# - Bit 2: NDM Local depths, N_tot = 3000, Flat FoM
# - Bit 3: NDM Typical depths, N_tot = 3000, Redshift dependent FoM
# - Bit 4: NDM Typical depths, N_tot = 3000, Flat FoM, f_NoZ = 1 (rather than 0.25)
# - Bit 5: RF tuned for 2400 deg
# - Bit 6: RF tuned for 3000 deg
# - Bit 7: RF tuned for 3000 deg with an additional gmag cut.

# If all bits are off, that is, bit mask = 0, the object is not selected for observation.

# Selection 
# 1) Given the files, for each filed, generate NDM samples according to the various criteria.
# To do this, first apply "Bit 1" selection to all objects and flip on the corresponding bit if selected.
# 2) Perform this repeatedly until all NDM options are exhausted.
# 3) For each NDM option, remember to save the cell_select for each options used. "cell_select_option_XX"
# For Bit 2, "cell_select_option_2_fname_gXX_rXX_zXX", where g r z are followed by the typical depths.
# This is to remember exactly what was done. For each run, specify which bin sizes were used

# External RF selections are provided in the form of boolean vectors that I can apply directly to the files produced above.




# Remember:
# - Use the same quality cuts for both RF and NDM.


# For now, I use one of the Tractor files for this. 

fname = ""


























