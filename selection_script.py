import numpy as np
import numba as nb
from astropy.io import ascii, fits

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)    


def load_fits_table(fname):
    """Given the file name, load  the first extension table."""
    return fits.open(fname)[1].data


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
    gflux, rflux, zflux = gflux_raw/tbl["mw_transmission_g"], rflux_raw/tbl["mw_transmission_r"],zflux_raw/tbl["mw_transmission_z"]
    givar, rivar, zivar = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    objtype = tbl["type"]
    
    return bid, objtype, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask


def mag2flux(mag):
    return 10**(0.4*(22.5-mag))


def apply_selection(fname, option=1):
    """
    Take file name (fname), extract useful quantities, apply appropriate quality cuts,
    and given gflux, rflux, zflux of samples, return a boolean vector that gives the selection.

    Temporary use only, beginning October 16, 2017. 
    """
    if option == 1:
        cell_select = np.load("cell_select_002.npy")
    elif option == 2:
        cell_select = np.load("cell_select_005.npy")        
    else:
        "Must choose either option 1 or 2."
        assert False
    
    # Extract necessary columns.
    bid, objtype, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5(fname)

    # Compute parameterization
    mu_g = flux2asinh_mag(gflux, band = "g")
    mu_r = flux2asinh_mag(rflux, band = "r")
    mu_z = flux2asinh_mag(zflux, band = "z")

    var_x = mu_g - mu_z
    var_y = mu_g - mu_r    

    # Place holder for the ansewr
    Nobjs = ra.size
    iselect = np.zeros(Nobjs, dtype=bool)



    # Quality and flux cuts
    ibool = bp & (g_allmask==0) & (r_allmask==0) & (z_allmask==0)\
    & (givar>0) & (rivar>0) & (zivar>0)\
    & (gflux > mag2flux(24)) & (gflux < mag2flux(21))\
    & np.logical_or( ((0.55*(var_x)+0.) > (var_y)) & (var_y < 1.3), var_y <0.3) # Reject most of low redshift conntaminants by line cuts


    # Focusing on the subset that passes the above cut
    bid, objtype, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar, rivar, zivar, r_dev, r_exp, g_allmask, r_allmask, z_allmask = load_tractor_DR5(fname, ibool=ibool)

    # var_x, var_y, gmag. Width (0.01, 0.01, 0.01)
    var_x_limits = [0.25, 2.45]
    var_y_limits = [-0.25, 1.05]
    gmag_limits = [21.5, 24.]
    num_bins = [220, 130, 250]

    # Compute parametrization again
    mu_g = flux2asinh_mag(gflux, band = "g")
    mu_r = flux2asinh_mag(rflux, band = "r")
    mu_z = flux2asinh_mag(zflux, band = "z")

    var_x = mu_g - mu_z
    var_y = mu_g - mu_r
    gmag = flux2mag(gflux)

    samples = [var_x, var_y, gmag]

    # Generate cell number 
    cell_number = multdim_grid_cell_number(samples, 3, [var_x_limits, var_y_limits, gmag_limits], num_bins)
    
    # Sort the cell number
    idx_sort = cell_number.argsort()
    cell_number = cell_number[idx_sort]

    # Placeholder for selection vector
    iselect_tmp = check_in_arr2(cell_number, cell_select)
#     iselect_tmp = np.in1d(cell_number, cell_select)

    # The last step is necessary in order for iselect to have the same order as the input sample variables.
    idx_undo_sort = idx_sort.argsort()        
    iselect_tmp = iselect_tmp[idx_undo_sort]

    # Answer
    iselect[ibool] = iselect_tmp

    return iselect



def flux2asinh_mag(flux, band = "g"):
    """
    Returns asinh magnitude. The b parameter is set following discussion surrounding
    eq (9) of the paper on luptitude. b = 1.042 sig_f. 
    
    Sig_f for each fitler has been obtained based on DEEP2 deep fields and is in nanomaggies.

    Sig_oii is based on DEEP2 OII flux values in 10^-17 ergs/cm^2/s unit.
    """
    b = None
    if band == "g":
        b = 1.042 * 0.0284297
    elif band == "r":
        b = 1.042 * 0.0421544
    elif band == "z":
        b = 1.042 * 0.122832
    elif band == "oii":
        b = 1.042 * 0.574175
    return 22.5-2.5 * np.log10(b) - 2.5 * np.log10(np.e) * np.arcsinh(flux/(2*b))



def asinh_mag2flux(mu, band = "g"):
    """
    Invsere of flux2asinh_mag
    """
    b = None
    if band == "g":
        b = 1.042 * 0.0284297
    elif band == "r":
        b = 1.042 * 0.0421544
    elif band == "z":
        b = 1.042 * 0.122832
    elif band == "oii":
        b = 1.042 * 0.574175
        
    flux = 2* b * np.sinh((22.5-2.5 * np.log10(b) - mu) / (2.5 * np.log10(np.e)))
    return flux


@nb.jit
def check_in_arr2(arr1, arr2):
    """
    Given two sorted integer arrays arr1, arr2, return a boolean vector of size arr1.size,
    where each element i indicate whether the value of arr1[i] is in arr2.
    """
    N_arr1 = arr1.size
    N_arr2 = arr2.size
    
    # Vector to return
    iselect = np.zeros(N_arr1, dtype=bool)
    
    # First, check whether elements from arr1 is within the range of arr2
    if (arr1[-1] < arr2[0]) or (arr1[0] > arr2[-1]):
        return iselect
    else: # Otherwise, for each element in arr2, incrementally search for in arr1 the same elements
        idx = 0
        for arr2_current_el in arr2:
            while arr1[idx] < arr2_current_el: # Keep incrementing arr1 idx until we reach arr2_current value.
                idx+=1
            if arr1[idx] == arr2_current_el:
                while arr1[idx] == arr2_current_el:
                    iselect[idx] = True
                    idx+=1
                
        return iselect    


def multdim_grid_cell_number(samples, ND, limits, num_bins):
    """
    Given samples array, return the cell each sample belongs to, where the cell is an 
    element of a ND-dimensional grid defined by limits and numb_bins.
    
    More specifically, each cell can be identified by its bin indices.
    If there are three variables, v0, v1, v2, which have N0, N1, N2 number 
    of bins, and the cell corresponds to (n0, n1, n2)-th bin,
    then cell_number = (n0* N1 * N2) + (n1 * N2) + n2. 
    
    Note that we use zero indexing. If an object falls outside the binning range,
    it's assigned cell number "-1".
    
    INPUT: All arrays must be numpy arrays. 
        - samples, ND: List of linear numpy arrays [var1, var2, var3, ...] with size [Nsample].
        Cell numbers calculated based on the first ND variables. 
        - limits: List of limits for each dimension. 
        - num_bins: Number of bins to use for each dimension

    Output:
        - cell_number
    """
    Nsample = samples[0].size
    cell_number = np.zeros(Nsample, dtype=int)
    ibool = np.zeros(Nsample, dtype=bool) # For global correction afterwards.
    
    for i in range(ND): # For each variable to be considered.
        X = samples[i]
        Xmin, Xmax = limits[i]
        _, dX = np.linspace(Xmin, Xmax, num_bins[i]+1, endpoint=True, retstep=True)
        X_bin_idx = gen_bin_idx(X, Xmin, dX) # bin_idx of each sample
        if i < ND-1:
            cell_number += X_bin_idx * np.multiply.reduce(num_bins[i+1:])
        else:
            cell_number += X_bin_idx
        
        # Correction. If obj out of bound, assign -1.
        ibool = np.logical_or.reduce(((X_bin_idx < 0), (X_bin_idx >= num_bins[i]), ibool))
        
    cell_number[ibool] = -1
    
    return cell_number        


def gen_bin_idx(X, Xmin, dX):
    """
    Given a linear array of numbers and minimum, 
    compute bin index corresponding to each sample.
    
    dX is spacing between the bins.
    """
    
    return np.floor((X-Xmin)/float(dX)).astype(int)