# Used model 2 parameterization. See below.

import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys
import matplotlib.pyplot as plt

import extreme_deconvolution as XD
from model_class import *


print "# ----- Error model functions ----- # "
def gen_flux_noise(Nsample, flim, sn=5):
    """
    Given the limiting flux and signal to noise, generate Nsample noise sample.
    """
    sig = flim/float(sn)
    return np.random.normal(0, sig, Nsample).T

def MC_flux_oii_error_conv(ArcSinh_zg, ArcSinh_rg, gflux, ArcSinh_oiig, gf_lim=mag2flux(24.7), rf_lim=mag2flux(24.3), zf_lim=mag2flux(23.5)):
    """
    Given the sample from the intrinsic distribution, compute grz-flux and add
    the noise, and re-compute the observed distribution sample. Also, return
    convariance matrix for each data point.
    
    Default values based on the estimated noise limits in DEEP2-DR5
    """
    Nsample = gflux.size
    
    # Invert to get ratio and multiply to get flux
    zflux = np.sinh(ArcSinh_zg) * gflux
    rflux = np.sinh(ArcSinh_rg) * gflux
    oii = np.sinh(ArcSinh_oiig) * gflux
        
    # Compute covariance matrix, given the original values and the characteristic noise size.
    Covar = gen_flux_oii_covar(gflux, rflux, zflux, oii, gf_lim, rf_lim, zf_lim)
    
    
    # Generate error
    gerr = gen_flux_noise(Nsample, gf_lim)
    rerr = gen_flux_noise(Nsample, rf_lim)
    zerr = gen_flux_noise(Nsample, zf_lim)
#     oii_err = gen_oii_noise(Nsample, sig) # Yet to be written
    
    # Add error
    gflux = gflux + gerr
    rflux = rflux + rerr
    zflux = zflux + zerr
#     oii = oii + oii_err
    
    # Compute the original variables
    ArcSinh_zg_conv = np.arcsinh(zflux/gflux)
    ArcSinh_rg_conv = np.arcsinh(rflux/gflux)
    ArcSinh_oiig_conv = np.arcsinh(oii/gflux)    
    
    return ArcSinh_zg_conv, ArcSinh_rg_conv, gflux, ArcSinh_oiig_conv, Covar
    
    
    
    
def gen_flux_oii_covar(gflux, rflux, zflux, oii, gf_lim, rf_lim, zf_lim, sn=5, ND=5):
    """
    Note that in the original parameterization of grz-flux, the covariance matrix is diagonal,
    and constant. However, in any other parameterization, the covariance matrix will vary
    as a function of the variables.

    The desired covariance matrix in the new parameterization can be obatined C_y = M C_x M^T,
    where C_x is the original covariance and M^T = d(y1, y2, y3)/d(x1, x2, x3).
    
    Here, specific parameterization arcsinh(z/g), arcsinh(r/g), g, arcsinh(oii/g) is considered.
    
    For now we assume oii has no noise.
    
    C_x = Diag([dz^2, dr^2, dg^z, 0])
    
    M given as in the expression below.
    
    ND is the dimension of Covar matrices and has to be at least 4. 
    """
    Nsample = gflux.size
    Covar = np.zeros((Nsample, ND, ND))
    
    sn = float(sn)
    Cx = np.diag([(zf_lim/sn)**2, (rf_lim/sn)**2, (gf_lim/sn)**2, 0])
    
    for i in range(Nsample):
        g, r, z, o = gflux[i], rflux[i], zflux[i], oii[i]
        M00, M01, M02, M03 = 1/np.sqrt(g**2+z**2), 0, -z/(g*np.sqrt(g**2+z**2)), 0
        M10, M11, M12, M13 = 0, 1/np.sqrt(g**2+r**2), -r/(g*np.sqrt(g**2+r**2)), 0
        M20, M21, M22, M23 = 0, 0, 1, 0
        M30, M31, M32, M33 = 0, 0, -o/(g*np.sqrt(g**2+o**2)), 1/np.sqrt(g**2+o**2)
        
        M = np.array([[M00, M01, M02, M03],
                            [M10, M11, M12, M13],
                            [M20, M21, M22, M23],
                            [M30, M31, M32, M33]])
        
        Covar[i][:4,:4] = np.dot(np.dot(M, Cx), M.T)
    
    return Covar

    


print "# ----- Plot features ----- #"
mag_min = 21.5

# var limits
lim_y = [-.25, 2.2] # rf/gf
lim_x = [-.5, 4.5] # zf/gf
lim_z = [-.25, 4.] # gf
lim_oii = [0, 5] # oii/gf
lim_redz = [0.5, 1.7]

# bin widths
dx = 0.05
dy = 0.025 
doii = 0.05
dz = 2.5e-2
dred_z = 0.025


# var names
var_y_name = r"$sinh^{-1} (f_r/f_g)$"  
var_x_name = r"$sinh^{-1} (f_z/f_g)$"
var_z_name = r"$f_g$"
oii_name =  r"$sinh^{-1} (OII/f_g)$"
red_z_name = r"$\eta$"

# var lines
var_x_lines = []# np.asarray([1/2.5**2, 1/2.5, 1., 2.5, 2.5**2])
var_y_lines = []#[1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
redz_lines = [0.6, 1.1, 1.6] # Redz
var_z_lines = [mag2flux(f) for f in [21, 22, 23, 24, 24.25, 25.]]
oii_lines = []

# Plot variables
lims = [lim_x, lim_y, lim_z, lim_oii, lim_redz]
binws = [dx, dy, dz, doii, dred_z]
var_names = [var_x_name, var_y_name, var_z_name, oii_name, red_z_name]
lines = [var_x_lines, var_y_lines, var_z_lines, oii_lines, redz_lines]



print "# ----- Get data means and covariance ----- #"
# data_storage = model2(0)

# # Note OII here is actually OII/gflux
# var_x, var_y, var_z, OII, redz, w, iELG =\
#     data_storage.var_x, data_storage.var_y, data_storage.var_z, data_storage.oii, data_storage.red_z, data_storage.w, data_storage.iELG
# data = np.asarray([var_x[iELG], var_y[iELG], var_z[iELG], OII[iELG], redz[iELG]])
# weight = w[iELG]

# data_means = np.average(data, axis=1, weights=weight)
# data_cov = np.cov(data, aweights=weight)



print "# ----- Generate mock data ----- #"
Nsample = 5000
mock_data_MoG1 = MoG1(Nsample)
mock_data_MoG2 = MoG2(Nsample)


num_vars = 5
num_cat = 1
mean1, cov1 = MoG1_mean_cov()
mean2, cov2 = MoG2_mean_cov()

MoG_amps = [np.array([1.]), np.array([0.5, 0.5])]
MoG_means = [np.array([mean1]), np.array([mean1, mean2])]
MoG_covs = [np.array([cov1]), np.array([cov1, cov2])]

ND = 5 # Dimension of MoG to plot



print "# ----- Generate error convolved distribution ----- #"
# Error convolution to first
ArcSinh_zg, ArcSinh_rg, gflux, ArcSinh_oiig, redz = mock_data_MoG1
ArcSinh_zg_conv, ArcSinh_rg_conv, gflux, ArcSinh_oiig_conv, Covar1 =\
	MC_flux_oii_error_conv(ArcSinh_zg, ArcSinh_rg, gflux, ArcSinh_oiig)
mock_data_MoG1_err_conv = [ArcSinh_zg_conv, ArcSinh_rg_conv, gflux, ArcSinh_oiig_conv, redz]


# Error convolution to second
ArcSinh_zg, ArcSinh_rg, gflux, ArcSinh_oiig, redz = mock_data_MoG2
ArcSinh_zg_conv, ArcSinh_rg_conv, gflux, ArcSinh_oiig_conv, Covar2 =\
    MC_flux_oii_error_conv(ArcSinh_zg, ArcSinh_rg, gflux, ArcSinh_oiig)
mock_data_MoG2_err_conv = [ArcSinh_zg_conv, ArcSinh_rg_conv, gflux, ArcSinh_oiig_conv, redz]





print "# ----- Plot the original samples with true models ----- #"
for i, md in enumerate([mock_data_MoG1, mock_data_MoG2]):
    if i == -1:
        pass
    else:
        variables = [md]
        weights = None
        amps1 = MoG_amps[i]
        means1 = MoG_means[i]
        covs1 = MoG_covs[i]
        fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
        # Corr plots without annotation
#         ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines, category_names=["ELG"], pt_sizes=[2.5], colors=None, ft_size_legend = 15, lw_dot=2)
        # Add ellipses
        ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
                                  var_names, weights, lines=lines, category_names=["ELG"],\
                                  pt_sizes=[3.5], colors=None, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
                                plot_MoG1=True, amps1=amps1, means1=means1, covs1=covs1, ND1=ND, color_MoG1="blue")

        plt.tight_layout()
        plt.savefig("MoG%d-mock-data-N%d.png" % (i+1, Nsample), dpi=200, bbox_inches="tight")
        # plt.show()
        plt.close()     



print "# ----- Fit MoGs to the mock data and error-convolved mock data ----- #"
# Fit MoGs to the mock data.
suffixes = ["mock1", "mock2"]
for i, md in enumerate([mock_data_MoG1, mock_data_MoG2]):
    print suffixes[i]
    Ydata = np.asarray(md).T
    Ycovar = gen_diag_data_covar(Ydata.shape[0], var=[0, 0, 0, 0, 0])
    MODELS = fit_GMM(Ydata, Ycovar, 5, 5, NK_list=[1, 2, 3], Niter=5, fname_suffix=suffixes[i], MaxDIM=True)

# Fit the error convolved data points
suffixes = ["mock1-err-conv", "mock2-err-conv"]
Covars = [Covar1, Covar2]
for i, md in enumerate([mock_data_MoG1_err_conv, mock_data_MoG2_err_conv]):
    print suffixes[i]
    Ydata = np.asarray(md).T
    Ycovar = Covars[i]
    MODELS = fit_GMM(Ydata, Ycovar, 5, 5, NK_list=[1, 2, 3], Niter=5, fname_suffix=suffixes[i], MaxDIM=True)





print "# ----- Plot the fits ----- #"
# Plot only the best fits!
num_vars = 5
num_cat = 1
ND = 5 # Dimension of MoG to plot
mean1, cov1 = MoG1_mean_cov()
mean2, cov2 = MoG2_mean_cov()

MoG_amps = [np.array([1.]), np.array([0.5, 0.5])]
MoG_means = [np.array([mean1]), np.array([mean1, mean2])]
MoG_covs = [np.array([cov1]), np.array([cov1, cov2])]


# To be used for saving files.
var_names_txt = ["zg", "rg", "fg", "oiig", "redz"]

# Plot of all the samples
for i, md in enumerate([mock_data_MoG1, mock_data_MoG2]):
    if i == -1:
        pass
    else:
        # Truth
        variables = [md]
        weights = None
        amps1 = MoG_amps[i]
        means1 = MoG_means[i]
        covs1 = MoG_covs[i]
        
        MODELS = np.load("MODELS-mock%d.npy" % (i+1)).item()
        
        # Plotting the fits
        for j, var_num_tuple in enumerate(MODELS.keys()): # For each selection of variables
            if len(var_num_tuple) < 5: # Only plot the last models.
                pass
            else:
                # Generate the name tag. Sequence of variables used to model.
                vars_str = []
                for vn in var_num_tuple:
                    vars_str.append(var_names_txt[vn])

                # Models corresponding to the tuples
                ms = MODELS[var_num_tuple]
                for K in ms.keys(): # For each component number tried
                    vars_str_tmp = list(np.copy(vars_str))
                    vars_str_tmp.append("K"+str(K))
                    vars_str_tmp = "-".join(vars_str_tmp)
                    print vars_str_tmp
                    
                    # Fits
                    m = ms[K]
                    amps_fit  = m["amps"]
                    means_fit  = m["means"]
                    covs_fit = m["covs"]        
        
                    fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
                    # Corr plots without annotation
                    ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
                                              var_names, weights, lines=lines, category_names=["ELG"],\
                                              pt_sizes=[3.5], colors=None, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
                                            plot_MoG1=True, amps1=amps1, means1=means1, covs1=covs1, ND1=ND, color_MoG1="blue",\
                                              plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit,\
                                              means_general=means_fit, covs_general=covs_fit, color_general="red")

                    plt.tight_layout()
                    plt.savefig("MoG%d-mock-data-N%d-fit-%s.png" % (i+1, Nsample, vars_str_tmp), dpi=200, bbox_inches="tight")
                    # plt.show()
                    plt.close()    



# Plot only the best fits!

# True parameters
num_vars = 5
num_cat = 1
ND = 5 # Dimension of MoG to plot
mean1, cov1 = MoG1_mean_cov()
mean2, cov2 = MoG2_mean_cov()

MoG_amps = [np.array([1.]), np.array([0.5, 0.5])]
MoG_means = [np.array([mean1]), np.array([mean1, mean2])]
MoG_covs = [np.array([cov1]), np.array([cov1, cov2])]


# To be used for saving files.
var_names_txt = ["zg", "rg", "fg", "oiig", "redz"]

# Plot of all the samples
for i, md in enumerate([mock_data_MoG1_err_conv, mock_data_MoG2_err_conv]):
    if i == -1:
        pass
    else:
        # Truth
        variables = [md]
        weights = None
        amps1 = MoG_amps[i]
        means1 = MoG_means[i]
        covs1 = MoG_covs[i]
        
        MODELS = np.load("MODELS-mock%d-err-conv.npy" % (i+1)).item()
        
        # Plotting the fits
        for j, var_num_tuple in enumerate(MODELS.keys()): # For each selection of variables
            if len(var_num_tuple) < 5: # Only plot the last models.
                pass
            else:
                # Generate the name tag. Sequence of variables used to model.
                vars_str = []
                for vn in var_num_tuple:
                    vars_str.append(var_names_txt[vn])

                # Models corresponding to the tuples
                ms = MODELS[var_num_tuple]
                for K in ms.keys(): # For each component number tried
                    vars_str_tmp = list(np.copy(vars_str))
                    vars_str_tmp.append("K"+str(K))
                    vars_str_tmp.append("err-conv")
                    vars_str_tmp = "-".join(vars_str_tmp)
                    print vars_str_tmp
                    
                    # Fits
                    m = ms[K]
                    amps_fit  = m["amps"]
                    means_fit  = m["means"]
                    covs_fit = m["covs"]        
        
                    fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
                    # Corr plots without annotation
                    ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
                                              var_names, weights, lines=lines, category_names=["ELG"],\
                                              pt_sizes=[3.5], colors=None, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
                                            plot_MoG1=True, amps1=amps1, means1=means1, covs1=covs1, ND1=ND, color_MoG1="blue",\
                                              plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit,\
                                              means_general=means_fit, covs_general=covs_fit, color_general="red")

                    plt.tight_layout()
                    plt.savefig("MoG%d-mock-data-N%d-fit-%s.png" % (i+1, Nsample, vars_str_tmp), dpi=200, bbox_inches="tight")
                    # plt.show()
                    plt.close()    