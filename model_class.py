import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys
import os.path

import numpy as np
import matplotlib.pyplot as plt

def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)


def load_tractor_DR5_matched_to_DEEP2_full(ibool=None):
    """
    Load select columns. From all fields.
    """
    tbl1 = load_fits_table("DR5-matched-to-DEEP2-f2-glim25.fits")
    tbl2 = load_fits_table("DR5-matched-to-DEEP2-f3-glim25.fits")    
    tbl3 = load_fits_table("DR5-matched-to-DEEP2-f4-glim25.fits")

    tbl1_size = tbl1.size
    tbl2_size = tbl2.size
    tbl3_size = tbl3.size    
    field = np.ones(tbl1_size+tbl2_size+tbl3_size, dtype=int)
    field[:tbl1_size] = 2 # Ad hoc solution    
    field[tbl1_size:tbl1_size+tbl2_size] = 3 # Ad hoc solution
    field[tbl1_size+tbl2_size:] = 4 
    tbl = np.hstack([tbl1, tbl2, tbl3])
    if ibool is not None:
        tbl = tbl[ibool]
        field = field[ibool]

    ra, dec = load_radec(tbl)
    bid = tbl["brickid"]
    bp = tbl["brick_primary"]
    r_dev, r_exp = tbl["shapedev_r"], tbl["shapeexp_r"]
    gflux_raw, rflux_raw, zflux_raw = tbl["flux_g"], tbl["flux_r"], tbl["flux_z"]
    gflux, rflux, zflux = gflux_raw/tbl["mw_transmission_g"], rflux_raw/tbl["mw_transmission_r"],zflux_raw/tbl["mw_transmission_z"]
    mw_g, mw_r, mw_z = tbl["mw_transmission_g"], tbl["mw_transmission_r"], tbl["mw_transmission_z"]    
    givar, rivar, zivar = tbl["flux_ivar_g"], tbl["flux_ivar_r"], tbl["flux_ivar_z"]
    g_allmask, r_allmask, z_allmask = tbl["allmask_g"], tbl["allmask_r"], tbl["allmask_z"]
    objtype = tbl["type"]
    tycho = tbl["TYCHOVETO"]
    B, R, I = tbl["BESTB"], tbl["BESTR"], tbl["BESTI"]
    cn = tbl["cn"]
    w = tbl["TARG_WEIGHT"]
    # Proper weights for NonELG and color selected but unobserved classes. 
    w[cn==6] = 1
    w[cn==8] = 0    
    red_z, z_err, z_quality = tbl["RED_Z"], tbl["Z_ERR"], tbl["ZQUALITY"]
    oii, oii_err = tbl["OII_3727"]*1e17, tbl["OII_3727_ERR"]*1e17
    D2matched = tbl["DEEP2_matched"]
    BRI_cut = tbl["BRI_cut"].astype(int).astype(bool)
    rex_expr, rex_expr_ivar = tbl["rex_shapeExp_r"], tbl["rex_shapeExp_r_ivar"]
    
    return bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn,\
        w, red_z, z_err, z_quality, oii, oii_err, D2matched, rex_expr, rex_expr_ivar, field




class parent_model:
    def __init__(self, sub_sample_num):
        # Basic class variables
        self.areas = np.load("spec-area.npy")
        self.mag_max = 24 # We model moderately deeper than 24. But only model down to 21.
        self.mag_min = 22
        self.category = ["NonELG", "NoZ", "ELG"]
        self.colors = ["black", "red", "blue"]

        # Model variables
        self.gflux, self.gf_err, self.rflux, self.rf_err, self.zflux, self.zf_err, self.rex_expr, self.rex_expr_ivar,\
        self.red_z, self.z_err, self.oii, self.oii_err, self.w, self.field, self.iELG, self.iNoZ, self.iNonELG, self.objtype\
        = self.import_data_DEEP2_full()

        # Extended vs non-extended
        self.ipsf = (self.objtype=="PSF")
        self.iext = (~self.ipsf)

        self.var_x = self.zflux
        self.var_y = self.rflux
        self.var_z = self.gflux
        self.var_w = self.oii 

        # Plot variables
        # var limits
        # self.lim_exp_r = [-.05, 1.05]
        self.lim_redz = [0.5, 1.7]
        self.lim_w = [0, 50]
        self.lim_x = [-.75, mag2flux(self.mag_min-1)] # z
        self.lim_y = [-.25, mag2flux(self.mag_min-1)] # r
        self.lim_z = [-.25, mag2flux(self.mag_min)] # g    

        # bin widths
        self.dz = 2.5e-2 # g
        self.dx = self.dy = self.dz*2 # z, r 
        self.dred_z = 0.025
        self.dw = 0.5
        # self.dr = 0.01

        # var names
        self.var_x_name = r"$f_z$"
        self.var_y_name = r"$f_r$"
        self.var_z_name = r"$f_g$"
        self.var_w_name =  r"$OII$"        
        self.red_z_name = r"$\eta$"
        # self.r_exp_name = r"$r_{exp}$"

        # var lines
        self.var_x_lines = [mag2flux(f) for f in [21, 22, 23, 24, 24.25, 25.]]
        self.var_y_lines = [mag2flux(f) for f in [21, 22, 23, 24, 24.25, 25.]]
        self.var_z_lines = [mag2flux(f) for f in [21, 22, 23, 24, 24.25, 25.]]
        self.var_w_lines = [8]        
        self.redz_lines = [0.6, 1.1, 1.6] # Redz
        # self.exp_r_lines = [0.25, 0.5, 0.75, 1.0]

        # Trainig idices and area
        self.sub_sample_num = sub_sample_num # Determine which sub sample to use
        self.iTrain, self.area_train = self.gen_train_set_idx()

        # MODELS: GMM fit to the data
        self.MODELS = [None, None, None]


    def gen_train_set_idx(self):
        """
        Generate sub sample set indices and corresponding area.
        Note: Data only from Field 3 and 4 are considered
        """
        area_F34 = self.areas[1]+self.areas[2]

        # 0: Full F34 data
        if self.sub_sample_num == 0:
            iTrain = self.field !=2
            area_train = area_F34
        # 1: F3 data only
        if self.sub_sample_num == 1:
            iTrain = self.field == 3
            area_train = self.areas[1]
        # 2: F4 data only
        if self.sub_sample_num == 2:
            iTrain = self.field == 4
            area_train = self.areas[2]
        # 3-7: CV1-CV5: Sub-sample F34 into five-fold CV sets.
        if self.sub_sample_num in [3, 4, 5, 6, 7]:
            iTrain, area_train = self.gen_train_set_idx_cv()
        # 8-10: Magnitude changes. For power law use full data. 
        # g in [22.5, 23.5], [22.75, 23.75], [23, 24]. 
        if self.sub_sample_num == 8:
            iTrain = (self.gflux > mag2flux(23.5)) & (self.gflux < mag2flux(22.5))  & (self.field!=2)
            area_train = area_F34
        if self.sub_sample_num == 9:
            iTrain = (self.gflux > mag2flux(23.75)) & (self.gflux < mag2flux(22.75)) & (self.field!=2)
            area_train = area_F34
        if self.sub_sample_num == 10:
            iTrain = (self.gflux > mag2flux(24.)) & (self.gflux < mag2flux(23.)) & (self.field!=2)
            area_train = area_F34                        

        return iTrain, area_train



    def fit_MoG(self, NK_list, model_tag="", cv_tag="", cache=False):
        """
        Fit MoGs to data. Note that here we only consider fitting to 3 or 5 dimensions.

        If cache = True, then search to see if there are models already fit and if available use them.
        """
        cache_success = False
        if cache:
            for i in range(3):
                model_fname = "./MODELS-%s-%s-%s.npy" % (self.category[i], model_tag, cv_tag)
                if os.path.isfile(model_fname):
                    self.MODELS[i] = np.load(model_fname).item()
                    cache_success = True
                    print "Cached result will be used for MODELS-%s-%s-%s." % (self.category[i], model_tag, cv_tag)

        if not cache_success: # If cached result was not requested or was searched for but not found.
            # Dimension of model
            ND = 3
            # Number of variables up to which MoG is being proposed
            ND_fit = 3
            for i, ibool in enumerate([self.iNonELG, self.iNoZ]):
                print "Fitting MoGs to %s" % self.category[i]
                ifit = ibool & self.iTrain
                Ydata = np.array([self.var_x[ifit], self.var_y[ifit], self.var_z[ifit]]).T
                Ycovar = self.gen_covar(ifit, ND=3)
                weight = self.w[ifit]
                self.MODELS[i] = fit_GMM(Ydata, Ycovar, ND, ND_fit, NK_list=NK_list, Niter=5, fname_suffix="%s-%s-%s" % (self.category[i], model_tag, cv_tag), MaxDIM=True, weight=weight)

            i = 2
            # Dimension of model
            ND = 5
            # Number of variables up to which MoG is being proposed
            ND_fit = 5
            print "Fitting MoGs to %s" % self.category[i]        
            ifit = self.iELG & self.iTrain
            Ydata = np.array([self.var_x[ifit], self.var_y[ifit], self.var_z[ifit], self.var_w[ifit], self.red_z[ifit]]).T
            Ycovar = self.gen_covar(ifit, ND=5)
            weight = self.w[ifit]
            self.MODELS[i] = fit_GMM(Ydata, Ycovar, ND, ND_fit, NK_list=NK_list, Niter=5, fname_suffix="%s-%s-%s" % (self.category[i], model_tag, cv_tag), MaxDIM=True, weight=weight)

        return




    def visualize_fit(self, model_tag="", cv_tag="", cum_contour=False):
        """
        Make corr plots of the various classes with fits overlayed.
        If cum_contour is True, then instead of plotting individual component gaussians,
        plt the cumulative gaussian fit.
        """

        print "Corr plot - var_xyz - Separately"
        num_cat = 1
        num_vars = 3

        lims = [self.lim_x, self.lim_y, self.lim_z]
        binws = [self.dx, self.dy, self.dz]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines]

        for i, ibool in enumerate([self.iNonELG, self.iNoZ]):
            print "Plotting %s" % self.category[i]

            # Take the real data points.
            variables = []
            weights = []                
            iplot = np.copy(ibool) & self.iTrain
            variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot]])
            weights.append(self.w[iplot]/self.area_train)

            MODELS = self.MODELS[i] # Take the model for the category.
            # Plotting the fits
            for j, var_num_tuple in enumerate(MODELS.keys()): # For each selection of variables
                if len(var_num_tuple) < 3: # Only plot the last models.
                    pass
                else:
                    # Models corresponding to the tuples
                    ms = MODELS[var_num_tuple]
                    for K in ms.keys(): # For each component number tried
                        print "K: %d" % K
                        # Fits
                        m = ms[K]
                        amps_fit  = m["amps"]
                        means_fit  = m["means"]
                        covs_fit = m["covs"]        
            
                        fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(25, 25))
                        # Corr plots without annotation
                        ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
                                                  var_names, weights, lines=lines, category_names=[self.category[i]],\
                                                  pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
                                                  plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit,\
                                                  means_general=means_fit, covs_general=covs_fit, color_general="red", cum_contour=cum_contour)
                        plt.tight_layout()
                        if cum_contour:
                            plt.savefig("%s-%s-data-%s-fit-K%d-cum-contour.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
                        else:
                            plt.savefig("%s-%s-data-%s-fit-K%d.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
                        # plt.show()
                        plt.close()



        print "Corr plot - var_xyz, var_w, red_z - ELG only"
        num_cat = 1
        num_vars = 5
        lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_w, self.lim_redz]
        binws = [self.dx, self.dy, self.dz, self.dw, self.dred_z]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.var_w_name, self.red_z_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.var_w_lines, self.redz_lines]

        iplot = np.copy(self.iELG) & self.iTrain
        i = 2 # For category
        variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.var_w[iplot], self.red_z[iplot]]]
        weights = [self.w[iplot]/self.area_train]

        MODELS = self.MODELS[i] # Take the model for the category.
        # Plotting the fits
        for j, var_num_tuple in enumerate(MODELS.keys()): # For each selection of variables
            if len(var_num_tuple) < 5: # Only plot the last models.
                pass
            else:
                # Models corresponding to the tuples
                ms = MODELS[var_num_tuple]
                for K in ms.keys(): # For each component number tried                        
                    print "K: %d" % K                
                    # Fits
                    m = ms[K]
                    amps_fit  = m["amps"]
                    means_fit  = m["means"]
                    covs_fit = m["covs"]        
        
                    fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
                    # Corr plots without annotation
                    ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
                                              var_names, weights, lines=lines, category_names=[self.category[i]],\
                                              pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
                                              plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit,\
                                              means_general=means_fit, covs_general=covs_fit, color_general="red", cum_contour=cum_contour)
                    plt.tight_layout()
                    if cum_contour:
                        plt.savefig("%s-%s-data-%s-fit-K%d-cum_contour.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
                    else:
                        plt.savefig("%s-%s-data-%s-fit-K%d.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
                    # plt.show()
                    plt.close()

        return



    def gen_covar(self, ifit, ND=5):
        """
        Covariance matrix in the original grz-oii-redz space is diagonal.
        """
        Nsample = np.sum(ifit)
        Covar = np.zeros((Nsample, ND, ND))        

        var_err_list = self.zf_err[ifit], self.rf_err[ifit], self.gf_err[ifit], self.oii_err[ifit], np.zeros(np.sum(ifit))

        for i in range(Nsample):
            tmp = []
            for j in range(ND):
                tmp.append(var_err_list[j][i]**2) # var = err^2
            Covar[i] = np.diag(tmp)
        
        return Covar




    def gen_train_set_idx_cv(self):
        """
        Given a cv number, return the relevant CV partition. 
        Below hacked code was used to free the partitioning. 
        Note that area was scaled according to the weight ratio.

        from model_class import *

        parent_instance = parent_model(0)
        field = parent_instance.field
        weight = parent_instance.w
        areas = parent_instance.areas

        nobjs_list=[]
        for i in [2, 3, 4]:
            nobjs_list.append((field == i).sum())

        # x is your dataset
        nobjs_F34 = nobjs_list[1]+nobjs_list[2]
        nobjs_F2 = nobjs_list[0]
        nobjs_total = nobjs_F2+nobjs_F34
        nobjs_per_cv = nobjs_F34/5
        nobjs_last_cv = nobjs_F34-nobjs_per_cv*4
        area_F34 = areas[1]+areas[2]
        weight_F34 = weight[field!=2]
        weight_F34_total = weight_F34.sum()

        indices = np.random.permutation(nobjs_F34)+nobjs_F2
        training_idx_list = []
        area_list = []

        # cv1
        ibool = np.zeros(nobjs_total, dtype=bool)
        ibool[indices[nobjs_per_cv:]]=True
        training_idx_list.append(ibool)
        area_list.append(area_F34 * weight[training_idx_list[0]].sum()/weight_F34_total)
        # cv234
        for i in range(1, 4, 1):
            ibool = np.zeros(nobjs_total, dtype=bool)    
            ibool[np.concatenate((indices[:nobjs_per_cv*i], indices[nobjs_per_cv*(i+1):]))]=True    
            training_idx_list.append(ibool) 
            area_list.append(area_F34 * weight[training_idx_list[i]].sum()/weight_F34_total)
        # cv5
        ibool = np.zeros(nobjs_total, dtype=bool)
        ibool[indices[:nobjs_per_cv*4]]=True
        training_idx_list.append(ibool)
        area_list.append(area_F34 * weight[training_idx_list[4]].sum()/weight_F34_total)

        training_idx_list = np.asarray(training_idx_list)
        area_list = np.asarray(area_list)

        np.save("cv_training_idx", training_idx_list)
        np.save("cv_area", area_list)
        """

        cv = self.sub_sample_num-3
        train_list = np.load("cv_training_idx.npy")
        area_list = np.load("cv_area.npy")
        return train_list[cv], area_list[cv]


    def plot_data(self, model_tag="", cv_tag="", plot_rex=False):
        """
        Use self model/plot variables to plot the data given an external figure ax_list.
        Save the resulting image using title_str (does not include extension)

        If plot_rex=False, then plot all data together. If True, plot according psf and non-psf type.
        """

        print "Corr plot - var_xyz - all classes together"
        lims = [self.lim_x, self.lim_y, self.lim_z]
        binws = [self.dx, self.dy, self.dz]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines]
        num_cat = 3
        num_vars = 3

        if plot_rex:
            for e in [(self.ipsf, "PSF"), (self.iext, "EXT")]:
                iselect, tag = e
                print tag
                variables = []
                weights = []                
                for ibool in [self.iNonELG, self.iNoZ, self.iELG]:
                    iplot = np.copy(ibool) & self.iTrain
                    iplot = iplot & iselect
                    variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot]])
                    weights.append(self.w[iplot]/self.area_train)

                fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
                ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines, category_names=self.category, pt_sizes=None, colors=self.colors, ft_size_legend = 15, lw_dot=2)
                plt.savefig("%s-%s-data-all-%s.png" % (model_tag, cv_tag, tag), dpi=200, bbox_inches="tight")
                plt.close()           
        else:
            variables = []
            weights = []            
            for ibool in [self.iNonELG, self.iNoZ, self.iELG]:
                iplot = np.copy(ibool) & self.iTrain
                variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot]])
                weights.append(self.w[iplot]/self.area_train)
            fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
            ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines, category_names=self.category, pt_sizes=None, colors=self.colors, ft_size_legend = 15, lw_dot=2)
            plt.savefig("%s-%s-data-all.png" % (model_tag, cv_tag), dpi=200, bbox_inches="tight")
            plt.close()        



        print "Corr plot - var_xyz - separately"
        lims = [self.lim_x, self.lim_y, self.lim_z]
        binws = [self.dx, self.dy, self.dz]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines]
        num_cat = 1
        num_vars = 3

        if plot_rex:
            for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
                print "Plotting %s" % self.category[i]                                
                for e in [(self.ipsf, "PSF"), (self.iext, "EXT")]:
                    iselect, tag = e
                    print tag
                    variables = []
                    weights = []                
                    fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))                
                    iplot = np.copy(ibool) & self.iTrain
                    iplot = iplot & iselect
                    variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot]])
                    weights.append(self.w[iplot]/self.area_train)

                    ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines, category_names=[self.category[i]], pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2)
                    plt.savefig("%s-%s-data-%s-%s.png" % (model_tag, cv_tag, self.category[i], tag), dpi=200, bbox_inches="tight")
                    plt.close() 
        else:
            for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
                print "Plotting %s" % self.category[i]                
                variables = []
                weights = []                
                iplot = np.copy(ibool) & self.iTrain
                variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot]])
                weights.append(self.w[iplot]/self.area_train)

                fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
                ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2)

                plt.savefig("%s-%s-data-%s.png" % (model_tag, cv_tag, self.category[i]), dpi=200, bbox_inches="tight")
                plt.close()


        print "Corr plot - var_xyz, var_w, red_z - ELG only"
        num_cat = 1
        num_vars = 5
        lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_w, self.lim_redz]
        binws = [self.dx, self.dy, self.dz, self.dw, self.dred_z]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.var_w_name, self.red_z_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.var_w_lines, self.redz_lines]

        if plot_rex:
            for e in [(self.ipsf, "PSF"), (self.iext, "EXT")]:
                iselect, tag = e
                print tag
                variables = []
                weights = []    
                iplot = np.copy(self.iELG) & iselect & self.iTrain
                i = 2 # For category
                variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.var_w[iplot], self.red_z[iplot]]]
                weights = [self.w[iplot]/self.area_train]

                fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(50, 50))
                ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2)

                plt.savefig("%s-%s-data-ELG-%s-redz-oii.png" % (model_tag, cv_tag, tag), dpi=200, bbox_inches="tight")
                plt.close()            

        else:
            iplot = np.copy(self.iELG) & self.iTrain
            i = 2 # For category
            variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.var_w[iplot], self.red_z[iplot]]]
            weights = [self.w[iplot]/self.area_train]

            fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(50, 50))
            ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2)

            plt.savefig("%s-%s-data-ELG-redz-oii.png" % (model_tag, cv_tag), dpi=200, bbox_inches="tight")
            plt.close()



    def import_data_DEEP2_full(self):
        """Return DEEP2-DR5 data."""
        bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn, w, red_z, z_err, z_quality, oii, oii_err, D2matched, rex_expr, rex_expr_ivar, field\
            = load_tractor_DR5_matched_to_DEEP2_full()

        ifcut = (gflux > mag2flux(self.mag_max)) & (gflux < mag2flux(self.mag_min))
        ibool = (D2matched==1) & ifcut
        nobjs_cut = ifcut.sum()
        nobjs_matched = ibool.sum()

        print "Fraction of unmatched objects with g [%.1f, %.1f]: %.2f percent" % (self.mag_min, self.mag_max, 100 * (nobjs_cut-nobjs_matched)/float(nobjs_cut))
        print "We consider only the matched set. After fitting various densities, we scale the normalization by the amount we ignored in our fit due to unmatched set."

        bid, objtype, tycho, bp, ra, dec, gflux_raw, rflux_raw, zflux_raw, gflux, rflux, zflux, givar,\
        rivar, zivar, mw_g, mw_r, mw_z, r_dev, r_exp, g_allmask, r_allmask, z_allmask, B, R, I, BRI_cut, cn, w, red_z, z_err, z_quality, oii, oii_err, D2matched, rex_expr, rex_expr_ivar, field\
            = load_tractor_DR5_matched_to_DEEP2_full(ibool = ibool)

        # Define categories
        iELG, iNoZ, iNonELG = category_vector_generator(z_quality, z_err, oii, oii_err, BRI_cut, cn)

        # error
        gf_err = np.sqrt(1./givar)/mw_g
        rf_err = np.sqrt(1./rivar)/mw_r
        zf_err = np.sqrt(1./zivar)/mw_z

        return gflux, gf_err, rflux, rf_err, zflux, zf_err, rex_expr, rex_expr_ivar,\
            red_z, z_err, oii, oii_err, w, field, iELG, iNoZ, iNonELG, objtype




class model1(parent_model):
    """
    parametrization: rflux/gflux, zflux/gflux, gflux, oii/gflux, redz
    """
    def __init__(self, sub_sample_num):
        parent_model.__init__(self, sub_sample_num)

        # Re-parametrizing variables
        self.var_y, self.var_x, self.var_z, self.var_w = self.var_reparam() 

        # Plot variables
        # var limits
        self.lim_x = [-1, 14.] # zf/gf
        self.lim_y = [-.25, 7.5]# rf/gf        
        self.lim_w =[0, 100] # oii/gflux
        # self.lim_z = [-.25, mag2flux(self.mag_min)] # gf

        # bin widths
        self.dx = 0.1 # zf/gf
        self.dy = 0.1        
        self.dw = 1
        # self.dz # from the parent

        # var names
        self.var_y_name = r"$f_r/f_g$"
        self.var_x_name = r"$f_z/f_g$"
        self.var_z_name = r"$f_g$"
        self.var_w_name =  r"$OII/f_g$"
        self.red_z_name = r"$\eta$"
        # self.r_exp_name = r"$r_{exp}$"

        # var lines
        self.var_y_lines = [1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        self.var_x_lines = [1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        self.var_w_lines = []        
        # self.var_z_lines = [mag2flux(f) for f in [21, 22, 23, 24, 24.25]]
        self.redz_lines = [0.6, 1.1, 1.6] # Redz
        # self.exp_r_lines = [0.25, 0.5, 0.75, 1.0]

    def var_reparam(self):
        return self.rflux/self.gflux, self.zflux/self.gflux, self.gflux, self.oii/self.gflux

    def gen_covar(self, ifit, ND=5):
        """
        Covariance matrix corresponding to the new parametrization.
        - var_list: Contains a list of variables in the original space: zf, rf, zf, oii, redz
        - var_err_list: List of errors of the variables in the other list.

        Note: This function is incorret. Do not use.
        """
        Nsample = np.sum(ifit)
        Covar = np.zeros((Nsample, ND, ND))

        zflux, rflux, gflux, oii, red_z = self.zflux[ifit], self.rflux[ifit], self.gflux[ifit], self.oii[ifit], self.red_z[ifit]
        var_err_list = self.zf_err[ifit], self.rf_err[ifit], self.gf_err[ifit], self.oii_err[ifit], np.zeros(np.sum(Nsample))

        if ND == 3:
            for i in range(Nsample):
                # Construct the original space covariance matrix in 4 x 4 subspace.
                tmp = []
                for j in range(ND):
                    tmp.append(var_err_list[j][i]**2) # var = err^2
                Cx = np.diag(tmp)

                g, r, z, o = gflux[i], rflux[i], zflux[i], oii[i]
                M00, M01, M02 = 1/g, 0, -z/g**2
                M10, M11, M12 = 0, 1/g, -r/g**2
                M20, M21, M22 = 0, 0, 1
                
                M = np.array([[M00, M01, M02],
                            [M10, M11, M12],
                            [M20, M21, M22]])
                
                Covar[i] = np.dot(np.dot(M, Cx), M.T)
        elif ND == 5:
            for i in range(Nsample):
                # Construct the original space covariance matrix in 4 x 4 subspace.
                tmp = []
                for j in range(4):
                    tmp.append(var_err_list[j][i]**2) # var = err^2
                Cx = np.diag(tmp)

                g, r, z, o = gflux[i], rflux[i], zflux[i], oii[i]
                M00, M01, M02, M03 = 1/g, 0, -z/g**2, 0
                M10, M11, M12, M13 = 0, 1/g, -r/g**2, 0
                M20, M21, M22, M23 = 0, 0, 1, 0
                M30, M31, M32, M33 = 0, 0, -o/g**2, 1/g
                
                M = np.array([[M00, M01, M02, M03],
                                    [M10, M11, M12, M13],
                                    [M20, M21, M22, M23],
                                    [M30, M31, M32, M33]])
                
                Covar[i][:4,:4] = np.dot(np.dot(M, Cx), M.T)
        else: 
            print "The input number of variables need to be either 3 or 5."
            assert False

        return Covar        




class model2(parent_model):
    """
    parametrization: arcsinh zflux/gflux (x), arcsinh rflux/gflux (y), arcsinh oii/gflux (z), redz, gmag
    """
    def __init__(self, sub_sample_num):
        parent_model.__init__(self, sub_sample_num)

        # Re-parametrizing variables
        self.var_x, self.var_y, self.var_z, self.gmag = self.var_reparam() 

        # Plot variables
        # var limits
        self.lim_x = [-.75, 4.5] # zf/gf
        self.lim_y = [-.1, 2.2] # rf/gf        
        self.lim_z = [0, 5] # oii/gf
        self.lim_gmag = [22., 24.0]

        # bin widths
        self.dx = 0.05         
        self.dy = 0.025
        self.dz = 0.05
        self.dgmag = 0.025

        # var names
        self.var_x_name = r"$sinh^{-1} (f_z/f_g/2)$"        
        self.var_y_name = r"$sinh^{-1} (f_r/f_g/2)$"  
        self.var_z_name = r"$sinh^{-1} (OII/f_g/2)$"
        self.red_z_name = r"$\eta$"
        self.gmag_name  = r"$g$"

        # var lines
        self.var_x_lines = [1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        self.var_y_lines = [1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        self.var_z_lines = []
        self.redz_lines = [0.6, 1.1, 1.6] # Redz
        self.gmag_lines = [21, 22, 23, 24, 24.25]

        # Fit parameters for pow law
        self.MODELS_pow = [None, None, None]

    def var_reparam(self):
        return np.arcsinh(self.zflux/self.gflux/2.), np.arcsinh(self.rflux/self.gflux/2.), np.arcsinh(self.oii/self.gflux/2.), flux2mag(self.gflux)


    def plot_data(self, model_tag="", cv_tag=""):
        """
        Use self model/plot variables to plot the data given an external figure ax_list.
        Save the resulting image using title_str (does not include extension)
        """

        print "Corr plot - var_xyz - all classes together"
        lims = [self.lim_x, self.lim_y, self.lim_gmag]
        binws = [self.dx, self.dy, self.dgmag]
        var_names = [self.var_x_name, self.var_y_name, self.gmag_name]
        lines = [self.var_x_lines, self.var_y_lines, self.gmag_lines]
        num_cat = 3
        num_vars = 3

        variables = []
        weights = []            
        for ibool in [self.iNonELG, self.iNoZ, self.iELG]:
            iplot = np.copy(ibool) & self.iTrain
            variables.append([self.var_x[iplot], self.var_y[iplot], self.gmag[iplot]])
            weights.append(self.w[iplot]/self.area_train)
        fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
        ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines, category_names=self.category, pt_sizes=None, colors=self.colors, ft_size_legend = 15, lw_dot=2)
        plt.savefig("%s-%s-data-all.png" % (model_tag, cv_tag), dpi=200, bbox_inches="tight")
        plt.close()        



        print "Corr plot - var_xyz - separately"
        lims = [self.lim_x, self.lim_y, self.lim_gmag]
        binws = [self.dx, self.dy, self.dgmag]
        var_names = [self.var_x_name, self.var_y_name, self.gmag_name]
        lines = [self.var_x_lines, self.var_y_lines, self.gmag_lines]
        num_cat = 1
        num_vars = 3


        for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
            print "Plotting %s" % self.category[i]                
            variables = []
            weights = []                
            iplot = np.copy(ibool) & self.iTrain
            variables.append([self.var_x[iplot], self.var_y[iplot], self.gmag[iplot]])
            weights.append(self.w[iplot]/self.area_train)

            fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
            ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2)

            plt.savefig("%s-%s-data-%s.png" % (model_tag, cv_tag, self.category[i]), dpi=200, bbox_inches="tight")
            plt.close()


        print "Corr plot - var_xyz, red_z, gmag - ELG only"
        num_cat = 1
        num_vars = 5
        lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_redz, self.lim_gmag]
        binws = [self.dx, self.dy, self.dz, self.dred_z, self.dgmag]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.red_z_name, self.gmag_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.redz_lines, self.gmag_lines]

        iplot = np.copy(self.iELG) & self.iTrain
        i = 2 # For category
        variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.red_z[iplot], self.gmag[iplot]]]
        weights = [self.w[iplot]/self.area_train]

        fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(50, 50))
        ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2)

        plt.savefig("%s-%s-data-ELG-redz-oii.png" % (model_tag, cv_tag), dpi=200, bbox_inches="tight")
        plt.close()


    def fit_MoG(self, NK_list, model_tag="", cv_tag="", cache=False, Niter=5):
        """
        Fit MoGs to data. Note that here we only consider fitting to 2 or 4 dimensions.

        If cache = True, then search to see if there are models already fit and if available use them.
        """
        cache_success = False
        if cache:
            for i in range(3):
                model_fname = "./MODELS-%s-%s-%s.npy" % (self.category[i], model_tag, cv_tag)
                if os.path.isfile(model_fname):
                    self.MODELS[i] = np.load(model_fname).item()
                    cache_success = True
                    print "Cached result will be used for MODELS-%s-%s-%s." % (self.category[i], model_tag, cv_tag)

        if not cache_success: # If cached result was not requested or was searched for but not found.
            # For NonELG and NoZ
            ND = 2 # Dimension of model
            ND_fit = 2 # Number of variables up to which MoG is being proposed
            for i, ibool in enumerate([self.iNonELG, self.iNoZ]):
                print "Fitting MoGs to %s" % self.category[i]
                ifit = ibool & self.iTrain
                Ydata = np.array([self.var_x[ifit], self.var_y[ifit]]).T
                Ycovar = self.gen_covar(ifit, ND=ND)
                weight = self.w[ifit]
                self.MODELS[i] = fit_GMM(Ydata, Ycovar, ND, ND_fit, NK_list=NK_list, Niter=Niter, fname_suffix="%s-%s-%s" % (self.category[i], model_tag, cv_tag), MaxDIM=True, weight=weight)

            # For ELG
            i = 2
            ND = 4 # Dimension of model
            ND_fit = 4 # Number of variables up to which MoG is being proposed
            print "Fitting MoGs to %s" % self.category[i]
            ifit = self.iELG & self.iTrain
            Ydata = np.array([self.var_x[ifit], self.var_y[ifit], self.var_z[ifit], self.red_z[ifit]]).T
            Ycovar = self.gen_covar(ifit, ND=ND)
            weight = self.w[ifit]
            self.MODELS[i] = fit_GMM(Ydata, Ycovar, ND, ND_fit, NK_list=NK_list, Niter=Niter, fname_suffix="%s-%s-%s" % (self.category[i], model_tag, cv_tag), MaxDIM=True, weight=weight)

        return


    def fit_dNdm(self, model_tag="", cv_tag="", cache=False, Niter=5, bw=0.025):
        """
        Fit mag pow laws
        """
        cache_success = False
        if cache:
            for i in range(3):
                model_fname = "./MODELS-%s-%s-%s-pow.npy" % (self.category[i], model_tag, cv_tag)
                if os.path.isfile(model_fname):
                    self.MODELS_pow[i] = np.load(model_fname)
                    cache_success = True
                    print "Cached result will be used for MODELS-%s-%s-%s-pow." % (self.category[i], model_tag, cv_tag)
        if not cache_success:
            for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
                print "Fitting power law for %s" % self.category[i]
                ifit = self.iTrain & ibool
                mag = self.gmag[ifit]
                weight = self.w[ifit]
                self.MODELS_pow[i] = dNdm_fit(mag, weight, bw, self.mag_min, self.mag_max, self.area_train, niter = Niter)
                np.save("MODELS-%s-%s-%s-pow.npy" % (self.category[i], model_tag, cv_tag), self.MODELS_pow[i])

        return 






    def gen_covar(self, ifit, ND=4):
        """
        Covariance matrix corresponding to the new parametrization.
        Original parameterization: zf, rf, oii, redz, gf
        New parameterization is given by the model2.
        """
        Nsample = np.sum(ifit)
        Covar = np.zeros((Nsample, ND, ND))

        zflux, rflux, gflux, oii, red_z = self.zflux[ifit], self.rflux[ifit], self.gflux[ifit], self.oii[ifit], self.red_z[ifit]
        var_err_list = [self.zf_err[ifit], self.rf_err[ifit], self.oii_err[ifit], np.zeros(np.sum(Nsample)), self.gf_err[ifit]]


        for i in range(Nsample):
            # Construct the original space covariance matrix in 5 x 5 subspace.
            tmp = []
            for j in range(5):
                tmp.append(var_err_list[j][i]**2) # var = err^2
            Cx = np.diag(tmp)

            if ND == 2:
                g, r, z, o = gflux[i], rflux[i], zflux[i], oii[i]
                M00, M01, M02, M03, M04 = 1/np.sqrt(g**2+z**2), 0, 0, 0, -z/(g*np.sqrt(g**2+z**2))
                M10, M11, M12, M13, M14 = 0, 1/np.sqrt(g**2+r**2), 0, 0, -r/(g*np.sqrt(g**2+r**2))
                M = np.array([[M00, M01, M02, M03, M04],
                            [M10, M11, M12, M13, M14]])
                
                Covar[i] = np.dot(np.dot(M, Cx), M.T)
            elif ND == 4:
                # Construct the affine transformation matrix.
                g, r, z, o = gflux[i], rflux[i], zflux[i], oii[i]
                M00, M01, M02, M03, M04 = 1/np.sqrt(g**2+z**2), 0, 0, 0, -z/(g*np.sqrt(g**2+z**2))
                M10, M11, M12, M13, M14 = 0, 1/np.sqrt(g**2+r**2), 0, 0, -r/(g*np.sqrt(g**2+r**2))
                M20, M21, M22, M23, M24 = 0, 0, 1/np.sqrt(g**2+o**2), 0, -o/(g*np.sqrt(g**2+o**2))
                M30, M31, M32, M33, M34 = 0, 0, 0, 1, 0
                
                M = np.array([[M00, M01, M02, M03, M04],
                                    [M10, M11, M12, M13, M14],
                                    [M20, M21, M22, M23, M24],
                                    [M30, M31, M32, M33, M34]])
                Covar[i] = np.dot(np.dot(M, Cx), M.T)
            else: 
                print "The input number of variables need to be either 2 or 4."
                assert False

        return Covar



    def visualize_fit(self, model_tag="", cv_tag="", cum_contour=False):
        """
        Make corr plots of the various classes with fits overlayed.
        If cum_contour is True, then instead of plotting individual component gaussians,
        plot the cumulative gaussian fit. Also, plot the power law function corresponding
        to the magnitude dimension.
        """

        print "Corr plot - var_xyz - Separately"
        num_cat = 1
        num_vars = 3

        lims = [self.lim_x, self.lim_y, self.lim_gmag]
        binws = [self.dx, self.dy, self.dgmag]
        var_names = [self.var_x_name, self.var_y_name, self.gmag_name]
        lines = [self.var_x_lines, self.var_y_lines, self.gmag_lines]

        for i, ibool in enumerate([self.iNonELG, self.iNoZ]):
            print "Plotting %s" % self.category[i]

            # Take the real data points.
            variables = []
            weights = []                
            iplot = np.copy(ibool) & self.iTrain
            variables.append([self.var_x[iplot], self.var_y[iplot], self.gmag[iplot]])
            weights.append(self.w[iplot]/self.area_train)

            MODELS = self.MODELS[i] # Take the model for the category.
            MODELS_pow = self.MODELS_pow[i] # Power law for magnitude.

            # Plotting the fits
            for j, var_num_tuple in enumerate(MODELS.keys()): # For each selection of variables
                if len(var_num_tuple) < 2: # Only plot the last models.
                    pass
                else:
                    # Models corresponding to the tuples
                    ms = MODELS[var_num_tuple]
                    for K in ms.keys(): # For each component number tried
                        print "K: %d" % K
                        # Fits
                        m = ms[K]
                        amps_fit  = m["amps"]
                        means_fit  = m["means"]
                        covs_fit = m["covs"]        
            
                        fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(25, 25))
                        # Corr plots without annotation
                        ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
                                                  var_names, weights, lines=lines, category_names=[self.category[i]],\
                                                  pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
                                                  plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit,\
                                                  means_general=means_fit, covs_general=covs_fit, color_general="red", cum_contour=cum_contour,\
                                                  plot_pow=True, pow_model=MODELS_pow, pow_var_num=2)
                        plt.tight_layout()
                        if cum_contour:
                            plt.savefig("%s-%s-data-%s-fit-K%d-cum-contour.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
                        else:
                            plt.savefig("%s-%s-data-%s-fit-K%d.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
                        # plt.show()
                        plt.close()



        print "Corr plot - var_xyz, var_w, red_z - ELG only"
        num_cat = 1
        num_vars = 5
        lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_redz, self.lim_gmag]
        binws = [self.dx, self.dy, self.dz, self.dred_z, self.dgmag]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.red_z_name, self.gmag_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.redz_lines, self.gmag_lines]

        iplot = np.copy(self.iELG) & self.iTrain
        i = 2 # For category
        variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.red_z[iplot], self.gmag[iplot]]]
        weights = [self.w[iplot]/self.area_train]

        MODELS = self.MODELS[i] # Take the model for the category.
        MODELS_pow = self.MODELS_pow[i]

        # Plotting the fits
        for j, var_num_tuple in enumerate(MODELS.keys()): # For each selection of variables
            if len(var_num_tuple) < 4: # Only plot the last models.
                pass
            else:
                # Models corresponding to the tuples
                ms = MODELS[var_num_tuple]
                for K in ms.keys(): # For each component number tried                        
                    print "K: %d" % K                
                    # Fits
                    m = ms[K]
                    amps_fit  = m["amps"]
                    means_fit  = m["means"]
                    covs_fit = m["covs"]        
        
                    fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
                    # Corr plots without annotation
                    ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws,\
                                              var_names, weights, lines=lines, category_names=[self.category[i]],\
                                              pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2, hist_normed=True,\
                                              plot_MoG_general=True, var_num_tuple=var_num_tuple, amps_general=amps_fit,\
                                              means_general=means_fit, covs_general=covs_fit, color_general="red", cum_contour=cum_contour,\
                                              plot_pow=True, pow_model=MODELS_pow, pow_var_num=4)
                    plt.tight_layout()
                    if cum_contour:
                        plt.savefig("%s-%s-data-%s-fit-K%d-cum_contour.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
                    else:
                        plt.savefig("%s-%s-data-%s-fit-K%d.png" % (model_tag, cv_tag, self.category[i], K), dpi=200, bbox_inches="tight")
                    # plt.show()
                    plt.close()

        return

