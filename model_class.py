import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
import sys
import corner 

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
        self.mag_max = 24.25 # We model moderately deeper than 24. But only model down to 21.
        self.mag_min = 21.5
        self.category = ["NonELG", "NoZ", "ELG"]
        self.colors = ["black", "red", "blue"]

        # Model variables
        self.gflux, self.gf_err, self.rflux, self.rf_err, self.zflux, self.zf_err, self.rex_expr, self.rex_expr_ivar,\
        self.red_z, self.z_err, self.oii, self.oii_err, self.w, self.field, self.iELG, self.iNoZ, self.iNonELG, self.objtype\
        = self.import_data_DEEP2_full()

        # Extended vs non-extended
        self.ipsf = (self.objtype=="PSF")
        self.iext = (~self.ipsf)

        self.var_x = self.rflux
        self.var_y = self.zflux
        self.var_z = self.gflux 

        # Plot variables
        # var limits
        # self.lim_exp_r = [-.05, 1.05]
        self.lim_redz = [0.5, 1.7]
        self.lim_oii = [0, 50]
        self.lim_x = [-.25, mag2flux(self.mag_min-1)] # r
        self.lim_y = [-.75, mag2flux(self.mag_min-1)] # z
        self.lim_z = [-.25, mag2flux(self.mag_min)] # g    

        # bin widths
        self.dz = 2.5e-2
        self.dx = self.dy = self.dz*2        
        self.dred_z = 0.025
        self.doii = 0.5
        # self.dr = 0.01

        # var names
        self.var_x_name = r"$f_r$"
        self.var_y_name = r"$f_z$"
        self.var_z_name = r"$f_g$"
        self.red_z_name = r"$\eta$"
        self.oii_name =  r"$OII$"
        # self.r_exp_name = r"$r_{exp}$"

        # var lines
        self.var_x_lines = [mag2flux(f) for f in [21, 22, 23, 24, 24.25, 25.]]
        self.var_y_lines = [mag2flux(f) for f in [21, 22, 23, 24, 24.25, 25.]]
        self.var_z_lines = [mag2flux(f) for f in [21, 22, 23, 24, 24.25, 25.]]
        self.redz_lines = [0.6, 1.1, 1.6] # Redz
        self.oii_lines = [8]
        # self.exp_r_lines = [0.25, 0.5, 0.75, 1.0]

        # Trainig idices and area
        self.sub_sample_num = sub_sample_num # Determine which sub sample to use
        self.iTrain, self.area_train = self.gen_train_set_idx()


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


        print "Corr plot - var_xyz, red_z, oii - ELG only"
        num_cat = 1
        num_vars = 5
        lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_oii, self.lim_redz]
        binws = [self.dx, self.dy, self.dz, self.doii, self.dred_z]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.oii_name, self.red_z_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.oii_lines, self.redz_lines]

        if plot_rex:
            for e in [(self.ipsf, "PSF"), (self.iext, "EXT")]:
                iselect, tag = e
                print tag
                variables = []
                weights = []    
                iplot = np.copy(self.iELG) & iselect & self.iTrain
                i = 2 # For category
                variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.oii[iplot], self.red_z[iplot]]]
                weights = [self.w[iplot]/self.area_train]

                fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(50, 50))
                ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=None, colors=None, ft_size_legend = 15, lw_dot=2)

                plt.savefig("%s-%s-data-ELG-%s-redz-oii.png" % (model_tag, cv_tag, tag), dpi=200, bbox_inches="tight")
                plt.close()            

        else:
            iplot = np.copy(self.iELG) & self.iTrain
            i = 2 # For category
            variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.oii[iplot], self.red_z[iplot]]]
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
        self.var_y, self.var_x, self.var_z, self.oii = self.var_reparam() 

        # Plot variables
        # var limits
        self.lim_y = [-.25, 7.5]# rf/gf
        self.lim_x = [-1, 14.] # zf/gf
        self.lim_oii =[0, 100] # oii/gflux
        # self.lim_z = [-.25, mag2flux(self.mag_min)] # gf

        # bin widths
        self.dy = 0.1
        self.dx = 0.1 # zf/gf
        self.doii = 1
        # self.dz # from the parent

        # var names
        self.var_y_name = r"$f_r/f_g$"
        self.var_x_name = r"$f_z/f_g$"
        self.var_z_name = r"$f_g$"
        self.oii_name =  r"$OII/f_g$"
        self.red_z_name = r"$\eta$"
        # self.r_exp_name = r"$r_{exp}$"

        # var lines
        self.var_y_lines = [1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        self.var_x_lines = [1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        # self.var_z_lines = [mag2flux(f) for f in [21, 22, 23, 24, 24.25]]
        self.redz_lines = [0.6, 1.1, 1.6] # Redz
        self.oii_lines = []
        # self.exp_r_lines = [0.25, 0.5, 0.75, 1.0]

    def var_reparam(self):
        return self.rflux/self.gflux, self.zflux/self.gflux, self.gflux, self.oii/self.gflux



class model2(parent_model):
    """
    parametrization: arcsinh zflux/gflux, arcsinh rflux/gflux, gflux, oii/gflux, redz
    """
    def __init__(self, sub_sample_num):
        parent_model.__init__(self, sub_sample_num)
        # Re-parametrizing variables
        self.var_y, self.var_x, self.var_z, self.oii = self.var_reparam() 

        # Plot variables
        # var limits
        self.lim_y = [-.1, 2.2] # rf/gf
        self.lim_x = [-.75, 4.5] # zf/gf
        self.lim_oii = [0, 100] # oii/gf
        # self.lim_z = [-.25, mag2flux(self.mag_min)] # gf

        # bin widths
        self.dy = 0.025
        self.dx = 0.05 
        self.doii = 1
        # self.dz = 2.5e-2

        # var names
        self.var_y_name = r"$sinh^{-1} (f_r/f_g/2)$"  
        self.var_x_name = r"$sinh^{-1} (f_z/f_g/2)$"
        self.var_z_name = r"$f_g$"
        self.oii_name =  r"$OII/f_g$"
        self.red_z_name = r"$\eta$"
        # self.r_exp_name = r"$r_{exp}$"

        # var lines
        self.var_y_lines = []# np.asarray([1/2.5**2, 1/2.5, 1., 2.5, 2.5**2])
        self.var_x_lines = []#[1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        self.redz_lines = [0.6, 1.1, 1.6] # Redz
        self.oii_lines = []
        # self.var_z_lines = [mag2flux(f) for f in [21, 22, 23, 24, 25]]
        # self.exp_r_lines = [0.25, 0.5, 0.75, 1.0]

    def var_reparam(self):
        return np.arcsinh(self.rflux/self.gflux/2.), np.arcsinh(self.zflux/self.gflux/2.), self.gflux, self.oii/self.gflux
