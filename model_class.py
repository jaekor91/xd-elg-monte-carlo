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
    def __init__(self):
        # Basic class variables
        self.areas = np.load("spec-area.npy")
        self.mag_max = 24 # We only model between 24 and 21
        self.mag_min = 21
        self.category = ["NonELG", "NoZ", "ELG"]
        self.colors = ["black", "red", "blue"]

        # Model variables
        self.gflux, self.gf_err, self.rflux, self.rf_err, self.zflux, self.zf_err, self.rex_expr, self.rex_expr_ivar,\
        self.red_z, self.z_err, self.oii, self.oii_err, self.w, self.field, self.iELG, self.iNoZ, self.iNonELG, self.objtype\
        = self.import_data_DEEP2_full()

        # Extended vs non-extended
        self.ipsf = (self.objtype=="PSF")
        self.iext = (~self.ipsf)

        self.var_x = self.gflux
        self.var_y = self.rflux
        self.var_z = self.zflux

        # Plot variables
        # var limits
        self.lim_exp_r = [-.05, 1.05]
        self.lim_redz = [0.5, 1.7]
        self.lim_oii = [0, 25]
        self.lim_x = [-.25, mag2flux(22)] # g
        self.lim_y = [-.25, mag2flux(21.5)] # r
        self.lim_z = [-.25, mag2flux(21.)] # z    

        # bin widths
        self.dx = self.dy = (self.lim_x[1]-self.lim_x[0])/50.
        self.dz = 2*self.dx
        self.dr = 0.01
        self.dred_z = 0.025
        self.doii = 0.5

        # var names
        self.var_x_name = r"$f_g$"
        self.var_y_name = r"$f_r$"
        self.var_z_name = r"$f_z$"
        self.oii_name =  r"$OII$"
        self.r_exp_name = r"$r_{exp}$"
        self.red_z_name = r"$\eta$"

        # var lines
        self.var_x_lines = [mag2flux(f) for f in [21, 22, 23, 24, 25]]
        self.var_y_lines = [mag2flux(f) for f in [21, 22, 23, 24, 25]]
        self.var_z_lines = [mag2flux(f) for f in [21, 22, 23, 24, 25]]
        self.exp_r_lines = [0.25, 0.5, 0.75, 1.0]
        self.redz_lines = [0.6, 1.1, 1.6] # Redz
        self.oii_lines = [8]

        # Trainig variable
        self.iTrain = self.field != 2

    def plot_data(self, model_tag="", cv_tag="", train=False, plot_rex=False):
        """
        Use self model/plot variables to plot the data given an external figure ax_list.
        Save the resulting image using title_str (does not include extension)

        If train = True, then only plot what is designated as training data.
        If plot_rex=False, then plot all data together. If True, plot according psf and non-psf type.
        """

        print "Corr plot - var_xyz and r_exp - all classes together"
        lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_exp_r]
        binws = [self.dx, self.dy, self.dz, self.dr]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.r_exp_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.exp_r_lines]
        num_cat = 3
        num_vars = 4        

        if plot_rex:
            for e in [(self.ipsf, "PSF"), (self.iext, "EXT")]:
                iselect, tag = e
                print tag
                variables = []
                weights = []                
                for ibool in [self.iNonELG, self.iNoZ, self.iELG]:
                    iplot = np.copy(ibool)
                    if train:
                        iplot = iplot & self.iTrain
                    iplot = iplot & iselect
                    variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.rex_expr[iplot]])
                    weights.append(self.w[iplot])

                fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
                ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines, category_names=self.category, pt_sizes=[2.5, 2.5, 2.5], colors=self.colors, ft_size_legend = 15, lw_dot=2)
                plt.savefig("%s-%s-data-all-%s.png" % (model_tag, cv_tag, tag), dpi=200, bbox_inches="tight")
                plt.close()           
        else:
            variables = []
            weights = []            
            for ibool in [self.iNonELG, self.iNoZ, self.iELG]:
                iplot = np.copy(ibool)
                if train:
                    iplot = iplot & self.iTrain
                variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.rex_expr[iplot]])
                weights.append(self.w[iplot])
            fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
            ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines, category_names=self.category, pt_sizes=[2.5, 2.5, 2.5], colors=self.colors, ft_size_legend = 15, lw_dot=2)
            plt.savefig("%s-%s-data-all.png" % (model_tag, cv_tag), dpi=200, bbox_inches="tight")
            plt.close()        



        print "Corr plot - var_xyz and r_exp - separately"
        lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_exp_r]
        binws = [self.dx, self.dy, self.dz, self.dr]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.r_exp_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.exp_r_lines]
        num_cat = 1
        num_vars = 4        

        if plot_rex:
            for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
                print "Plotting %s" % self.category[i]                                
                for e in [(self.ipsf, "PSF"), (self.iext, "EXT")]:
                    iselect, tag = e
                    print tag
                    variables = []
                    weights = []                
                    fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))                
                    iplot = np.copy(ibool)
                    if train:
                        iplot = iplot & self.iTrain
                    iplot = iplot & iselect
                    variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.rex_expr[iplot]])
                    weights.append(self.w[iplot])

                    ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights, lines=lines, category_names=[self.category[i]], pt_sizes=[2.5], colors=None, ft_size_legend = 15, lw_dot=2)
                    plt.savefig("%s-%s-data-%s-%s.png" % (model_tag, cv_tag, self.category[i], tag), dpi=200, bbox_inches="tight")
                    plt.close() 
        else:
            for i, ibool in enumerate([self.iNonELG, self.iNoZ, self.iELG]):
                print "Plotting %s" % self.category[i]                
                variables = []
                weights = []                
                iplot = np.copy(ibool)
                if train:
                    iplot = iplot & self.iTrain
                variables.append([self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.rex_expr[iplot]])
                weights.append(self.w[iplot])

                fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(35, 35))
                ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=[2.5], colors=None, ft_size_legend = 15, lw_dot=2)

                plt.savefig("%s-%s-data-%s.png" % (model_tag, cv_tag, self.category[i]), dpi=200, bbox_inches="tight")
                plt.close()


        print "Corr plot - var_xyz, r_exp, red_z, oii - ELG only"
        num_cat = 1
        num_vars = 6
        lims = [self.lim_x, self.lim_y, self.lim_z, self.lim_exp_r, self.lim_redz, self.lim_oii]
        binws = [self.dx, self.dy, self.dz, self.dr, self.dred_z, self.doii]
        var_names = [self.var_x_name, self.var_y_name, self.var_z_name, self.r_exp_name, self.red_z_name, self.oii_name]
        lines = [self.var_x_lines, self.var_y_lines, self.var_z_lines, self.exp_r_lines, self.redz_lines, self.oii_lines]

        if plot_rex:
            for e in [(self.ipsf, "PSF"), (self.iext, "EXT")]:
                iselect, tag = e
                print tag
                variables = []
                weights = []    
                iplot = np.copy(self.iELG) & iselect
                if train:
                    iplot = iplot & self.iTrain
                i = 2 # For category
                variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.rex_expr[iplot], self.red_z[iplot], self.oii[iplot]]]
                weights = [self.w[iplot]]

                fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(50, 50))
                ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=[2.5], colors=None, ft_size_legend = 15, lw_dot=2)

                plt.savefig("%s-%s-data-ELG-%s-redz-oii.png" % (model_tag, cv_tag, tag), dpi=200, bbox_inches="tight")
                plt.close()            

        else:
            iplot = np.copy(self.iELG)
            if train:
                iplot = iplot & self.iTrain
            i = 2 # For category
            variables = [[self.var_x[iplot], self.var_y[iplot], self.var_z[iplot], self.rex_expr[iplot], self.red_z[iplot], self.oii[iplot]]]
            weights = [self.w[iplot]]

            fig, ax_list = plt.subplots(num_vars, num_vars, figsize=(50, 50))
            ax_dict = make_corr_plots(ax_list, num_cat, num_vars, variables, lims, binws, var_names, weights=weights, lines=lines, category_names=[self.category[i]], pt_sizes=[2.5], colors=None, ft_size_legend = 15, lw_dot=2)

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
    parametrization: redz, oii, rex_r, zflux/rflux, rflux/gflux, gflux
    """
    def __init__(self):
        parent_model.__init__(self)

        # Re-parametrizing variables
        self.var_x, self.var_y, self.var_z= self.var_reparam(self.gflux, self.rflux, self.zflux) 

        # Plot variables
        # var limits
        self.lim_x = [-.1, 5.] # rf/gf
        self.lim_y = [-.1, 5.] # zf/rf
        self.lim_z = [0., mag2flux(22.)] # gf

        # bin widths
        self.dx = 0.05
        self.dy = 0.05
        self.dz = 2.5e-2

        # var names
        self.var_x_name = r"$f_z/f_r$"
        self.var_y_name = r"$f_r/f_g$"
        self.var_z_name = r"$f_g$"
        self.oii_name =  r"$OII$"
        self.r_exp_name = r"$r_{exp}$"
        self.red_z_name = r"$\eta$"

        # var lines
        self.var_x_lines = [1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        self.var_y_lines = [1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        self.var_z_lines = [mag2flux(f) for f in [21, 22, 23, 24, 25]]
        self.exp_r_lines = [0.25, 0.5, 0.75, 1.0]
        self.redz_lines = [0.6, 1.1, 1.6] # Redz
        self.oii_lines = [8]

    def var_reparam(self, g, r, z):
        return z/r, r/g, g



class model2(parent_model):
    """
    parametrization: redz, oii, rex_r, arcsinh zflux/rflux, arcsinh rflux/gflux, gflux
    """
    def __init__(self):
        parent_model.__init__(self)
        # Re-parametrizing variables
        self.var_x, self.var_y, self.var_z= self.var_reparam(self.gflux, self.rflux, self.zflux) 

        # Plot variables
        # var limits
        self.lim_x = [-.1, 2.] # rf/gf
        self.lim_y = [-.1, 2.] # zf/rf
        self.lim_z = [0., mag2flux(22.)] # gf

        # bin widths
        self.dx = 0.025
        self.dy = 0.025
        self.dz = 2.5e-2

        # var names
        self.var_x_name = r"$sinh^{-1} (f_z/f_r)$"  
        self.var_y_name = r"$sinh^{-1} (f_r/f_g)$"
        self.var_z_name = r"$f_g$"
        self.oii_name =  r"$OII$"
        self.r_exp_name = r"$r_{exp}$"
        self.red_z_name = r"$\eta$"

        # var lines
        self.var_x_lines = []# np.asarray([1/2.5**2, 1/2.5, 1., 2.5, 2.5**2])
        self.var_y_lines = []#[1/2.5**2, 1/2.5, 1., 2.5, 2.5**2]
        self.var_z_lines = [mag2flux(f) for f in [21, 22, 23, 24, 25]]
        self.exp_r_lines = [0.25, 0.5, 0.75, 1.0]
        self.redz_lines = [0.6, 1.1, 1.6] # Redz
        self.oii_lines = [8]

    def var_reparam(self, g, r, z):
        return np.arcsinh(z/r/2.), np.arcsinh(r/g/2.), g
