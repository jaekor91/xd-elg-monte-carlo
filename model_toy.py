import numpy as np
from xd_elg_utils import *
import sys
import os.path
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def mag2flux(mag):
    return 10**(0.4*(22.5-mag))

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)


class toy_model:
    """
    parametrization: g-r, g-z, g, g-OII, z
    """    
    def __init__(self):
        # Basic class variables
        self.category = ["star", "galaxy"]
        self.colors = ["black", "red", "orange"]

        # Plot variables
        # var limits
        self.lim_x = [-2.5, 2.5] # g-z
        self.lim_y = [-2.5, 2.5] # g-r
        self.lim_z = [-10, 10] # g-oii
        self.lim_gmag = [21, 24.0]

        # var names
        self.var_x_name = r"$g-z$"        
        self.var_y_name = r"$g-r$"  
        self.var_z_name = r"$g-oii$"
        self.red_z_name = r"$\eta$"
        self.gmag_name  = r"$g$"

        # ----- MC Sample Variables ----- # 
        self.area_MC = 100

        # FoM value options.
        self.FoM_option = "flat"

        # Flux range to draw the sample from. Slightly larger than the range we are interested.
        self.fmin_MC = mag2flux(24.25) # Note that around 23.8, the power law starts to break down.
        self.fmax_MC = mag2flux(18.00)
        self.fcut = mag2flux(24.) # After noise addition, we make a cut at 24.
        # Original sample.
        # 0: Star, 1: Galaxy
        self.NSAMPLE = [None, None]
        self.gflux0 = [None, None] # 0 for original
        self.rflux0 = [None, None] # 0 for original
        self.zflux0 = [None, None] # 0 for original
        self.oii0 = [None, None] # Although only ELG class has oii and redz, for consistency, we have three elements lists.
        self.redz0 = [None, None]
        # Default noise levels
        self.glim_err = 23.8
        self.rlim_err = 23.4
        self.zlim_err = 22.4
        self.oii_lim_err = 8 # 7 sigma
        # Noise seed. err_seed ~ N(0, 1). This can be transformed by scaling appropriately.
        self.g_err_seed = [None, None] # Error seed.
        self.r_err_seed = [None, None] # Error seed.
        self.z_err_seed = [None, None] # Error seed.
        self.oii_err_seed = [None, None] # Error seed.
        # Noise convolved values
        self.gflux_obs = [None, None] # obs for observed
        self.rflux_obs = [None, None] # obs for observed
        self.zflux_obs = [None, None] # obs for observed
        self.oii_obs = [None, None] # Although only ELG class has oii and redz, for consistency, we have three elements lists.

        # FoM per sample. Note that FoM depends on the observed property such as OII.
        self.FoM_obs = [None, None]

        # Observed final distributions
        self.var_x_obs = [None, None] # r-z
        self.var_y_obs = [None, None] # g-r
        self.var_z_obs = [None, None] # g-oii
        self.redz_obs = [None, None] 
        self.gmag_obs = [None, None]

        # Cell number
        self.cell_number_obs = [None, None]

        # Selection grid limits and number of bins 
        # var_x, var_y, gmag. Width (0.05, 0.05, 0.025)
        self.var_x_limits = [-6, 6]
        self.var_y_limits = [-6, 6]
        self.gmag_limits = [18, 24.]
        self.num_bins = [240, 240, 240]

        # Sigma widths to be used in kernel approximation.
        self.sigmas = [1., 1., 1.]

        # Cell_number in selection
        self.cell_select = None

        # Desired nubmer of objects
        self.num_desired = 2100

        # Regularization number when computing utility
        # In a square field, we expect about 20K objects.
        self.N_regular = 1e3

        self.FoM_star = -1



    def set_FoM_option(self, FoM_option):
        self.FoM_option = FoM_option
        return None
    
    def set_f_NoZ(self, fNoZ):
        self.f_NoZ = fNoZ
        return None

    def set_num_desired(self, Ntot):
        self.num_desired = Ntot
        return None


    # def var_reparam(self, gflux, rflux, zflux, oii = None):
    #     """
    #     Given the input variables return the model3 parametrization as noted above.
    #     """
    #     mu_g = flux2asinh_mag(gflux, band = "g")
    #     mu_r = flux2asinh_mag(rflux, band = "r")
    #     mu_z = flux2asinh_mag(zflux, band = "z")
    #     if oii is not None:
    #         mu_oii = flux2asinh_mag(oii, band = "oii")
    #         return mu_g - mu_z, mu_g - mu_r, mu_g - mu_oii, flux2mag(gflux)
    #     else:
    #         return mu_g - mu_z, mu_g - mu_r, flux2mag(gflux)

    def set_area_MC(self, val):
        self.area_MC = val
        return

    def gen_sample_intrinsic(self):
        """
        Generate sample from MoG x Power Law that is normalized to per sq. deg. density.

        The parameters are harcoded in.

        # dNdf tuned such that for g [21, 24] there are 6K/2K stars/galaxies.        
        """

        # ---- Stars ---- #
        i=0
        # Load parameters
        alpha, A = -1.5, 2140*3/4.
        amps, means, covs = np.array([1]), np.array([[-1., 1.]]), np.array([[[0.75, 0.5], [0.5, 0.75]]])
        NSAMPLE = int(round(integrate_pow_law(alpha, A, self.fmin_MC, self.fmax_MC) *  self.area_MC))#
        print "# of star sample generated: %d" % NSAMPLE

        # Sample flux
        gflux = gen_pow_law_sample(self.fmin_MC, NSAMPLE, alpha, exact=True, fmax=self.fmax_MC)
        g = flux2mag(gflux)

        # Generate Nsample from MoG.
        MoG_sample = sample_MoG(amps, means, covs, NSAMPLE)

        # Compute other fluxes
        gz, gr = MoG_sample[:,0], MoG_sample[:,1]
        z = g - gz
        r = g - gr
        zflux = mag2flux(z)
        rflux = mag2flux(r)        

        # Save the generated fluxes
        self.gflux0[i] = gflux
        self.rflux0[i] = rflux
        self.zflux0[i] = zflux
        self.NSAMPLE[i] = NSAMPLE           
        


        # ---- Galaxies ---- #
        i=1
        # Load parameters
        alpha, A = -3, 252.5
        amps, means = np.array([1]), np.array([[.5, -0.5, 4, 1.0]])
        covs = np.array([[ [1, 0.5, 0.5, 0],
                         [0.5, 1, 0.5, 0],
                         [0.5, 0.5, 1, 0],
                         [0, 0, 0, 0.05]]
                         ])
        NSAMPLE = int(round(integrate_pow_law(alpha, A, self.fmin_MC, self.fmax_MC) * self.area_MC))#
        print "# of galaxy sample generated: %d" % NSAMPLE        

        # Sample flux
        gflux = gen_pow_law_sample(self.fmin_MC, NSAMPLE, alpha, exact=True, fmax=self.fmax_MC)
        g = flux2mag(gflux)

        # Generate Nsample from MoG.
        MoG_sample = sample_MoG(amps, means, covs, NSAMPLE)

        # Compute other fluxes
        gz, gr = MoG_sample[:,0], MoG_sample[:,1]
        z = g - gz
        r = g - gr
        zflux = mag2flux(z)
        rflux = mag2flux(r)        

        # Save the generated fluxes
        self.gflux0[i] = gflux
        self.rflux0[i] = rflux
        self.zflux0[i] = zflux
        self.NSAMPLE[i] = NSAMPLE    

        goii, redz = MoG_sample[:,2], MoG_sample[:,3]
        oii = g - goii # mag
        oii_flux = mag2flux(oii)

        # Saving
        self.redz0[i] = redz
        self.oii0[i] = oii_flux
        self.NSAMPLE[i] = NSAMPLE
        # oii error seed
        self.oii_err_seed[i] = gen_err_seed(NSAMPLE)

        
        # Gen err seed and save
        for i in range(2):
            self.g_err_seed[i] = gen_err_seed(self.NSAMPLE[i])
            self.r_err_seed[i] = gen_err_seed(self.NSAMPLE[i])
            self.z_err_seed[i] = gen_err_seed(self.NSAMPLE[i])             
        return


    def plot_grz_g(self, option=0, save_fig=False, plot_fig=True, title_str="test.png"):
        """
        Make a two-panel plot of observed distribution.
        1) g-z vs. g-r
        2) g hist

        option
        0: Plot everything together in the same color
        1: Plot star/galaxy in different colors
        2: Plot star/galaxy bad/galaxy good in different colors! 
        """

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if option == 0:
            for i in range(2):
                # plot g-z vs. g-r
                ax1.scatter(self.var_x_obs[i], self.var_y_obs[i], edgecolors="none", c="black", s=5)
            ax1.set_xlabel(r"$g-z$", fontsize=20)
            ax1.set_ylabel(r"$g-r$", fontsize=20)
            ax1.axis("equal")
            ax1.set_xlim([-4, 4])
            ax1.set_ylim([-4, 4])
            # plot g hist
            ax2.hist(np.concatenate((self.gmag_obs[0], self.gmag_obs[1])), \
                bins=np.arange(22, 24.1, 0.05), histtype="step", color="black", lw=2)
            ax2.set_xlabel(r"$g$", fontsize=20)
            ax2.set_ylabel(r"$dN/dg$", fontsize=20)
            ax2.set_ylim([0, 360])
            ax2.set_xlim([22, 24])                        

        elif option == 1:
            colors = ["black", "red"]
            labels= ["star", "galaxy"]
            for i in [1, 0]:
                # plot g-z vs. g-r
                ax1.scatter(self.var_x_obs[i], self.var_y_obs[i], edgecolors="none", c=colors[i], s=5, label=labels[i])
                # plot g hist
                ax2.hist(self.gmag_obs[i], \
                    bins=np.arange(22, 24.1, 0.05), histtype="step", color=colors[i], lw=2, label=labels[i])
            # ax1 deocorations 
            ax1.set_xlabel(r"$g-z$", fontsize=20)
            ax1.set_ylabel(r"$g-r$", fontsize=20)
            ax1.axis("equal")
            ax1.set_xlim([-4, 4])
            ax1.set_ylim([-4, 4])
            ax1.legend(loc="upper left", fontsize=15)
            # ax2 decorations
            ax2.set_xlabel(r"$g$", fontsize=20)
            ax2.set_ylabel(r"$dN/dg$", fontsize=20)
            ax2.set_xlim([22, 24])            
            ax2.set_ylim([0, 200])
            ax2.legend(loc="upper left", fontsize=15) 
        else:
            pass


        if save_fig:
            plt.savefig(title_str, dpi=400, bbox_inches="tight")

        if plot_fig:
            plt.show()

        plt.close()

        return


    def plot_grz(self, option=0, save_fig=False, plot_fig=True, title_str="test.png", color_cut = False):
        """
        Plot g-z vs. g-r with a color cut boundary
        """

        fig, ax1 = plt.subplots(1, figsize=(5, 5))

        def y(x):
            x1, x2 = 0, 1
            y1, y2 = 0.75, 1.75
            m = (y2-y1)/float(x2-x1)
            b = (y2-m*x2)
            return m*x+b

        if option == 0:
            for i in range(2):
                # plot g-z vs. g-r
                ax1.scatter(self.var_x_obs[i], self.var_y_obs[i], edgecolors="none", c="black", s=5)
                ax1.plot([-4, 4], [y(-4), y(4)], c="green", lw=2)
            ax1.set_xlabel(r"$g-z$", fontsize=20)
            ax1.set_ylabel(r"$g-r$", fontsize=20)
            ax1.axis("equal")
            ax1.set_xlim([-4, 4])
            ax1.set_ylim([-4, 4])               
        elif option == 1:
            colors = ["black", "red"]
            labels= ["star", "galaxy"]
            for i in [1, 0]:
                # plot g-z vs. g-r
                ax1.scatter(self.var_x_obs[i], self.var_y_obs[i], edgecolors="none", c=colors[i], s=5, label=labels[i])

            # ax1 deocorations 
            ax1.plot([-4, 4], [y(-4), y(4)], c="green", lw=2)            
            ax1.set_xlabel(r"$g-z$", fontsize=20)
            ax1.set_ylabel(r"$g-r$", fontsize=20)
            ax1.axis("equal")
            ax1.set_xlim([-4, 4])
            ax1.set_ylim([-4, 4])
            ax1.legend(loc="upper left", fontsize=15)
           

        if save_fig:
            plt.savefig(title_str, dpi=400, bbox_inches="tight")

        if plot_fig:
            plt.show()

        plt.close()

        return        


    def apply_color_cut(self):
        """
        g-r< g-z selection.  

        y < x
        """

        # Star selected
        i = 0
        Nstar = (self.var_x_obs[i]+0.75 > self.var_y_obs[i]).sum() / float(self.area_MC)

        # Galaxy selected
        i = 1
        Ngal = (self.var_x_obs[i]+0.75 > self.var_y_obs[i]).sum() / float(self.area_MC)

        # Ndensity
        Ndensity = Nstar+Ngal

        # Eff
        eff = Ngal/(Ndensity)

        return Ndensity, eff



    def set_err_lims(self, glim, rlim, zlim, oii_lim):
        """
        Set the error characteristics.
        """
        self.glim_err = glim
        self.rlim_err = rlim 
        self.zlim_err = zlim
        self.oii_lim_err = oii_lim

        return



    def gen_err_conv_sample(self):
        """
        Given the error properties glim_err, rlim_err, zlim_err, oii_lim_err, add noise to the intrinsic density
        sample and compute the parametrization.
        """
        print "Convolving error and re-parametrizing"        
        # star, galaxy
        for i in range(2):
            print "%s" % self.category[i]
            self.gflux_obs[i] = self.gflux0[i] + self.g_err_seed[i] * mag2flux(self.glim_err)/5.
            self.rflux_obs[i] = self.rflux0[i] + self.r_err_seed[i] * mag2flux(self.rlim_err)/5.
            self.zflux_obs[i] = self.zflux0[i] + self.z_err_seed[i] * mag2flux(self.zlim_err)/5.

            # Make flux cut
            ifcut = self.gflux_obs[i] > self.fcut
            self.gflux_obs[i] = self.gflux_obs[i][ifcut]
            self.rflux_obs[i] = self.rflux_obs[i][ifcut]
            self.zflux_obs[i] = self.zflux_obs[i][ifcut]

            # Compute model parametrization
            g = flux2mag(self.gflux_obs[i])
            r = flux2mag(self.rflux_obs[i])
            z = flux2mag(self.zflux_obs[i])
            self.var_x_obs[i] = g-z
            self.var_y_obs[i] = g-r
            self.gmag_obs[i] = flux2mag(self.gflux_obs[i])

            # Number of samples after the cut.
            Nsample = self.gmag_obs[i].size

            # More parametrization to compute for ELGs. Also, compute FoM.
            if i==1:
                # oii parameerization. 
                self.oii_obs[i] = self.oii0[i] + self.oii_err_seed[i] * (self.oii_lim_err/7.) # flux 
                self.oii_obs[i] = self.oii_obs[i][ifcut] # flux
                oii = flux2mag(self.oii_obs[i]) # Flux to mag conversion
                self.var_z_obs[i] = g-oii

                # Redshift has no uncertainty
                self.redz_obs[i] = self.redz0[i][ifcut]

                # Gen FoM 
                self.FoM_obs[i] = self.gen_FoM(i, Nsample, self.oii_obs[i], self.redz_obs[i])
            else:
                # Gen FoM 
                self.FoM_obs[i] = self.gen_FoM(i, Nsample)

        return



    def gen_FoM(self, cat, Nsample, oii=None, redz=None):
        """
        Give the category number
        0: Stars
        1: Galaxies
        compute the appropriate FoM corresponding to each sample.
        """
        if cat == 0:
            return np.ones(Nsample, dtype=float) * self.FoM_star
        elif cat == 1:
            if (oii is None) or (redz is None):
                "You must provide oii AND redz"
                assert False
            else:
                if self.FoM_option == "flat":# Flat option
                    ibool = (oii>8) & (redz > 0.6) # For objects that lie within this criteria
                    FoM = np.zeros(Nsample, dtype=float)
                    FoM[ibool] = 1.0
                elif self.FoM_option == "NoOII": # NoOII means objects without OII values are also included.
                    ibool = (redz > 0.6) # For objects that lie within this criteria
                    FoM = np.zeros(Nsample, dtype=float)
                    FoM[ibool] = 1.0
                elif self.FoM_option == "Linear_redz": # FoM linearly scale with redshift
                    ibool = (oii>8) & (redz > 0.6) & (redz <1.6) # For objects that lie within this criteria
                    FoM = np.zeros(Nsample, dtype=float)
                    FoM[ibool] = 1 + (redz[ibool]-0.6) * 5. # This means redz = 1.6 has FoM of 2.
                elif self.FoM_option == "Quadratic_redz": # FoM linearly scale with redshift
                    ibool = (oii>8) & (redz > 0.6) & (redz <1.6) # For objects that lie within this criteria
                    FoM = np.zeros(Nsample, dtype=float)
                    FoM[ibool] = 1 + 10 * (redz[ibool]-0.6) ** 2 # This means redz = 1.6 has FoM of 2.                    

                return FoM




    def gen_selection_volume_scipy(self):
        """
        Given the generated sample (intrinsic val + noise), generate a selection volume,
        using kernel approximation to the number density. That is, when tallying up the 
        number of objects in each cell, use a gaussian kernel centered at the cell where
        the particle happens to fall.

        This version is different from the vanila version with kernel option in that
        the cross correlation or convolution is done using scipy convolution function.

        Note we don't alter the generated sample in any way.
        
        Strategy:
            - Construct a multi-dimensional histogram.
            - Perform FFT convolution with a gaussian kernel. Use padding.
            - Given the resulting convolved MD histogram, we can flatten it and order them according to
            utility. Currently, the utility is FoM divided by Ntot.
            - We can either predict the number density to define a cell of selected cells 
            or remember the order of the entire (or half of) the cells. Then when selection is applied
            we can include objects up to the number we want by adjust the utility threshold.
            This would require remember the number density of all the objects.
        """
        # Create MD histogarm of each type of objects. 
        # 0: star, 1: galaxy
        MD_hist_N_star, MD_hist_N_gal_good, MD_hist_N_gal_bad = None, None, None
        MD_hist_N_FoM = None # Tally of FoM corresponding to all objects in the category.
        MD_hist_N_total = None # Tally of all objects.

        # star
        i = 0
        samples = np.array([self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]).T
        MD_hist_N_star, edges = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits])
        FoM_tmp, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.FoM_obs[i])
        MD_hist_N_FoM = FoM_tmp
        MD_hist_N_total = np.copy(MD_hist_N_star)

        # gal (good and bad)
        i=1
        samples = np.array([self.var_x_obs[i], self.var_y_obs[i], self.gmag_obs[i]]).T
        Nsample = self.redz_obs[i].size
        w_good = np.ones(Nsample, dtype=bool) # (self.redz_obs[i]>0.6) & (self.redz_obs[i]<1.6) & (self.oii_obs[i]>8) # Only objects in the correct redshift and OII ranges.
        w_bad = np.zeros(Nsample, dtype=bool)# (self.redz_obs[i]>0.6) & (self.redz_obs[i]<1.6) & (self.oii_obs[i]<8) # Only objects in the correct redshift and OII ranges.
        MD_hist_N_gal_good, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=w_good)
        MD_hist_N_gal_bad, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=w_bad)
        FoM_tmp, _ = np.histogramdd(samples, bins=self.num_bins, range=[self.var_x_limits, self.var_y_limits, self.gmag_limits], weights=self.FoM_obs[i])
        MD_hist_N_FoM += FoM_tmp
        MD_hist_N_total += MD_hist_N_gal_good
        MD_hist_N_total += MD_hist_N_gal_bad

        # Applying Gaussian filtering
        sigma_limit = 5
        gaussian_filter(MD_hist_N_star, self.sigmas, order=0, output=MD_hist_N_star, mode='constant', cval=0.0, truncate=sigma_limit)
        gaussian_filter(MD_hist_N_gal_good, self.sigmas, order=0, output=MD_hist_N_gal_good, mode='constant', cval=0.0, truncate=sigma_limit)
        gaussian_filter(MD_hist_N_gal_bad, self.sigmas, order=0, output=MD_hist_N_gal_bad, mode='constant', cval=0.0, truncate=sigma_limit)
        gaussian_filter(MD_hist_N_FoM, self.sigmas, order=0, output=MD_hist_N_FoM, mode='constant', cval=0.0, truncate=sigma_limit)
        gaussian_filter(MD_hist_N_total, self.sigmas, order=0, output=MD_hist_N_total, mode='constant', cval=0.0, truncate=sigma_limit)


        # Compute utility
        # Change the FoM according to the crums.
        utility = MD_hist_N_FoM/(MD_hist_N_total + (self.N_regular * self.area_MC / float(np.multiply.reduce(self.num_bins))))# Note the multiplication by the area.

        # Flatten utility array
        utility_flat = utility.flatten()

        # Order cells according to utility
        # This corresponds to cell number of descending order sorted array.
        idx_sort = (-utility_flat).argsort()

        # Flatten other arrays.
        # Sort flattened arrays according to utility.        
        MD_hist_N_star_flat = MD_hist_N_star.flatten()[idx_sort]
        MD_hist_N_gal_good_flat = MD_hist_N_gal_good.flatten()[idx_sort]
        MD_hist_N_gal_bad_flat = MD_hist_N_gal_bad.flatten()[idx_sort]
        MD_hist_N_FoM_flat = MD_hist_N_FoM.flatten()[idx_sort]
        MD_hist_N_total_flat = MD_hist_N_total.flatten()[idx_sort]        

        # Starting from the keep including cells until the desired number is eached.        
        Ntotal = 0
        counter = 0
        for ntot in MD_hist_N_total_flat:
            if Ntotal > (self.num_desired * self.area_MC): 
                break            
            Ntotal += ntot
            counter +=1

        # Predicted numbers in the selection.
        Ntotal = np.sum(MD_hist_N_total_flat[:counter])/float(self.area_MC)
        Ngood = np.sum(MD_hist_N_gal_good_flat[:counter])/float(self.area_MC)
        N_star = np.sum(MD_hist_N_star_flat[:counter])/float(self.area_MC)
        N_gal_bad = np.sum(MD_hist_N_gal_bad_flat[:counter])/float(self.area_MC)
        eff = (Ngood/float(Ntotal))    
            

        # Save the selection
        self.cell_select = np.sort(idx_sort[:counter])

        # Return the answer
        return eff, Ntotal, Ngood, N_star, N_gal_bad
    




    # def apply_selection(self, gflux, rflux, zflux):
    #     """
    #     Model 3
    #     Given gflux, rflux, zflux of samples, return a boolean vector that gives the selection.
    #     """
    #     mu_g = flux2asinh_mag(gflux, band = "g")
    #     mu_r = flux2asinh_mag(rflux, band = "r")
    #     mu_z = flux2asinh_mag(zflux, band = "z")

    #     var_x = mu_g - mu_z
    #     var_y = mu_g - mu_r
    #     gmag = flux2mag(gflux)

    #     samples = [var_x, var_y, gmag]

    #     # Generate cell number 
    #     cell_number = multdim_grid_cell_number(samples, 3, [self.var_x_limits, self.var_y_limits, self.gmag_limits], self.num_bins)

    #     # Sort the cell number
    #     idx_sort = cell_number.argsort()
    #     cell_number = cell_number[idx_sort]

    #     # Placeholder for selection vector
    #     iselect = check_in_arr2(cell_number, self.cell_select)

    #     # The last step is necessary in order for iselect to have the same order as the input sample variables.
    #     idx_undo_sort = idx_sort.argsort()        
    #     return iselect[idx_undo_sort]

    def cell_select_centers(self):
        """
        Return selected cells centers. Model3.
        """
        limits = [self.var_x_limits, self.var_y_limits, self.gmag_limits]
        Ncell_select = self.cell_select.size # Number of cells in the selection
        centers = [None, None, None]

        for i in range(3):
            Xmin, Xmax = limits[i]
            bin_edges, dX = np.linspace(Xmin, Xmax, self.num_bins[i]+1, endpoint=True, retstep=True)
            bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.
            idx = (self.cell_select % np.multiply.reduce(self.num_bins[i:])) //  np.multiply.reduce(self.num_bins[i+1:])
            centers[i] = bin_centers[idx.astype(int)]

        return np.asarray(centers).T



    def gen_select_boundary_slices(self, slice_dir = 2, model_tag="", cv_tag="", centers=None, plot_ext=False,\
        gflux_ext=None, rflux_ext=None, zflux_ext=None, ibool_ext = None,\
        var_x_ext=None, var_y_ext=None, gmag_ext=None, use_parameterized_ext=False,\
        pt_size=10, pt_size_ext=10, alpha_ext=0.5, guide=False, output_sparse=False, increment=10):
        """
        Model3

        Given slice direction, generate slices of boundary

        0: var_x
        1: var_y
        2: gmag

        If plot_ext True, then plot user supplied external objects.

        If centers is not None, then use it instead of generating one.

        If use_parameterized_ext, then the user may provide already parameterized version of the external data points.

        If guide True, then plot the guide line.

        If output_sparse=True, then only 10% of the boundaries are plotted and saved.
        """

        slice_var_tag = ["g-z", "g-r", "gmag"]
        var_names = [self.var_x_name, self.var_y_name, self.gmag_name]

        if centers is None:
            centers = self.cell_select_centers()

        if guide:
            x_guide = np.arange(-10, 10)
            y_guide = x_guide + 0.75

        if plot_ext:
            if use_parameterized_ext:
                if ibool_ext is not None:
                    var_x_ext = var_x_ext[ibool_ext]
                    var_y_ext = var_y_ext[ibool_ext]
                    gmag_ext = gmag_ext[ibool_ext]                
            else:     
                pass
                # if ibool_ext is not None:
                #     gflux_ext = gflux_ext[ibool_ext]
                #     rflux_ext = rflux_ext[ibool_ext]
                #     zflux_ext = zflux_ext[ibool_ext]

                # mu_g, mu_r, mu_z = flux2asinh_mag(gflux_ext, band="g"), flux2asinh_mag(rflux_ext, band="r"), flux2asinh_mag(zflux_ext, band="z")
                # var_x_ext = mu_g-mu_z
                # var_y_ext = mu_g-mu_r
                # gmag_ext = flux2mag(gflux_ext)

            variables = [var_x_ext, var_y_ext, gmag_ext]

        limits = [self.var_x_limits, self.var_y_limits, self.gmag_limits]        
        Xmin, Xmax = limits[slice_dir]
        bin_edges, dX = np.linspace(Xmin, Xmax, self.num_bins[slice_dir]+1, endpoint=True, retstep=True)

        print slice_var_tag[slice_dir]
        if output_sparse:
            iterator = range(0, self.num_bins[slice_dir], increment)
        else:
            iterator = range(self.num_bins[slice_dir])

        for i in iterator: 
            ibool = (centers[:, slice_dir] < bin_edges[i+1]) & (centers[:, slice_dir] > bin_edges[i])
            centers_slice = centers[ibool, :]
            fig = plt.figure(figsize=(7, 7))
            idx = range(3)
            idx.remove(slice_dir)
            plt.scatter(centers_slice[:,idx[0]], centers_slice[:,idx[1]], edgecolors="none", c="green", alpha=0.5, s=pt_size)
            if plot_ext:
                ibool = (variables[slice_dir] < bin_edges[i+1]) & (variables[slice_dir] > bin_edges[i])
                plt.scatter(variables[idx[0]][ibool], variables[idx[1]][ibool], edgecolors="none", c="red", s=pt_size_ext, alpha=alpha_ext)
            plt.xlabel(var_names[idx[0]], fontsize=15)
            plt.ylabel(var_names[idx[1]], fontsize=15)

            if guide and (slice_dir==2):
                plt.plot(x_guide, y_guide, c="orange", lw = 2)
            # plt.axis("equal")
            plt.xlim(limits[idx[0]])
            plt.ylim(limits[idx[1]])
            title_str = "%s [%.3f, %.3f]" % (var_names[slice_dir], bin_edges[i], bin_edges[i+1])
            print i, title_str
            plt.title(title_str, fontsize=15)
            plt.savefig("%s-%s-boundary-%s-%d.png" % (model_tag, cv_tag, slice_var_tag[slice_dir], i), bbox_inches="tight", dpi=200)
        #     plt.show()
            plt.close()        

