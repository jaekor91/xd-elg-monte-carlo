import numpy as np
import numpy.lib.recfunctions as rec

from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

from os import listdir
from os.path import isfile, join

import scipy.stats as stats
import scipy.optimize as opt
import matplotlib.pyplot as plt
# import extreme_deconvolution as XD

import confidence_contours as cc
from confidence_level_height_estimation import confidence_level_height_estimator, summed_gm, inverse_cdf_gm

# Matplot ticks
import matplotlib as mpl
mpl.rcParams['xtick.major.size'] = 15
mpl.rcParams['xtick.major.width'] = 2.
mpl.rcParams['ytick.major.size'] = 15
mpl.rcParams['ytick.major.width'] = 2.
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20

colors = ["orange", "grey", "brown", "purple", "red", "salmon","black", "white","blue"]
cnames = ["Gold", "Silver", "LowOII", "NoOII", "LowZ", "NoZ", "D2reject", "DR3unmatched","D2unobserved"]

large_random_constant = -999119283571
deg2arcsec=3600


def return_file(fname):
    with open (fname, "r") as myfile:
        data=myfile.readlines()
    return data

def HMS2deg(ra=None, dec=None):
    rs, ds = 1, 1
    if dec is not None:
        D, M, S = [float(i) for i in dec.split(":")]
        if str(D)[0] == '-':
            ds, D = -1, abs(D)
        dec= D + (M/60) + (S/3600)

    if ra is not None:
        H, M, S = [float(i) for i in ra.split(":")]
        if str(H)[0] == '-':
            rs, H = -1, abs(H)
        ra = (H*15) + (M/4) + (S/240)

    if (ra is not None) and (dec is not None):
        return ra, dec 
    elif ra is not None: 
        return ra
    else:
        return dec
    
def MMT_study_color(grz, field, mask=None):
    """
    field:
    - 0 corresponds to 16hr
    - 1 corresponds to 23hr
    """
    g,r,z = grz
    if mask is not None:
        g = g[mask]
        r = r[mask]
        z = z[mask]
    if field == 0:
        return (g<24) & ((g-r)<0.8) & np.logical_or(((r-z)>(0.7*(g-r)+0.2)), (g-r)<0.2)
    else:
        return (g<24) & ((g-r)<0.8) & np.logical_or(((r-z)>(0.7*(g-r)+0.2)), (g-r)<0.2) & (g>20)
    
def MMT_DECaLS_quality(fits, mask=None):
    gany,rany,zany = load_grz_anymask(fits)
    givar, rivar, zivar = load_grz_invar(fits)
    bp = load_brick_primary(fits)
    if bp[0] == 0:
        bp = (bp==0)
    elif type(bp[0])==np.bool_:
        bp = bp # Do nothing    
    else:
        bp = bp=="T"
    r_dev, r_exp = load_shape(fits)
    
    if mask is not None:
        gany, rany, zany = gany[mask], rany[mask], zany[mask]
        givar, rivar, zivar =givar[mask], rivar[mask], zivar[mask]
        bp = bp[mask]
        r_dev, r_exp = r_dev[mask], r_exp[mask]
        
    return (gany==0)&(rany==0)&(zany==0)&(givar>0)&(rivar>0)&(zivar>0)&(bp)&(r_dev<1.5)&(r_exp<1.5)

def load_MMT_specdata(fname, fib_idx=None):
    """
    Given spHect* file address, return wavelength (x),
    flux value (d), inverse variance (divar), and
    AND_mask.

    If fib_idx is not None, then return only spectra
    indicated. 
    """
    table_spec = fits.open(fname)
    x = table_spec[0].data
    d = table_spec[1].data
    divar = table_spec[2].data # Inverse variance
    AND_mask = table_spec[3].data

    if fib_idx is not None:
        x = x[fib_idx,:]
        d = d[fib_idx,:]
        divar = divar[fib_idx,:]
        AND_mask = AND_mask[fib_idx,:]
    return x, d, divar, AND_mask

def box_car_avg(d,window_pixel_size=50,mask=None):
    """
    Take a running average of window_piexl_size pixels.
    Exclude masked pixels from averaging.
    """
    # Filter
    v = np.ones(window_pixel_size)
    
    # Array that tells how many pixels were used
    N_sample = np.ones(d.size)
    if mask is not None:
        N_sample[mask]=0
    N_sample = np.convolve(N_sample, v,mode="same")
    
    # Running sum of the data excluding masked pixels
    if mask is not None:
        d[mask]=0
    d_boxed = np.convolve(d, v,mode="same")
    
    # Taking average
    d_boxed /= N_sample
    
    return d_boxed



def process_spec(d, divar, width_guess, x_mean, mask=None):
    """
    Given the data vector d and its corresopnding inverse
    variance and pix_sigma width for the filter
    compute, integrated flux A, var(A), and chi sq.
    Also, return S2N. width_guess is in Angstrom.
    """
    # Filter
    pix_sigma = width_guess/x_mean # Peak width in terms of pixels
    filter_size = np.ceil(pix_sigma*4*2) # How large the filter has to be to encompass 4-sig
    # If filter size is odd, add one.
    if filter_size%2==0:
        filter_size+=1

    # Centered around the filter, create a gaussian.
    v_center = int(filter_size/2)
    v = np.arange(int(filter_size))-v_center
    v = np.exp(-(v**2)/(2*(pix_sigma**2)))/(pix_sigma*np.sqrt(2*np.pi))
    # Note: v = G(A=1)
    
    # If mask is used, then block out the appropriate
    # portion.
    if mask is not None:
        d[mask]=0
        divar[mask]=0
    
    # varA: Running sum of (ivar*v^2) excluding masked pixels
    varA = np.convolve(divar, v**2, mode="same")
    
    # A_numerator: Running sum of (d*v*ivar)
    A_numerator = np.convolve(d*divar, v, mode="same")
    A = A_numerator/varA
    
    # SN
    S2N = A/np.sqrt(varA)
    
    # Compute reduced chi. sq.
    # To do, compute the number of samples used.
    # Filter
    v_N = np.ones(int(filter_size))    
    N_sample = np.ones(d.size)
    if mask is not None:
        N_sample[mask]=0
    N_sample = np.convolve(N_sample, v_N, mode="same")    

    # Chi sq. # -1 since we are only estimating one parameter.
    chi = -(-2*A_numerator*A+varA*(A**2))/(N_sample-1) 
    
    return A, varA, chi, S2N

def median_filter(data, mask=None, window_pixel_size=50):
    """
    Given the data array and window size, compute median
    without mask.
    """
    array_length = data.size
    
    if window_pixel_size%2==0:
        window_pixel_size+=1
        
    if mask is None:
        mask = np.ones(array_length)
        
    pass_mask = np.logical_not(mask)

    ans = np.zeros(array_length)
    for i in range(array_length):
        idx_l = max(0, i-int(window_pixel_size/2))
        idx_h = min(array_length, i+int(window_pixel_size/2))+1
#         print(idx_l, idx_h)
        tmp = data[idx_l:idx_h][pass_mask[idx_l:idx_h]]
#         print(tmp.size)
        ans[i] = np.median(tmp)
    return ans    

def spec_lines():
    emissions = [3727.3, 4102.8, 4340, 4861.3, 4959,5006.8, 6562.8, 6716]
    absorptions  = [3933.7, 3968.6, 4304.4, 5175.3, 5984.0]
    return emissions, absorptions

def OII_wavelength():
    return 3727.3


def plot_fit(x, d, A, S2N, chi, threshold=5, mask=None, mask_caution=None, xmin=4500, xmax=8500, s=1,\
             plot_show=True, plot_save=False, save_dir=None, plot_title=""):
    """
    Plot a spectrum and its fits.
    """
    if mask is not None:
        S2N[mask] = 0
        A[mask] = 0
        chi[mask] = 0
        d[mask] = 0

    # Limit plot range
    ibool = (x>xmin)&(x<xmax)
    x_masked = x[ibool]
    S2N_masked = S2N[ibool]
    chi_masked = chi[ibool]
    A_masked = A[ibool]
    d_masked = d[ibool]
    if mask_caution is not None:
        mask_caution = mask_caution[ibool]

    # Emission and absorption lines
    emissions, absorptions = spec_lines()
    OII_line = OII_wavelength()
    
    # Find peaks in S2N. Must have 5-sigma.
    isig5 = (S2N_masked>threshold)
    # Create a vector that tells where a peak cluster starts and end. 
    S2N_start_end = np.zeros_like(S2N_masked)
    S2N_start_end[isig5] = 1
    S2N_start_end[1:] = S2N_start_end[1:]-S2N_start_end[:-1]
    S2N_start_end[0] = 0
    # For each [1,...,-1] cluster, finding the idx of maximum and find
    # the corresponding x value.
    starts = np.where(S2N_start_end==1)[0]
    ends = np.where(S2N_start_end==-1)[0]
    z_peak_list = []
    s2n_peak_list = []
    oii_flux_list = []
    for i in range(len(ends)):
        start = starts[i]
        end = ends[i]
        if (start==(end-1)) or (start==end):
            val = S2N_masked[start]
        else:
            val = np.max(S2N_masked[start:end])
        idx = np.where(S2N_masked ==val)[0]
        z_peak_list.append(x_masked[idx]/OII_line-1)
        s2n_peak_list.append(S2N_masked[idx])
        oii_flux_list.append(A_masked[idx])
        
    for guess_num,z_pk in enumerate(z_peak_list):
        info_str = "-".join(["z%.2f"%z_pk,"oii%.2f"%oii_flux_list[guess_num], "s2n%.2f"%s2n_peak_list[guess_num]])
        title_str = "-".join([plot_title, "guess%d"%guess_num , info_str])
        # Create a figure where x-axis is shared
        ft_size = 15        
        fig, (ax0, ax1,ax2,ax3) = plt.subplots(4,figsize=(12,10),sharex=True)

        # Draw lines
        for em in emissions:
            ax0.axvline(x=(em*(z_pk+1)), ls="--", lw=2, c="red")
        for ab in absorptions:
            ax0.axvline(x=(ab*(z_pk+1)), ls="--", lw=2, c="green")            
        ax0.axvline(x=(OII_line*(z_pk+1)), ls="--", lw=2, c="blue")                        
        ax0.set_title(title_str, fontsize=ft_size)
        ax0.plot(x_masked,d_masked,lw=1, c="black")
        ax0.set_xlim([xmin, xmax])
        ax0.set_ylim([max(np.min(d_masked)*1.1,-2),np.max(d_masked)*1.1])
        ax0.set_ylabel(r"Original Flux", fontsize=ft_size)

        # Draw lines
        for em in emissions:
            ax1.axvline(x=(em*(z_pk+1)), ls="--", lw=2, c="red")
        for ab in absorptions:
            ax1.axvline(x=(ab*(z_pk+1)), ls="--", lw=2, c="green")            
        ax1.axvline(x=(OII_line*(z_pk+1)), ls="--", lw=2, c="blue")
        ax1.scatter(x_masked,A_masked,s=s, c="black", edgecolor="none")
        ax1.scatter(x_masked[isig5],A_masked[isig5],s=s, c="red", edgecolor="none")    
        if mask_caution is not None:
            ax1.scatter(x_masked[mask_caution],A_masked[mask_caution],s=s, c="blue", edgecolor="none")                    
        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([-2,np.max(A_masked)*1.1])
        ax1.set_ylabel(r"Integrated Flux", fontsize=ft_size)

        # Draw lines
        for em in emissions:
            ax2.axvline(x=(em*(z_pk+1)), ls="--", lw=2, c="red")
        for ab in absorptions:
            ax2.axvline(x=(ab*(z_pk+1)), ls="--", lw=2, c="green")            
        ax2.axvline(x=(OII_line*(z_pk+1)), ls="--", lw=2, c="blue") 
        ax2.scatter(x_masked,S2N_masked,s=s, c="black", edgecolor="none")    
        ax2.scatter(x_masked[isig5],S2N_masked[isig5],s=s, c="red", edgecolor="none")        
        ax2.axhline(y=5, ls="--", lw=2, c="blue")
        if mask_caution is not None:
            ax2.scatter(x_masked[mask_caution],S2N_masked[mask_caution],s=s, c="blue", edgecolor="none")       
        ax2.set_xlim([xmin, xmax])
        ax2.set_ylim([-1,np.max(S2N_masked)*1.1])
        ax2.set_ylabel(r"S/N", fontsize=ft_size)

        # Draw lines
        for em in emissions:
            ax3.axvline(x=(em*(z_pk+1)), ls="--", lw=2, c="red")
        for ab in absorptions:
            ax3.axvline(x=(ab*(z_pk+1)), ls="--", lw=2, c="green")            
        ax3.axvline(x=(OII_line*(z_pk+1)), ls="--", lw=2, c="blue")
        ax3.scatter(x_masked,chi_masked,s=s, c="black", edgecolor="none")
        ax3.scatter(x_masked[isig5],chi_masked[isig5],s=s, c="red", edgecolor="none")        
        if mask_caution is not None:
            ax3.scatter(x_masked[mask_caution],chi_masked[mask_caution],s=s, c="blue", edgecolor="none")            
        ax3.set_xlim([xmin, xmax])
        ax3.set_ylim([-0.5,np.max(chi_masked)*1.1])    
        ax3.set_xlabel("Wavelength ($\AA$)", fontsize=ft_size)
        ax3.set_ylabel("neg. reduced $\chi^2$", fontsize=ft_size)    

        fig.subplots_adjust(hspace=0.05)
        if plot_save:
            plt.savefig(save_dir+title_str+".png", bbox_inches="tight", dpi=200)
        if plot_show:
            plt.show()
        plt.close() 
        
    return 

def process_spec_best(d, divar, width_guesses, x_mean, mask=None):
    """
    The same as process_spec(), except returns A, varA, chi, S2N values for
    best chi.
    
    width guesses is either a list or numpy array. 
    """
    width_guesses = np.asarray(width_guesses)
    
    # First
    A, varA, chi, S2N = process_spec(d, divar, width_guesses[0], x_mean, mask=mask)    
    for i in range(1,width_guesses.size):
        A_tmp, varA_tmp, chi_tmp, S2N_tmp = process_spec(d, divar, width_guesses[i], x_mean, mask=mask)
        # Swith values if chi squar is higher. Note the we defined chi sq.
        ibool = (chi_tmp>chi) #& ~np.isnan(chi_tmp) 
        # ibool = (np.abs(1-chi_tmp)<np.abs(1-chi)) #& ~np.isnan(chi_tmp)
        # ibool = S2N_tmp>S2N
        A[ibool] = A_tmp[ibool]
        varA[ibool] = varA_tmp[ibool]
        chi[ibool] = chi_tmp[ibool]
        S2N[ibool] = S2N_tmp[ibool]
        
    return  A, varA, chi, S2N    


def plot_spectrum(x,d,x2=None,d2=None, xmin=4000, xmax=8700, lw=0.25, lw2=1, mask=None, mask2=None):
    """
    Plot a spectrum given x,d.
    """
    if mask is not None:
        d[mask] = 0
    ibool = (x>xmin)&(x<xmax)        
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(x[ibool],d[ibool],lw=lw, c="black")
    
    if (x2 is not None) and (d2 is not None):
        ibool = (x2>xmin)&(x2<xmax)
        if mask2 is not None:
            d2[mask2]=0
        plt.plot(x2[ibool],d2[ibool],lw=lw2, c="red")
        
    ft_size = 15
    
    plt.xlim([xmin, xmax])
    plt.xlabel(r"Wavelength ($\AA$)", fontsize=ft_size)
    plt.ylabel(r"Flux ($10^{-17}$ ergs/cm^2/s/$\AA$)", fontsize=ft_size)
    plt.show()
    plt.close()

def plot_S2N(x, S2N, mask=None, xmin=4500, xmax=8500, s=1):
    """
    Plot a spectrum given x,d.
    """
    if mask is not None:
        S2N[mask] = 0
    ibool = (x>xmin)&(x<xmax)        
    
    fig = plt.figure(figsize=(10,5))
    S2N_masked = S2N[ibool]
    plt.scatter(x[ibool],S2N_masked,s=s, c="black")
        
    ft_size = 15
    
    plt.xlim([xmin, xmax])
    plt.ylim([np.min(S2N_masked)*1.2,np.max(S2N_masked)*1.2])
    plt.xlabel(r"Wavelength ($\AA$)", fontsize=ft_size)
    plt.ylabel(r"S/N", fontsize=ft_size)
    plt.show()
    plt.close()    



    
def MMT_radec(field, MMT_data_directory="./MMT_data/"):
    """
    field is one of [0,1,2]:
        - 0: 16hr observation 1
        - 1: 16hr observation 2
        - 2: 23hr observation
    MMT_data_directory: Where the relevant header files are stored.
    """
    num_fibers = 300

    if field==0:
        # 16hr2_1
        # Header file name
        fname = MMT_data_directory+"config1FITS_Header.txt"
        # Get info corresponding to the fibers
        OnlyAPID = [line for line in return_file(fname) if line.startswith("APID")]
        # Get the object type
        APID_types = [line.split("= '")[1].split(" ")[0] for line in OnlyAPID]
        # print(APID_types)
        # Getting index of targets only
        ibool1 = np.zeros(num_fibers,dtype=bool)
        for i,e in enumerate(APID_types):
            if e.startswith("5"):
                ibool1[i] = True        
        APID_targets = [OnlyAPID[i] for i in range(num_fibers) if ibool1[i]]
        fib = [i+1 for i in range(num_fibers) if ibool1[i]]
        # Extract ra,dec
        ra_str = [APID_targets[i].split("'")[1].split(" ")[1] for i in range(len(APID_targets))]
        dec_str = [APID_targets[i].split("'")[1].split(" ")[2] for i in range(len(APID_targets))]
        ra = [HMS2deg(ra=ra_str[i]) for i in range(len(ra_str))]
        dec = [HMS2deg(dec=dec_str[i]) for i in range(len(ra_str))]
    elif field==1:
        # 16hr2_2
        # Header file name
        fname = MMT_data_directory+"config2FITS_Header.txt"
        # Get info corresponding to the fibers
        OnlyAPID  = return_file(fname)[0].split("= '")[1:]
        # Get the object type
        APID_types = [line.split(" ")[0] for line in OnlyAPID]
        # print(APID_types)
        # Getting index of targets only
        ibool2 = np.zeros(num_fibers,dtype=bool)
        for i,e in enumerate(APID_types):
            if e.startswith("5"):
                ibool2[i] = True        
        APID_targets = [OnlyAPID[i] for i in range(num_fibers) if ibool2[i]]
        fib = [i+1 for i in range(num_fibers) if ibool2[i]]
        # print(APID_targets[0])
        # Extract ra,dec
        ra_str = [APID_targets[i].split(" ")[1] for i in range(len(APID_targets))]
        dec_str = [APID_targets[i].split(" ")[2] for i in range(len(APID_targets))]
        ra = [HMS2deg(ra=ra_str[i]) for i in range(len(ra_str))]
        dec = [HMS2deg(dec=dec_str[i]) for i in range(len(ra_str))]
    elif field==2:
        # 23hr
        # Header file name
        fname = MMT_data_directory+"23hrs_FITSheader.txt"
        # Get info corresponding to the fibers
        OnlyAPID  = return_file(fname)[0].split("= '")[1:]
        # Get the object type
        APID_types = [line.split(" ")[0] for line in OnlyAPID]
        # print(APID_types)
        # Getting index of targets only
        ibool3 = np.zeros(num_fibers,dtype=bool)
        for i,e in enumerate(APID_types):
            if e.startswith("3"):
                ibool3[i] = True        
        APID_targets = [OnlyAPID[i] for i in range(num_fibers) if ibool3[i]]
        fib = [i+1 for i in range(num_fibers) if ibool3[i]]        
        # print(APID_targets[0])
        # Extract ra,dec
        ra_str = [APID_targets[i].split(" ")[1] for i in range(len(APID_targets))]
        dec_str = [APID_targets[i].split(" ")[2] for i in range(len(APID_targets))]
        ra = [HMS2deg(ra=ra_str[i]) for i in range(len(ra_str))]
        dec = [HMS2deg(dec=dec_str[i]) for i in range(len(ra_str))]
    
    return np.asarray(ra), np.asarray(dec), np.asarray(fib)



def plot_dNdz_selection(cn, w, iselect1, redz, area, dz=0.05, gold_eff=1, silver_eff=1, NoZ_eff=0.25, NoOII_eff=0.6,\
    gold_eff2=1, silver_eff2=1, NoZ_eff2=0.25, NoOII_eff2=0.6,\
    cn2=None, w2=None, iselect2=None, redz2=None, plot_total=True, fname="dNdz.png", color1="black", color2="red", color_total="green",\
     label1="Selection 1", label2="Selection 2", label_total="DEEP2 Total", wNoOII=0.1, wNoZ=0.5, lw=1.5, \
     label_np1="nP=1", color_np1="blue", plot_np1 = True):
    """
    Given class number (cn), mask (iselect1), weights (w), redshifts, class efficiencies, plot the redshift
    histogram. 

    dz: Histogram binwidth
    **_eff: Gold and Silver are NOT always equal to one. NoZ and NoOII are objects wtih no redshift
        in DEEP2 but are guessed to have efficiency of about 0.25.
    **_eff2: The efficiencies for the second set.
    iselect2: If not None, used as another set of mask to plot dNdz histogram.
    plot_total: Plots total.
    fname: Saves in fname.
    color1: iselect1 color
    color2: iselect2 color
    color_total: total color
    label1, label2, lbael_total: Labels
    """

    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20

    if plot_total:
        ibool = np.logical_or((cn==0),(cn==1)) 
        plt.hist(redz[ibool], bins = np.arange(0.6,1.7,dz), weights=w[ibool]/area,\
                 histtype="step", color=color_total, label=label_total, lw=lw)

        # NoOII:
        ibool = (cn==3) 
        N_NoOII = NoOII_eff*w[ibool].sum();
        plt.bar(left=0.7, height =N_NoOII/(wNoOII/dz), width=wNoOII, bottom=0., alpha=0.5,color=color_total, \
                edgecolor =color_total, label=label_total+" NoOII (Proj.)", hatch="*")
        # NoZ:
        ibool = (cn==5) 
        N_NoZ = NoZ_eff*w[ibool].sum();
        plt.bar(left=1.4, height =N_NoZ/(wNoZ/dz), width=wNoZ, bottom=0., alpha=0.5,color=color_total, \
                edgecolor =color_total, label=label_total+" NoZ (Proj.)")


    if iselect2 is not None:
        # If the new cn, w and redz are given, then use those values. Else, use first set.
        if cn2 is None:
            redz2 = np.copy(redz)
            cn2 = np.copy(cn)
            w2 = np.copy(w)
        # appropriately weighing the objects.
        w_select2 = np.copy(w2)
        w_select2[cn2==0] *= gold_eff2
        w_select2[cn2==1] *= silver_eff2
        w_select2[cn2==3] *= NoOII_eff2
        w_select2[cn2==5] *= NoZ_eff2

        ibool = np.logical_or((cn2==0),(cn2==1)) & iselect2
        plt.hist(redz2[ibool], bins = np.arange(0.6,1.7,dz), weights=w_select2[ibool]/area,\
                 histtype="step", color=color2, label=label2, lw=lw)

        # NoOII:
        ibool = (cn2==3) & iselect2
        N_NoOII = w_select2[ibool].sum();
        plt.bar(left=0.7, height =N_NoOII/(wNoOII/dz), width=wNoOII, bottom=0., alpha=0.5,color=color2, \
                edgecolor =color2, label=label2+ " NoOII (Proj.)", hatch="*")
    
        plt.plot([0.7, 0.7+wNoOII], [N_NoOII/(wNoOII/dz)/NoOII_eff2, N_NoOII/(wNoOII/dz)/NoOII_eff2], color=color2, linewidth=2.0, ls="--")


        # NoZ:
        ibool = (cn2==5) & iselect2
        N_NoZ = w_select2[ibool].sum();
        plt.bar(left=1.4, height =N_NoZ/(wNoZ/dz), width=wNoZ, bottom=0., alpha=0.5,color=color2, \
                edgecolor =color2, label=label2+" NoZ (Proj.)")         

        plt.plot([1.4, 1.4+wNoZ], [N_NoZ/(wNoZ/dz)/NoZ_eff2, N_NoZ/(wNoZ/dz)/NoZ_eff2], color=color2, linewidth=2.0, ls="--")

    # Selection 1.
    # appropriately weighing the objects.
    w_select1 = np.copy(w)
    w_select1[cn==0] *= gold_eff
    w_select1[cn==1] *= silver_eff
    w_select1[cn==3] *= NoOII_eff
    w_select1[cn==5] *= NoZ_eff

    ibool = np.logical_or((cn==0),(cn==1)) & iselect1 # Total
    plt.hist(redz[ibool], bins = np.arange(0.6,1.7,dz), weights=w_select1[ibool]/area,\
             histtype="step", color=color1, label=label1, lw=lw)

    # NoOII:
    ibool = (cn==3) & iselect1
    N_NoOII = w_select1[ibool].sum();
    plt.bar(left=0.7, height =N_NoOII/(wNoOII/dz), width=wNoOII, bottom=0., alpha=0.5,color=color1, \
            edgecolor =color1, label=label1+" NoOII (Proj.)", hatch="*")

    plt.plot([0.7, 0.7+wNoOII], [N_NoOII/(wNoOII/dz)/NoOII_eff, N_NoOII/(wNoOII/dz)/NoOII_eff], color=color1, linewidth=2.0, ls="--")

    # NoZ:
    ibool = (cn==5) & iselect1
    N_NoZ = w_select1[ibool].sum();
    plt.bar(left=1.4, height =N_NoZ/(wNoZ/dz), width=wNoZ, bottom=0., alpha=0.5, color=color1, \
            edgecolor =color1, label=label1+" NoZ (Proj.)")

    plt.plot([1.4, 1.4+wNoZ], [N_NoZ/(wNoZ/dz)/NoZ_eff, N_NoZ/(wNoZ/dz)/NoZ_eff], color=color1, linewidth=2.0, ls="--")

    # Plotting np=1 line
    if plot_np1:
        X,Y = np1_line(dz)
        plt.plot(X,Y, color=color_np1, label=label_np1, lw=lw*2., ls="-.")

 
    plt.xlim([0.5,1.4+wNoZ+0.1])
    plt.legend(loc="upper right", fontsize=15)  
    ymax=260
    if plot_total:
        ymax = 450
    plt.ylim([0,ymax])
    # plt.legend(loc="upper left")
    plt.xlabel("Redshift z", fontsize=20)
    plt.ylabel("dN/d(%.3fz) per sq. degs."%dz, fontsize=20)
    plt.savefig(fname, bbox_inches="tight", dpi=400)
    # plt.show()
    plt.close()


def np1_line(dz=0.5):
    """
    Given the binwidth dz, return np=1 line.
    """
    X, Y = np.asarray([[0.14538014092363039, 1.1627906976744384],
    [0.17035196758073518, 2.906976744186011],
    [0.20560848729069203, 5.8139534883720785],
    [0.2731789775637742, 10.465116279069775],
    [0.340752313629068, 15.697674418604663],
    [0.4083256496943619, 20.930232558139494],
    [0.4729621281972476, 26.16279069767444],
    [0.5405354642625415, 31.395348837209326],
    [0.6081088003278353, 36.62790697674416],
    [0.6756821363931291, 41.860465116279045],
    [0.7403214606882265, 47.67441860465118],
    [0.8078919509613086, 52.32558139534882],
    [0.8754624412343909, 56.97674418604652],
    [0.9430357772996848, 62.209302325581405],
    [1.0106034217805555, 66.27906976744185],
    [1.0811107696160458, 70.93023255813955],
    [1.1486784140969166, 75],
    [1.2162432127855753, 78.48837209302326],
    [1.2867448690366428, 81.97674418604649],
    [1.3543096677253015, 85.46511627906978],
    [1.4248084781841568, 88.37209302325581],
    [1.4953072886430125, 91.27906976744185],
    [1.5687401108720649, 93.6046511627907],
    [1.6392389213309202, 96.51162790697674],
    [1.7097320402053522, 98.2558139534884],
    [1.7802280048719963, 100.58139534883719],
    [1.8507211237464292, 102.32558139534885],
    [1.9212113968286495, 103.48837209302326],
    [1.9917045157030815, 105.23255813953489]]).T 

    return X, Y*dz/0.1



def FDR_cut(grz):
    """
    Given a list [g,r,z] magnitudes, apply the cut and return an indexing boolean vector.
    """
    g,r,z=grz; yrz = (r-z); xgr = (g-r)
    ibool = (r<23.4) & (yrz>.3) & (yrz<1.6) & (xgr < (1.15*yrz)-0.15) & (xgr < (1.6-1.2*yrz))
    return ibool


def plot_grz_class(grz, cn, weight, area, mask=None, pick=None,fname=None,pt_size=0.5):
    """
    Given [g,r,z] list, cn, weight of objects in a catalog and a particular class number and area, 
    plot the selected one in its color.
    
    fname convention:
    
    cc-(grz or grzperp)-(mag)(lim)-(cn)(cname)-(mask1)-(mask2)-...
    """
    global colors
    global cnames
    bnd_lw =2
    
    # Unpack the colors.
    g,r,z=grz; xrz = (r-z); ygr = (g-r)
    if mask is not None:
        xrz = xrz[mask]
        ygr = ygr[mask]
        cn = cn[mask]
        weight = weight[mask]
    
    fig = plt.figure(figsize=(5,5))

    if pick is None:
        plt.scatter(xrz, ygr,c="black",s=pt_size, edgecolors="none")
    else:
        plt.scatter(xrz[cn==pick],ygr[cn==pick], c=colors[pick],s=pt_size*6, edgecolors="none", marker="s")
        raw = np.sum(cn==pick)
        if pick <6:
            density = np.sum(weight[cn==pick])/area
        else:
            density = np.sum(cn==pick)/area
        title_str = "%s: Raw=%d, Density=%d" %(cnames[pick],raw, density)
        plt.title(title_str,fontsize=20)

    # FDR boundary practice:
    plt.plot( [0.3, 0.30], [-4, 0.195],'k-', lw=bnd_lw, c="blue")
    plt.plot([0.3, 0.745], [0.195, 0.706], 'k-', lw=bnd_lw, c="blue")
    plt.plot( [0.745, 1.6], [0.706, -0.32],'k-', lw=bnd_lw, c="blue")
    plt.plot([1.6, 1.6], [-0.32, -4],'k-', lw=bnd_lw, c="blue")
    # Broad
#     plt.plot(xbroad,ybroad, linewidth=bnd_lw, c='blue')
    # Decoration
    plt.xlabel("$r-z$",fontsize=20)
    plt.ylabel("$g-r$",fontsize=20)
    plt.axis("equal")
    plt.axis([-.5, 2.0, -.5, 2.0])
    if fname is not None:
#         plt.savefig(fname+".pdf", bbox_inches="tight",dpi=200)
        plt.savefig(fname+".png", bbox_inches="tight",dpi=200)    
    # plt.show()
    plt.close()

def plot_grzflux_class(grzflux, cn, weight, area, mask=None, pick=None,fname=None,pt_size=0.5, show_plot=False, \
    xmin=0, xmax=100, ymin=0, ymax=100):
    """
    Given [gflux, rflux, zflux] list, cn, weight of objects in a catalog and a particular class number and area, 
    plot the selected one in its color.
    
    fname convention:
    
    cc-(grz or grzperp)-(mag)(lim)-(cn)(cname)-(mask1)-(mask2)-...
    """
    global colors
    global cnames
    bnd_lw =2
    
    # Unpack the colors.
    gflux,rflux,zflux=grzflux; xrz = zflux/rflux; ygr = rflux/gflux
    if mask is not None:
        xrz = xrz[mask]
        ygr = ygr[mask]
        cn = cn[mask]
        weight = weight[mask]
    
    fig = plt.figure(figsize=(5,5))

    if pick is None:
        plt.scatter(xrz, ygr,c="black",s=pt_size, edgecolors="none")
    else:
        plt.scatter(xrz[cn==pick],ygr[cn==pick], c=colors[pick],s=pt_size*6, edgecolors="none", marker="s")
        raw = np.sum(cn==pick)
        if pick <6:
            density = np.sum(weight[cn==pick])/area
        else:
            density = np.sum(cn==pick)/area
        title_str = "%s: Raw=%d, Density=%d" %(cnames[pick],raw, density)
        plt.title(title_str,fontsize=20)

    # # FDR boundary practice:
    # plt.plot( [0.3, 0.30], [-4, 0.195],'k-', lw=bnd_lw, c="blue")
    # plt.plot([0.3, 0.745], [0.195, 0.706], 'k-', lw=bnd_lw, c="blue")
    # plt.plot( [0.745, 1.6], [0.706, -0.32],'k-', lw=bnd_lw, c="blue")
    # plt.plot([1.6, 1.6], [-0.32, -4],'k-', lw=bnd_lw, c="blue")
    # Broad
#     plt.plot(xbroad,ybroad, linewidth=bnd_lw, c='blue')
    # Decoration
    plt.xlabel("$r-z$ flux ratio",fontsize=20)
    plt.ylabel("$g-r$ flux ratio",fontsize=20)
    plt.axis("equal")
    plt.axis([xmin, xmax, ymin, ymax])
    if fname is not None:
#         plt.savefig(fname+".pdf", bbox_inches="tight",dpi=200)
        plt.savefig(fname+".png", bbox_inches="tight",dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def plot_fluxratio_class(ratio1, ratio2, cn, weight, area, mask=None, pick=None,fname=None,pt_size=0.5, show_plot=False, \
    xmin=0, xmax=100, ymin=0, ymax=100, xlabel="z-w1 flux", ylabel="g-r flux"):
    """
    Given [gflux, rflux, zflux] list, cn, weight of objects in a catalog and a particular class number and area, 
    plot the selected one in its color.
    
    fname convention:
    
    cc-(grz or grzperp)-(mag)(lim)-(cn)(cname)-(mask1)-(mask2)-...
    """
    global colors
    global cnames
    bnd_lw =2
    
    # Unpack the colors.
    if mask is not None:
        ratio1 = ratio1[mask]
        ratio2 = ratio2[mask]
        cn = cn[mask]
        weight = weight[mask]
    
    fig = plt.figure(figsize=(5,5))

    if pick is None:
        plt.scatter(ratio1, ratio2,c="black",s=pt_size, edgecolors="none")
    else:
        plt.scatter(ratio1[cn==pick],ratio2[cn==pick], c=colors[pick],s=pt_size*6, edgecolors="none", marker="s")
        raw = np.sum(cn==pick)
        if pick <6:
            density = np.sum(weight[cn==pick])/area
        else:
            density = np.sum(cn==pick)/area
        title_str = "%s: Raw=%d, Density=%d" %(cnames[pick],raw, density)
        plt.title(title_str,fontsize=20)

    # # FDR boundary practice:
    # plt.plot( [0.3, 0.30], [-4, 0.195],'k-', lw=bnd_lw, c="blue")
    # plt.plot([0.3, 0.745], [0.195, 0.706], 'k-', lw=bnd_lw, c="blue")
    # plt.plot( [0.745, 1.6], [0.706, -0.32],'k-', lw=bnd_lw, c="blue")
    # plt.plot([1.6, 1.6], [-0.32, -4],'k-', lw=bnd_lw, c="blue")
    # Broad
#     plt.plot(xbroad,ybroad, linewidth=bnd_lw, c='blue')
    # Decoration
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)
    plt.axis("equal")
    plt.axis([xmin, xmax, ymin, ymax])
    if fname is not None:
#         plt.savefig(fname+".pdf", bbox_inches="tight",dpi=200)
        plt.savefig(fname+".png", bbox_inches="tight",dpi=200)
    if show_plot:
        plt.show()
    plt.close()


def plot_grz_class_all(grz, cn, weight, area, mask=None, fname=None, pt_size1=0.5, pt_size2=0.3):
    """
    Given [g,r,z] list, cn, weight of objects in a catalog and a particular class number and area, 
    plot all the objects in their respective colors. 
    
    fname convention:
    
    cc-(grz or grzperp)-(mag)(lim)-cnAll-(mask1)-(mask2)-...
    """
    global colors
    global cnames
    bnd_lw =2
    
    # Unpack the colors.
    g,r,z=grz; xrz = (r-z); ygr = (g-r)
    if mask is not None:
        xrz = xrz[mask]
        ygr = ygr[mask]
        cn = cn[mask]
        weight = weight[mask]
    
    fig = plt.figure(figsize=(5,5))

    for i,e in enumerate(cnames):
        if i < 6:
            plt.scatter(xrz[cn==i], ygr[cn==i], c=colors[i],s=pt_size1, edgecolors="none", marker="s")
        elif i ==6:
            plt.scatter(xrz[cn==i], ygr[cn==i], c=colors[i],s=pt_size2, edgecolors="none", marker="s")
            

    # FDR boundary practice:
    plt.plot( [0.3, 0.30], [-4, 0.195],'k-', lw=bnd_lw, c="blue")
    plt.plot([0.3, 0.745], [0.195, 0.706], 'k-', lw=bnd_lw, c="blue")
    plt.plot( [0.745, 1.6], [0.706, -0.32],'k-', lw=bnd_lw, c="blue")
    plt.plot([1.6, 1.6], [-0.32, -4],'k-', lw=bnd_lw, c="blue")
    # Broad
#     plt.plot(xbroad,ybroad, linewidth=bnd_lw, c='blue')
    # Decoration
    plt.xlabel("$r-z$",fontsize=20)
    plt.ylabel("$g-r$",fontsize=20)
    plt.axis("equal")
    plt.axis([-.5, 2.0, -.5, 2.0])
    if fname is not None:
#         plt.savefig(fname+".pdf", bbox_inches="tight",dpi=200)
        plt.savefig(fname+".png", bbox_inches="tight",dpi=200)    
    # plt.show()
    plt.close()



def load_params_XD_fit(i,K,tag=""):
    fname = ("%d-params-fit-amps-glim24-K%d"+tag+".npy") %(i, K)
    amp = np.load(fname)
    fname = ("%d-params-fit-means-glim24-K%d"+tag+".npy") %(i, K)
    mean= np.load(fname)
    fname = ("%d-params-fit-covars-glim24-K%d"+tag+".npy") %(i, K)
    covar  = np.load(fname)
    return amp, mean, covar

def load_params_XD_init(i,K,tag=""):
    fname = ("%d-params-init-amps-glim24-K%d"+tag+".npy") %(i, K)
    amp = np.load(fname)
    fname = ("%d-params-init-means-glim24-K%d"+tag+".npy") %(i, K)
    mean= np.load(fname)
    fname = ("%d-params-init-covars-glim24-K%d"+tag+".npy") %(i, K)
    covar  = np.load(fname)
    return amp, mean, covar

def plot_XD_fit(ydata, weight, Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar, mask=None, fname=None, pt_size=5, show=False):
    """
    See the output.
    """
    bnd_lw =1.5
    
    # Unpack the colors.
    xrz = ydata[:,0]; ygr = ydata[:,1]
    if mask is not None:
        ygr = ygr[mask]
        xrz = xrz[mask]
    
    # Broad boundary
    # xbroad, ybroad = generate_broad()
    
    # Figure ranges
    grmin = -.5
    rzmin = -.5
    grmax = 2.5
    rzmax = 2.5
    # histogram binwidth
    bw = 0.05
    # Number of components/linewidth
    K = Sxamp_init.size
    elw = 1.5 # ellipse linewidth
    ea = 0.75 # ellipse transparency
    
    
    # Create figure 
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,14))

    # 1: Plot the dots and the initial and final component gaussians.
    ax1.scatter(xrz,ygr, c="black",s=pt_size, edgecolors="none")
    # Initial
    for i in range(K):
        cc.plot_cov_ellipse(Sxcovar_init[i], Sxmean_init[i], volume=.6827, ax=ax1, ec="red", a=ea, lw=elw/2.) #1-sig 
        cc.plot_cov_ellipse(Sxcovar_init[i], Sxmean_init[i], volume=.9545, ax=ax1, ec="red", a=ea, lw=elw/2.) #2-sig
        cc.plot_cov_ellipse(Sxcovar[i], Sxmean[i], volume=.6827, ax=ax1, ec="blue", a=ea, lw=elw)#1-sig 
        cc.plot_cov_ellipse(Sxcovar[i], Sxmean[i], volume=.9545, ax=ax1, ec="blue", a=ea, lw=elw)#2-sig
    
    # FDR boundary:
    ax1.plot( [0.3, 0.30], [-4, 0.195],'k-', lw=bnd_lw, c="red")
    ax1.plot([0.3, 0.745], [0.195, 0.706], 'k-', lw=bnd_lw, c="red")
    ax1.plot( [0.745, 1.6], [0.706, -0.32],'k-', lw=bnd_lw, c="red")
    ax1.plot([1.6, 1.6], [-0.32, -4],'k-', lw=bnd_lw, c="red")    
    # Decoration
    ax1.set_xlabel("$r-z$",fontsize=18)
    ax1.set_ylabel("$g-r$",fontsize=18)
    ax1.axis("equal")
    ax1.axis([rzmin, rzmax, grmin, grmax])

    # 2: Histogram in r-z
    ax2.hist(ygr, bins = np.arange(grmin, grmax+0.9*bw, bw), weights=weight, normed=True, color="black", histtype="step", orientation="horizontal")
    # Gaussian components
    xvec = np.arange(-3,3,0.01) # vector range
    sum_init = np.zeros_like(xvec) # place holder for gaussians.
    sum_fit = np.zeros_like(xvec)
    for i in range(K):
        yvec = Sxamp_init[i]*stats.multivariate_normal.pdf(xvec, mean=Sxmean_init[i][1], cov=Sxcovar_init[i][1,1])        
        sum_init += yvec
        ax2.plot(yvec, xvec,lw=elw, color ="red", alpha=0.5)
        yvec = Sxamp[i]*stats.multivariate_normal.pdf(xvec, mean=Sxmean[i][1], cov=Sxcovar[i][1,1])        
        sum_fit += yvec
        ax2.plot(yvec, xvec,lw=elw, color ="blue", alpha=0.5)
    ax2.plot(sum_init, xvec,lw=elw*1.5, color ="red", alpha=1.)
    ax2.plot(sum_fit, xvec,lw=elw*1.5, color ="blue", alpha=1.)
    # Deocration     
    ax2.set_ylabel("$g-r$",fontsize=18)
    ax2.set_xlabel("Normalized density", fontsize=20)
    ax2.set_ylim([grmin, grmax])
    
    
    # 3: Histogram in g-r
    ax3.hist(xrz, bins = np.arange(grmin, grmax+0.9*bw, bw), weights=weight, normed=True, color="black", histtype="step")
    # Gaussian components
    xvec = np.arange(-3,3,0.01) # vector range
    sum_init = np.zeros_like(xvec) # place holder for gaussians.
    sum_fit = np.zeros_like(xvec)
    for i in range(K):
        yvec = Sxamp_init[i]*stats.multivariate_normal.pdf(xvec, mean=Sxmean_init[i][0], cov=Sxcovar_init[i][0,0])        
        sum_init += yvec
        ax3.plot(xvec,yvec,lw=elw, color ="red", alpha=0.5)
        yvec = Sxamp[i]*stats.multivariate_normal.pdf(xvec, mean=Sxmean[i][0], cov=Sxcovar[i][0,0])        
        sum_fit += yvec
        ax3.plot(xvec, yvec,lw=elw, color ="blue", alpha=0.5)
    ax3.plot(xvec,sum_init, lw=elw*1.5, color ="red", alpha=1.)
    ax3.plot(xvec,sum_fit,lw=elw*1.5, color ="blue", alpha=1.)    
    # Decoration
    ax3.set_xlabel("$r-z$",fontsize=18)
    ax3.set_ylabel("Normalized density", fontsize=20)
    ax3.set_xlim([rzmin, rzmax])
    
    
    # 4: Plot the dots and the isocontours of GMM at 2, 10, 50, 90, 98
    ax4.scatter(xrz,ygr, c="black",s=pt_size, edgecolors="none")
    # Plot isocontours
    magmin = min(grmin, rzmin)
    magmax = max(grmax, rzmax)
    vec = np.linspace(magmin,magmax, num=1e3,endpoint=True)
    X,Y = np.meshgrid(vec, vec) # grid of point
    Z = summed_gm(np.transpose(np.array([Y,X])), Sxmean, Sxcovar, Sxamp) # evaluation of the function on the grid
    Xrange = Yrange = [magmin,magmax]; # Estimating density levels.
    cvs = [0.98, 0.90, 0.50, 0.10, 0.02] # contour levels
    cvsP =inverse_cdf_gm(cvs,Xrange, Yrange, Sxamp, Sxcovar,  Sxmean, gridspacing=0.5e-2,gridnumber = 1e3)
    ax4.contour(X,Y,Z,cvsP,linewidths=1.5, colors=["black", "blue", "red", "orange", "yellow"])     
    # FDR boundary:
    ax4.plot( [0.3, 0.30], [-4, 0.195],'k-', lw=bnd_lw, c="red")
    ax4.plot([0.3, 0.745], [0.195, 0.706], 'k-', lw=bnd_lw, c="red")
    ax4.plot( [0.745, 1.6], [0.706, -0.32],'k-', lw=bnd_lw, c="red")
    ax4.plot([1.6, 1.6], [-0.32, -4],'k-', lw=bnd_lw, c="red")    
    # Decoration
    ax4.set_xlabel("$r-z$",fontsize=18)
    ax4.set_ylabel("$g-r$",fontsize=18)
    ax4.axis("equal")
    ax4.axis([rzmin, rzmax, grmin, grmax])
    
    
    if fname is not None:
#         plt.savefig(fname+".pdf", bbox_inches="tight",dpi=200)
        plt.savefig(fname+".png", bbox_inches="tight",dpi=200)    
    if show:
        plt.show()
    plt.close()
    
def XD_gr_rz_fit(ydata, ycovar, weight, niter, K, maxsnm=True, subsample = False, fixamp = None, snm=0, init_var=0.5**2, w_reg = 0.05**2):
    """
    Given the appropriately formmated data, make fits and return the best
    fit parameters (and the corresponding initial values).
    
    K: Number of components
    """
    # If subsmaple is true and the number of data points is greater than M ~ 2,000 than subsample before proceeding.
    M = 3000
    if (ydata.shape[0] > M) & subsample:
        a = np.arange(0,M,1, dtype=int)
        ibool = np.random.choice(a, size=M, replace=False, p=None)
        ydata = ydata[ibool]
        ycovar = ycovar[ibool]
        weight = weight[ibool]

    # Place holder for the log-likelihood
    loglike = large_random_constant
    best_loglike = large_random_constant

    # Make niter number of fits
    for i in range(niter):
        if (i%2==0) & (niter<=25):
            print(i)
        if (i%10==0) & (niter>25):
            print(i)            
            
        # Get initial condition
        xamp_init, xmean_init, xcovar_init = XD_init(K, ydata, init_var)

        # Copy the initial condition.
        xamp = np.copy(xamp_init); xmean = np.copy(xmean_init); xcovar = np.copy(xcovar_init)

        # XD fit
        loglike = XD.extreme_deconvolution(ydata, ycovar, xamp, xmean, xcovar, weight=weight, tol=1e-06, w=w_reg, maxsnm=maxsnm, fixamp=fixamp, splitnmerge=snm)

        if loglike > best_loglike:
            best_loglike = loglike
            Sxamp_init, Sxmean_init, Sxcovar_init = xamp_init, xmean_init, xcovar_init
            Sxamp, Sxmean, Sxcovar = xamp, xmean, xcovar

    
    return Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar    



def sample_GMM(Sxamp,Sxmean, Sxcovar, ycovar):
    """
    Return a sample based on the GMM input.
    """
    N = ycovar.shape[0] # Number of data points. 
    sample = []
    # For each data point, generate a sample based on the specified GMM. 
    for i in range(N):
        sample.append(sample_GMM_generate(Sxamp,Sxmean, Sxcovar, ycovar[i]))
    sample = np.asarray(sample)
#     print sample.shape, sample
    xgr_sample, yrz_sample = sample[:,0], sample[:,1]
    return xgr_sample, yrz_sample

def sample_GMM_generate(Sxamp,Sxmean, Sxcovar, cov):
    """
    sample from a gaussian mixture
    """
    # Number of components.
    K = Sxamp.size
    if K == 1:
#         print(Sxmean[0], (Sxcovar+cov)
        one_sample = np.random.multivariate_normal(Sxmean[0], (Sxcovar+cov)[0], size=1)[0]
        return one_sample
    
    # Choose from the number based on multinomial
    m = np.where(np.random.multinomial(1,Sxamp)==1)[0][0]
    # Draw from the m-th gaussian.
    one_sample = np.random.multivariate_normal(Sxmean[m], Sxcovar[m]+cov, size=1)[0]
    return one_sample

def plot_XD_fit_K(ydata, ycovar, Sxamp, Sxmean, Sxcovar, fname=None, pt_size=5, mask=None, show=False):
    """
    Used for model selection.
    """
    bnd_lw = 1.
    # Unpack the colors.
    xgr = ydata[:,0]; yrz = ydata[:,1]
    if mask is not None:
        yrz = yrz[mask]
        xgr = xgr[mask]
        
    # # Broad boundary
    # xbroad, ybroad = generate_broad()
    # Figure ranges
    grmin = -1.
    rzmin = -.75
    grmax = 2.5
    rzmax = 2.75
    
    # Create figure 
    f, axarr = plt.subplots(2, 2, figsize=(14,14))

    # First panel is the original.
    axarr[0,0].scatter(xgr,yrz, c="black",s=pt_size, edgecolors="none")
    # FDR boundary:
    axarr[0,0].plot([-4, 0.195], [0.3, 0.30], 'k-', lw=bnd_lw, c="red")
    axarr[0,0].plot([0.195, 0.706],[0.3, 0.745], 'k-', lw=bnd_lw, c="red")
    axarr[0,0].plot([0.706, -0.32], [0.745, 1.6], 'k-', lw=bnd_lw, c="red")
    axarr[0,0].plot([-0.32, -4],[1.6, 1.6], 'k-', lw=bnd_lw, c="red")
    # # Broad
    # axarr[0,0].plot(xbroad,ybroad, linewidth=bnd_lw, c='blue')
    # Decoration
    axarr[0,0].set_xlabel("$g-r$",fontsize=18)
    axarr[0,0].set_ylabel("$r-z$",fontsize=18)
    axarr[0,0].set_title("Data",fontsize=20)        
    axarr[0,0].axis("equal")
    axarr[0,0].axis([grmin, grmax, rzmin, rzmax]) 
    
    
    # The remaining three are simulation based on the fit.
    sim_counter = 1
    for i in range(1,4):
        xgr_sample, yrz_sample = sample_GMM(Sxamp,Sxmean, Sxcovar, ycovar)
        axarr[i//2, i%2].scatter(xgr_sample,yrz_sample, c="black",s=pt_size, edgecolors="none")
        # FDR boundary:
        axarr[i//2, i%2].plot([-4, 0.195], [0.3, 0.30], 'k-', lw=bnd_lw, c="red")
        axarr[i//2, i%2].plot([0.195, 0.706],[0.3, 0.745], 'k-', lw=bnd_lw, c="red")
        axarr[i//2, i%2].plot([0.706, -0.32], [0.745, 1.6], 'k-', lw=bnd_lw, c="red")
        axarr[i//2, i%2].plot([-0.32, -4],[1.6, 1.6], 'k-', lw=bnd_lw, c="red")
        # Broad
        # axarr[i//2, i%2].plot(xbroad,ybroad, linewidth=bnd_lw, c='blue')
        # Decoration
        axarr[i//2, i%2].set_xlabel("$g-r$",fontsize=18)
        axarr[i//2, i%2].set_ylabel("$r-z$",fontsize=18)
        axarr[i//2, i%2].set_title("Simulation %d" % sim_counter,fontsize=20); sim_counter+=1
        axarr[i//2, i%2].axis("equal")
        axarr[i//2, i%2].axis([grmin, grmax, rzmin, rzmax])     

    if fname is not None:
#         plt.savefig(fname+".pdf", bbox_inches="tight",dpi=200)
        plt.savefig(fname+".png", bbox_inches="tight",dpi=200)    
    if show:
        plt.show()
    plt.close()    

def save_params(Sxamp_init, Sxmean_init, Sxcovar_init, Sxamp, Sxmean, Sxcovar, i, K, tag=""):
    fname = ("%d-params-fit-amps-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxamp)
    fname = ("%d-params-fit-means-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxmean)
    fname = ("%d-params-fit-covars-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxcovar)
    # Initi parameters
    fname = ("%d-params-init-amps-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxamp_init)
    fname = ("%d-params-init-means-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxmean_init)
    fname = ("%d-params-init-covars-glim24-K%d"+tag) %(i, K)
    np.save(fname, Sxcovar_init)
    return

def amp_init(K):
    return np.ones(K,dtype=np.float)/np.float(K)

def mean_init(K, ydata):
    S = np.random.randint(low=0,high=ydata.shape[0],size=K)
    return ydata[S]

def covar_init(K, init_var):
    covar = np.zeros((K, 2,2))
    for i in range(K):
        covar[i] = np.diag((init_var, init_var))
    return covar 

def XD_init(K, ydata, init_var):
    xamp_init = amp_init(K)
    # print xamp_init, xamp_init.shape
    xmean_init = mean_init(K, ydata)
    # print xmean_init, xmean_init.shape
    xcovar_init = covar_init(K, init_var)
    # print xcovar_init, xcovar_init.shape
    
    return xamp_init, xmean_init, xcovar_init






def grz2gr_rz(grz):
    return np.transpose(np.asarray([grz[0]-grz[1], grz[1]-grz[2]]))

def grz2rz_gr(grz):
    return np.transpose(np.asarray([grz[1]-grz[2], grz[0]-grz[1]]))    

def fvar2mvar(f, fivar):
    return (1.08574)**2/(f**2 * fivar)
    
def gr_rz_covariance(grzflux, grzivar):
    gflux = grzflux[0]
    rflux = grzflux[1]
    zflux = grzflux[2]
    givar = grzivar[0]
    rivar = grzivar[1]
    zivar = grzivar[2]
    
    gvar = fvar2mvar(gflux,givar)
    rvar = fvar2mvar(rflux,rivar)
    zvar = fvar2mvar(zflux,zivar)
    
    gr_rz_covar = np.zeros((gvar.size ,2,2))
    for i in range(gvar.size):
#         if i % 100 == 0:
#             print i
        gr_rz_covar[i] = np.asarray([[gvar[i]+rvar[i], rvar[i]],[rvar[i], rvar[i]+zvar[i]]])
    
    return gr_rz_covar

def rz_gr_covariance(grzflux, grzivar):
    gflux = grzflux[0]
    rflux = grzflux[1]
    zflux = grzflux[2]
    givar = grzivar[0]
    rivar = grzivar[1]
    zivar = grzivar[2]
    
    gvar = fvar2mvar(gflux,givar)
    rvar = fvar2mvar(rflux,rivar)
    zvar = fvar2mvar(zflux,zivar)
    
    rz_gr_covar = np.zeros((gvar.size ,2,2))
    for i in range(gvar.size):
#         if i % 100 == 0:
#             print i
        rz_gr_covar[i] = np.asarray([[rvar[i]+zvar[i], rvar[i]],[rvar[i], gvar[i]+rvar[i]]])
    
    return rz_gr_covar    


def pow_legend(params_pow):
    alpha, A = params_pow
    return r"$A=%.2f,\,\, \alpha=%.2f$" % (A, alpha)

def broken_legend(params_broken):
    alpha, beta, fs, phi = params_broken
    return r"$\alpha=%.2f, \,\, \beta=%.2f, \,\, f_i=%.2f, \,\, \phi=%.2f$" % (alpha, beta, fs, phi)


def broken_pow_phi_init(flux_centers, best_params_pow, hist,bw, fluxS):
    """
    Return initial guess for phi.
    """
    # selecting one non-zero bin
    c_S = 0;
    while c_S == 0:
        S = np.random.randint(low=0,high=flux_centers.size,size=1)
        f_S = flux_centers[S]
        c_S = hist[S]
        
    alpha = -best_params_pow[0]
    beta = best_params_pow[0]
    phi = c_S/broken_pow_law([alpha, beta, fluxS, 1.], f_S)/bw
    
    # phi
    return phi[0]


def pow_law(params, flux):
    A = params[1]
    alpha = params[0]
    return A* flux**alpha

def broken_pow_law(params, flux):
    alpha = params[0]
    beta = params[1]
    fs = params[2]
    phi = params[3]
    return phi/((flux/fs)**alpha+(flux/fs)**beta + 1e-12)

def pow_param_init(left_hist, left_f, right_hist, right_f, bw):
    """
    Return initial guess for the exponent and normalization.
    """
    # selecting non-zero bin one from left and one from right. 
    c_L = 0; c_R = 0
    while c_L==0 or c_R == 0 or c_L >= c_R:
        L = np.random.randint(low=0,high=left_hist.size,size=1)
        f_L = left_f[L]
        c_L = left_hist[L]
        R = np.random.randint(low=0,high=right_hist.size,size=1)
        f_R = right_f[R]
        c_R = right_hist[R]
#     print(L,R)
    # exponent
    alpha_init = np.log(c_L/np.float(c_R))/np.log(f_L/np.float(f_R))
    A_init = c_L/(f_L**alpha_init * bw)
    
    ans = np.zeros(2, dtype=np.float)
    ans[0] = alpha_init
    ans[1] = A_init
    return ans

def mag2flux(mag):
    return 10**(0.4*(22.5-mag))
        


def dNdm_fit(mag, weight, bw, magmin, magmax, area, niter = 5, cn2fit=0, pow_tol =1e-5, broken_tol=1e-2, fname=None, lw=1.5):
    """
    Given the magnitudes and the corresponding weight, and the parameters for the histogram, 
    return the best fit parameters for a power law and a broken power law.
    
    Note: This function could be much more modular. But for now I keep it as it is.
    """
    # Computing the histogram.
    bins = np.arange(magmin, magmax+bw*0.9, bw) # I am not sure what this is necessary but this works.
    if cn2fit<6:
        hist, bin_edges = np.histogram(mag, weights=weight/np.float(area), bins=bins)
    else: # If D2reject, then do not weight except for the area.
        hist, bin_edges = np.histogram(mag,weights=np.ones(mag.size)/np.float(area), bins=bins)        

    # Compute the median magnitude
    magmed = np.median(mag)

    # Compute bin centers. Left set and right set.
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2.
    ileft = bin_centers < magmed
    # left and right counts
    left_hist = hist[ileft]
    right_hist = hist[~ileft]
    # left and right flux
    left_f = mag2flux(bin_centers[ileft])
    right_f = mag2flux(bin_centers[~ileft])
    flux_centers = mag2flux(bin_centers)


    # Place holder for the best parameters
    best_params_pow = np.zeros(2,dtype=np.float) 

    # Empty list for the negative log-likelihood
    list_nloglike = []
    best_nloglike = -100.

    # Define negative total loglikelihood function given the histogram.
    def ntotal_loglike_pow(params):
        """
        Total log likelihood.
        """
        total_loglike = 0

        for i in range(flux_centers.size):
            total_loglike += stats.poisson.logpmf(hist[i].astype(int), pow_law(params, flux_centers[i])*bw)

        return -total_loglike

    # fit for niter times 
    print("Fitting power law")
    counter = 0
    while counter < niter:
        # Generate initial parameters
        init_params = pow_param_init(left_hist, left_f, right_hist, right_f, bw)
    #     print(init_param)

        # Optimize the parameters.
        res = opt.minimize(ntotal_loglike_pow, init_params,tol=pow_tol,method="Nelder-Mead" )
        if res["success"]:
            counter+=1
            if counter % 2 == 0:
                print(counter)
#             print(counter)
    #         print(res["x"])
            fitted_params = res["x"]

            # Calculate the negative total likelihood
            nloglike = ntotal_loglike_pow(fitted_params)
            list_nloglike.append(nloglike)

            # If loglike is the highest among seen, then update the parameters.
            if nloglike > best_nloglike:
                best_nloglike = nloglike
                best_params_pow = fitted_params

#     print(best_params_pow)



    # Place holder for the best parameters
    best_params_broken = np.zeros(4,dtype=np.float) 

    # Empty list for the negative log-likelihood
    list_nloglike = []
    best_nloglike = -100.

    # Define negative total loglikelihood function given the histogram.
    def ntotal_loglike_broken(params):
        """
        Total log likelihood for broken power law.
        """
        total_loglike = 0

        for i in range(flux_centers.size):
            total_loglike += stats.poisson.logpmf(hist[i].astype(int), broken_pow_law(params, flux_centers[i])*bw)

        return -total_loglike


    # fit for niter times 
    print("Fitting broken power law")
    counter = 0
    while counter < niter:
        # Generate initial parameters
        phi = broken_pow_phi_init(flux_centers, best_params_pow, hist, bw, mag2flux(magmed))
        alpha = -best_params_pow[0]
        beta = best_params_pow[0]
        init_params = [alpha, beta,mag2flux(magmed), phi]
    #     print(init_params)

        # Optimize the parameters.
        res = opt.minimize(ntotal_loglike_broken, init_params,tol=broken_tol,method="Nelder-Mead" )
        if res["success"]:
            counter+=1
            if counter % 2 == 0:
                print(counter)            
    #         print(res["x"])
            fitted_params = res["x"]

            # Calculate the negative total likelihood
            nloglike = ntotal_loglike_broken(fitted_params)
            list_nloglike.append(nloglike)

            # If loglike is the highest among seen, then update the parameters.
            if nloglike > best_nloglike:
                best_nloglike = nloglike
                best_params_broken = fitted_params

#     print(best_params_broken)

    # power law fit
    xvec = np.arange(magmin, magmax, 1e-3)
    yvec = pow_law(best_params_pow, mag2flux(xvec))*np.float(bw)
    pow_str = pow_legend(best_params_pow)
    plt.plot(xvec,yvec, c = "red", label = pow_str, lw=lw)
    # broken  power law fit
    yvec = broken_pow_law(best_params_broken, mag2flux(xvec))*np.float(bw)
    broken_str = broken_legend(best_params_broken)
    plt.plot(xvec,yvec, c = "blue", label=broken_str, lw=lw)
    # hist
    plt.bar(bin_edges[:-1], hist, width=bw, alpha=0.5, color="g")
    # deocration
    plt.legend(loc="upper left")
    plt.xlim([magmin,magmax])
    plt.xlabel(r"Mag")
    plt.ylabel(r"Number per %.2f mag bin"%bw)
    if fname is not None:
        plt.savefig(fname+".png", bbox_inches="tight", dpi=400)
    # plt.show()
    plt.close()

    return best_params_pow, best_params_broken



def combine_grz(list1,list2,list3=None):
    """
    Convenience function for combining two or three sets data in a list.
    """
    if list3 is not None:
        g = np.concatenate((list1[0], list2[0], list3[0]))
        r = np.concatenate((list1[1], list2[1], list3[1]))
        z = np.concatenate((list1[2], list2[2], list3[2]))
    else:
        g = np.concatenate((list1[0], list2[0]))
        r = np.concatenate((list1[1], list2[1]))
        z = np.concatenate((list1[2], list2[2]))

    return [g, r,z]


def true_false_fraction(ibool):
    """
    Given boolean index array count true and false proportion and print.
    """
    counts = np.bincount(ibool)
    tot = np.sum(counts).astype(float)
    print("True: %d (%.4f)| False: %d (%.4f)" % (counts[1], counts[1]/tot, counts[0], counts[0]/tot))
    return [counts[1], counts[1]/tot, counts[0], counts[0]/tot]


def load_cn(fits):
    return fits["cn"].astype(int)


def load_DEEP2matched(table):
    return table["DEEP2_matched"][:]    


def window_mask(ra, dec, w_fname):
    """
    Given the ra,dec of objects and a window function file name, the function 
    returns an boolean array whose elements equal True when the corresponding 
    objects lie within regions where the map is positive.
    
    Note: the windowf.**.fits files were not set up in a convenient format 
    so below I had to perform a simple but tricky affine transformation. 
    """
    
    # Import the window map and get pixel limits.
    window = fits.open(w_fname)[0].data.T # Note that transpose.
    px_lim = window.shape[1]
    py_lim = window.shape[0]      
    
    # Creating WCS object for the window.  
    w = WCS(w_fname)
    
    # Convert ra/dec to pixel values and round.
    px, py = w.wcs_world2pix(ra, dec, 0)
    px_round = np.round(py).astype(int)
    py_round = np.round(px).astype(int)
  
    # Creating the array.
    idx = np.zeros(px_round.size, dtype=bool)
    for i in range(px.size):
        if (px_round[i]>=0) and (px_round[i]<px_lim) and (py_round[i]>=0) and (py_round[i]<py_lim): # Check if the object lies within the window frame. 
            if (window[py_round[i],px_round[i]]>0): # Check if the object is in a region where there is spectroscopy.
                idx[i] = True
    
    return idx


def est_spec_area(w_fname):
    """
    The following function estiamtes the spectroscopic area given a window 
    function file.
    """
    # Creating WCS object for the window.  
    w = WCS(w_fname)
    
    # Importing the window
    window = fits.open(w_fname)[0].data
    px_lim = window.shape[0]
    py_lim = window.shape[1]
   
    
    # Convert ra/dec to pixel values and round.
    ra, dec = w.wcs_pix2world([0,px_lim], [0,py_lim], 0)
    
    # Calculating the area
    area = (ra[1]-ra[0])*(dec[1]-dec[0])
    
    # Calculating the fraction covered by spectroscopy
    frac = (window>0).sum()/(px_lim*py_lim+1e-12)
    
    return frac*area
        
        
    
def import_zcat(z_fname):
    """
    Given DEEP2 redshift catalog filename, import and return relevant fields.
    """
    data = fits.open(z_fname)[1].data
    
    return data["OBJNO"], data["RA"], data["DEC"], data["OII_3727"], data["OII_3727_ERR"], data["ZHELIO"], data["ZHELIO_ERR"], data["ZQUALITY"], data["TARG_WEIGHT"]
    


def match_objno(objno1, objno2):
    """
    Given two objno arrays, return idx of items that match. This algorithm can be slow, O(N^2), but it should work.
    The input arrays have to be a set, meaning a list of unique items.
    """    
    global large_random_constant
    # Finding the intersection
    intersection = np.intersect1d(objno1, objno2)
    print("# of elements in intersection: %d"% intersection.size)
    
    # Creating placeholders for idx's to be returned.
    idx1 = np.ones(intersection.size,dtype=int)*large_random_constant
    idx2 = np.ones(intersection.size,dtype=int)*large_random_constant
    
    # Creating objno1, objno2 copies with integer tags before sorting.
    
    objno1_tagged = np.rec.fromarrays((objno1, range(objno1.size)),dtype=[('id', int), ('tag', int)])
    objno2_tagged = np.rec.fromarrays((objno2, range(objno2.size)),dtype=[('id', int), ('tag', int)])
    
    # Sorting according id
    objno1_tagged = np.sort(objno1_tagged, axis=0, order="id")
    objno2_tagged = np.sort(objno2_tagged, axis=0, order="id")
    
    # tags
    tags1 = objno1_tagged["tag"]
    tags2 = objno2_tagged["tag"]
    
    # values
    objno1_vals = objno1_tagged["id"]    
    objno2_vals = objno2_tagged["id"]
        
    # For each id in the intersection set, find the corresponding indices in objno1 and objno2 and save. 
    for i,e in enumerate(intersection):
        idx1[i] = tags1[np.searchsorted(objno1_vals,e)]
        idx2[i] = tags2[np.searchsorted(objno2_vals,e)]
    
    return idx1, idx2


def pcat_append(pcat, new_col, col_name, idx1, idx2):
    """
    Given DEEP2 pcat recarray and a pair of a field column, and a field name,
    append the new field column to the recarray using OBJNO-matched values and name the appended column the field name.
    Must provide appropriate idx values for both pcats and additional catalogs.
    """
    global large_random_constant
    new_col_sorted = np.ones(pcat.shape[0])*large_random_constant
    new_col_sorted[idx1] = new_col[idx2]
    
    new_pcat = rec.append_fields(pcat, col_name, new_col_sorted, dtypes=new_col_sorted.dtype, usemask=False, asrecarray=True)
    return new_pcat
    


def generate_class_col(pcat):
    """
    Given a pcat array with required fields, produce a column that classify objects into different classes.
    """
    # Extracting columns
    OII = pcat["OII_3727"]*1e17
    Z = pcat["RED_Z"]
    ZQUALITY = pcat["ZQUALITY"]
    OII_ERR = pcat["OII_3727_ERR"]
    BRIcut = pcat["BRI_cut"]
    
    
    # Placeholder for the class column.
    class_col = np.ones(pcat.shape[0],dtype=int)*large_random_constant
    
    # Gold, CN=0: OII>8, Z in [1.1, 1.6]
    ibool = (OII>8) & (Z>1.1) & (Z<1.6)  & (BRIcut==1) & (ZQUALITY>=3) & (OII_ERR>0) 
    class_col[ibool] = 0
    
    # Silver, CN=1: OII>8, Z in [0.6, 1.1]
    ibool = (OII>8) & (Z>0.6) & (Z<1.1) & (OII_ERR>0)  & (BRIcut==1) & (ZQUALITY>=3)
    class_col[ibool] = 1

    # LowOII, CN=2: OII<8, Z in [0.6, 1.6]
    ibool =  (OII<8) & (Z>0.6) & (Z<1.6)  & (OII_ERR>0) & (ZQUALITY>=3) & (BRIcut==1)
    class_col[ibool] = 2
    # Note that many ELG objects were assigned negative OII, which are unphysical. 

    # NoOII, CN=3: OII=?, Z in [0.6, 1.6] and secure redshift
    ibool = (Z>0.6) & (Z<1.6) & (OII_ERR<=0) & (ZQUALITY>=3) & (BRIcut==1)    
    class_col[ibool] = 3

    # LowZ, CN=4: OII=NA, Z outside [0.6, 1.6]
    ibool = np.logical_or((np.logical_or((Z>1.6), (Z<0.6)) & (ZQUALITY>=3)),(ZQUALITY==-1))  & (OII_ERR<=0) & (BRIcut==1)
    class_col[ibool] = 4

    # NoZ, CN=5: OII=NA, Z undetermined.
    ibool = np.logical_or.reduce(((ZQUALITY==-2) , (ZQUALITY==0) , (ZQUALITY==1) ,(ZQUALITY==2)))& (BRIcut==1)  & (OII_ERR<=0)
    class_col[ibool] = 5
    
    # D2reject, CN=6
    ibool = BRIcut!=1
    class_col[ibool] = 6
    
    # D2unobserved, CN=8
    ibool = (BRIcut==1) & (ZQUALITY<-10)   # Objects that were not assigned color-selection flag are classifed as DEEP2 color rejected objects.
    class_col[ibool] = 8  
    
    return class_col

def count_nn(arr):
    """
    Count the number of non-negative elements.
    """   
    return arr[arr>-1].size




def class_breakdown(fn, cn, weight, area, rwd="D"):
    """
    Given a list of class fields and corresponding weights and areas, return the breakdown of object 
    for each class. fn gives the field number. 
    """
    
    # Place holder for tallying
    counts = np.zeros(8)
    
    # Generate counts
    for i in range(len(fn)):
        # Computing counts
        if rwd == "R":
            tmp = generate_raw_breakdown(cn[i])
        elif rwd == "W":
            tmp = generate_weighted_breakdown(cn[i], weight[i])
        else:
            tmp = generate_density_breakdown(cn[i], weight[i], area[i])
        
        # Tallying counts
        if rwd in ["R", "W"]:
            counts += tmp
        else:
            counts += tmp/len(fn)
        
        # Printing counts
        print(str_counts(fn[i], rwd, tmp))            

    # Total or average counts
    if rwd in ["R", "W"]:
        print(str_counts("Total", rwd, counts))
    else:
        print(str_counts("Avg.", rwd, counts))


    
def str_counts(fn, rwd_str, counts):
    """
    Given the counts of various class of objects return a formated string.
    """
    if type(fn)==str:
        return_str = "%s & %s " % (fn,rwd_str)
    else:
        return_str = "%d & %s " % (fn,rwd_str)
        
    for i in range(counts.size):
        return_str += "& %d " % counts[i]
    
    return_str += "& %d " % np.sum(counts)
    
    return_str += latex_eol()
    
    return return_str
    
    
def generate_raw_breakdown(cn):
    return np.delete(np.bincount(cn.astype(int)),7)

def generate_weighted_breakdown(cn, weight):
    counts = np.zeros(8,dtype=int)
    for i,e in enumerate(np.delete(np.arange(0,9,1, dtype=int),7)):
        if (e!=6) and (e!=8):
            counts[i] = np.sum(weight[cn==e])
        else:
            counts[i] = np.sum(cn==e)
    return counts

def generate_density_breakdown(cn, weight,area):
    counts = np.zeros(8,dtype=int)
    for i,e in enumerate(np.delete(np.arange(0,9,1, dtype=int),7)):
        if (e!=6) and (e!=8):
            counts[i] = np.sum(weight[cn==e])/np.float(area)
        else:
            counts[i] = np.sum(cn==e)/np.float(area)
    return counts
    

def generate_table_header():
    return "Field & R/W/D & "+" & ".join(class_names()) + " & Total" + latex_eol() + latex_hline()

def latex_eol():
    return "\\\\ \\hline"

def latex_hline():
    return "\\hline"
    
def class_names():
    """
    Provide a list of class names.
    """
    return ["Gold", "Silver", "LowOII","NoOII", "LowZ", "NoZ", "D2reject", "D2unobserved"]



def return_bricknames(ra, dec, br_name, ra_range, dec_range,tol):
    ibool = (ra>(ra_range[0]-tol)) & (ra<(ra_range[1]+tol)) & (dec>(dec_range[0]-tol)) & (dec<(dec_range[1]+tol))
    return  br_name[ibool]






def combine_tractor(fits_directory):
    """
    Given the file directory, find all Tractor fits files combine them and return as a rec-array.
    """
    onlyfiles = [f for f in listdir(fits_directory) if isfile(join(fits_directory, f))]
    print("Number of files in %s %d" % (fits_directory, len(onlyfiles)-1))
    
    
    DR3 = None
    for i,e in enumerate(onlyfiles,start=1):
        # If the file ends with "fits"
        if e[-4:] == "fits":
            print("Combining file %d. %s" % (i,e))
            # If DR3 has been set with something.
            tmp_table = apply_mask(fits.open(fits_directory+e)[1].data)
            if DR3 is not None:
                DR3 = np.hstack((DR3, tmp_table))
            else:
                DR3 = tmp_table
                
    return DR3

def combine_tractor_nocut(fits_directory):
    """
    Given the file directory, find all Tractor fits files combine them and return as a rec-array.
    """
    onlyfiles = [f for f in listdir(fits_directory) if isfile(join(fits_directory, f))]
    print("Number of files in %s %d" % (fits_directory, len(onlyfiles)-1))
    
    
    DR3 = None
    for i,e in enumerate(onlyfiles,start=1):
        # If the file ends with "fits"
        if e[-4:] == "fits":
            print("Combining file %d. %s" % (i,e))
            # If DR3 has been set with something.
            tmp_table = fits.open(fits_directory+e)[1].data
            if DR3 is not None:
                DR3 = np.hstack((DR3, tmp_table))
            else:
                DR3 = tmp_table
                
    return DR3    

def apply_mask(table):
    """
    Given a tractor catalog table, apply the standard mask. brick_primary and flux inverse variance. 
    """
    brick_primary = load_brick_primary(table)
    givar, rivar, zivar = load_grz_invar(table)
    ibool = (brick_primary==True) & (givar>0) & (rivar>0) &(zivar>0) 
    table_trimmed = np.copy(table[ibool])

    return table_trimmed
    
def load_grz_anymask(fits):
    g_anymask = fits['DECAM_ANYMASK'][:][:,1]
    r_anymask = fits['DECAM_ANYMASK'][:][:,2]
    z_anymask = fits['DECAM_ANYMASK'][:][:,4]
    
    return g_anymask, r_anymask, z_anymask

def load_grz_allmask(fits):
    g_allmask = fits['DECAM_ALLMASK'][:][:,1]
    r_allmask = fits['DECAM_ALLMASK'][:][:,2]
    z_allmask = fits['DECAM_ALLMASK'][:][:,4]
    
    return g_allmask, r_allmask, z_allmask


def load_radec(fits):
    ra = fits["ra"][:]
    dec= fits["dec"][:]
    return ra, dec

def load_radec_ext(pcat):
    ra = pcat["RA_DEEP"]
    dec = pcat["DEC_DEEP"]    
    return ra, dec

def cross_match_catalogs(pcat, pcat_ref, tol=0.5):
    """
    Match pcat catalog to pcat_ref via ra and dec.
    Incorporate astrometric correction if any.
    """
    # Load radec
    ra, dec = load_radec_ext(pcat)
    ra_ref, dec_ref = load_radec_ext(pcat_ref)
    
    # Create spherematch objects
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)  
    c_ref = SkyCoord(ra=ra_ref*u.degree, dec=dec_ref*u.degree)  
    idx, idx_ref, d2d, d3d = c_ref.search_around_sky(c, 1*u.arcsec)
    
    # Find the median difference
    ra_med_diff = np.median(ra_ref[idx_ref]-ra[idx])
    dec_med_diff = np.median(dec_ref[idx_ref]-dec[idx])
    
    print("ra,dec discrepancy: %.3f, %.3f"%(ra_med_diff*3600, dec_med_diff*3600))
    
    # Finding matches again taking into account astrometric differnce.
    c = SkyCoord(ra=(ra+ra_med_diff)*u.degree, dec=(dec+dec_med_diff)*u.degree)  
    c_ref = SkyCoord(ra=ra_ref*u.degree, dec=dec_ref*u.degree)  
    idx, idx_ref, d2d, d3d = c_ref.search_around_sky(c, 1*u.arcsec)    
    
    return idx, idx_ref    


def load_brick_primary(fits):
    return fits['brick_primary'][:]


def load_shape(fits):
    r_dev = fits['SHAPEDEV_R'][:]
    r_exp = fits['SHAPEEXP_R'][:]
    return r_dev, r_exp


def load_star_mask(table):
    return table["TYCHOVETO"][:].astype(int).astype(bool)

def load_oii(fits):
    return fits["OII_3727"][:]

def new_oii_lim(N_new, N_old=2400):
    """
    Return the new OII low threshold given the updated fiber number in units of
    1e-17 ergs/A/cm^2/s
    """
    return 8*np.sqrt(N_new/N_old)

def frac_above_new_oii(oii, weight, new_oii_lim):
    """
    Given the oii and weights of the objects of interest and the new OII limit, return
    the proportion of objects that meet the new criterion.
    """
    ibool = oii>new_oii_lim
    return weight[ibool].sum()/weight.sum()    


def load_fits_table(fname):
    """Given the file name, load  the first extension table."""
    return fits.open(fname)[1].data

def save_fits(data, fname):
    """
    Given a rec array and a file name (with "fits" filename), save it.
    """
    cols = fits.ColDefs(np.copy(data)) # This is somehow necessary.
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fname, clobber=True)
    
    return 

def save_fits_join(data1,data2, fname):
    """
    Given a rec array and a file name (with "fits" filename), save it.
    """
    
    data = rec.merge_arrays((data1,data2), flatten=True, usemask=False,asrecarray=True)
    cols = fits.ColDefs(data) 
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fname, clobber=True)
    
    return 

def load_weight(fits):
    return fits["TARG_WEIGHT"]    

def fits_append(table, new_col, col_name, idx1, idx2, dtype="default", dtype_user=None):
    """
    Given fits table and field column/name pair,
    append the new field to the table using the idx1 and idx2 that correspond to 
    fits table and new column indices.

    If dtype =="default", then the default of float variable type is used.
    If dtype =="user", then user provided data type is used.
    """
    global large_random_constant
    new_col_sorted = np.ones(table.shape[0])*large_random_constant
    new_col_sorted[idx1] = new_col[idx2]
    
    if dtype=="default":
        new_table = rec.append_fields(table, col_name, new_col_sorted, dtypes=new_col_sorted.dtype, usemask=False, asrecarray=True)
    else:
        new_table = rec.append_fields(table, col_name, new_col_sorted, dtypes=dtype_user, usemask=False, asrecarray=True)


    return new_table

def load_fits_table(fname):
    """Given the file name, load  the first extension table."""
    return fits.open(fname)[1].data
    

def apply_star_mask(fits):
    ibool = ~load_star_mask(fits) 
    
    return fits[ibool]

def load_grz_flux(fits):
    """
    Return raw (un-dereddened) g,r,z flux values.
    """
    g = fits['decam_flux'][:][:,1]
    r = fits['decam_flux'][:][:,2]
    z = fits['decam_flux'][:][:,4]
    
    return g,r,z

def load_grz_flux_dereddened(fits):
    # Colors: DECam model flux in ugrizY
    # mag = 22.5-2.5log10(f)
    g = fits['decam_flux'][:][:,1]/fits['decam_mw_transmission'][:][:,1]
    r = fits['decam_flux'][:][:,2]/fits['decam_mw_transmission'][:][:,2]
    z = fits['decam_flux'][:][:,4]/fits['decam_mw_transmission'][:][:,4]
    return g, r, z    

def load_grz_invar(fits):
    givar = fits['DECAM_FLUX_IVAR'][:][:,1]
    rivar = fits['DECAM_FLUX_IVAR'][:][:,2]
    zivar = fits['DECAM_FLUX_IVAR'][:][:,4]
    return givar, rivar, zivar

def load_grz(fits):
    # Colors: DECam model flux in ugrizY
    # mag = 22.5-2.5log10(f)
    g = (22.5 - 2.5*np.log10(fits['decam_flux'][:][:,1]/fits['decam_mw_transmission'][:][:,1]))
    r = (22.5 - 2.5*np.log10(fits['decam_flux'][:][:,2]/fits['decam_mw_transmission'][:][:,2]))
    z = (22.5 - 2.5*np.log10(fits['decam_flux'][:][:,4]/fits['decam_mw_transmission'][:][:,4]))
    return g, r, z


def load_W1W2_flux(fits):
    """
    Return raw (un-dereddened) w1, w2 flux values.
    """
    w1flux = fits["WISE_FLUX"][:][:,1]
    w2flux = fits["WISE_FLUX"][:][:,2]
    return w1flux, w2flux

def load_W1W2_fluxinvar(fits):
    w1_ivar = fits["WISE_FLUX_IVAR"][:][:,1]
    w2_ivar = fits["WISE_FLUX_IVAR"][:][:,2]
    return w1_ivar, w2_ivar

def load_W1W2(fits):
    # Colors: DECam model flux in ugrizY
    # mag = 22.5-2.5log10(f)
    w1 = (22.5 - 2.5*np.log10(fits['WISE_FLUX'][:][:,1]/fits['WISE_MW_TRANSMISSION'][:][:,0]))
    w2 = (22.5 - 2.5*np.log10(fits['WISE_FLUX'][:][:,2]/fits['WISE_MW_TRANSMISSION'][:][:,1]))
    return w1, w2




def load_redz(fits):
    """
    Return redshift
    """
    return fits["RED_Z"]

 

def reasonable_mask(table, decam_mask = "all", SN = True):
    """
    Given DECaLS table, return a boolean index array that indicates whether an object passed flux positivity, reasonable color range, and allmask conditions
    Impose SN>2 and decam_mask cut, only if it's requested.
    """
    grzflux = load_grz_flux(table)
    ibool = is_grzflux_pos(grzflux)
    
    grz = load_grz(table)
    ibool &= is_reasonable_color(grz) 
    
    if decam_mask == "all":
        grz_allmask = load_grz_allmask(table)
        ibool &= pass_grz_decammask(grz_allmask)
    elif decam_mask == "any":
        grz_anymask = load_grz_anymask(table)
        ibool &= pass_grz_decammask(grz_anymask)        
    
    if SN:
        grzivar = load_grz_invar(table)
        ibool &= pass_grz_SN(grzflux, grzivar, thres=2)

    return ibool

def pass_grz_SN(grzflux, grzivar, thres=2):
    gf, rf, zf = grzflux
    gi, ri, zi = grzivar
    
    return ((gf*np.sqrt(gi))>thres)&((rf*np.sqrt(ri))>thres)&((zf*np.sqrt(zi))>thres)

def grz_S2N(grzflux, grzinvar):
    g,r,z = grzflux
    gi,ri,zi = grzinvar
    return g*np.sqrt(gi),r*np.sqrt(ri),z*np.sqrt(zi)

def grz_flux_error(grzinvar):
    """
    Given the inverse variance return flux error.
    """
    gi,ri,zi = grzinvar
    return np.sqrt(1/gi),np.sqrt(1/ri),np.sqrt(1/zi)

def mag_depth_Xsigma(f_err, sigma=5):
    """
    Given flux error, return five sigma depth
    """
    return flux2mag(f_err*sigma)

def flux2mag(flux):
    return 22.5-2.5*np.log10(flux)    
    
def pass_grz_decammask(grz_decammask):
    gm, rm, zm = grz_decammask
    return (gm==0) & (rm==0) & (zm==0)

def is_reasonable_color(grz):
    """
    Given grz mag list, check whether colors lie within a reasonable range.
    """
    g,r,z = grz
    gr = g-r
    rz = r-z
    
    return (gr>-0.75) & (gr<2.5) & (rz>-0.5) &(rz<2.7)


def is_grzflux_pos(grzflux):
    """
    Given a list [gflux, rflux, zflux], return a boolean array that tells whether each object has all good fluxes or not.
    """
    ibool = (grzflux[0]>0) & (grzflux[1]>0) & (grzflux[2]>0)
    return ibool



def check_astrometry(ra1,dec1,ra2,dec2,pt_size=0.3):
    """
    Given two sets of ra/dec's return median difference in degrees.
    """
    ra_diff = ra2-ra1
    dec_diff = dec2-dec1
    ra_med_diff = np.median(ra_diff)
    dec_med_diff = np.median(dec_diff)
    return ra_med_diff, dec_med_diff



def crossmatch_cat1_to_cat2(ra1, dec1, ra2, dec2, tol=1./(deg2arcsec+1e-12)):
    """
    Return indices of cat1 (e.g., DR3) and cat2 (e.g., DEE2) cross matched to tolerance. 

    Note: Function used to cross-match DEEP2 and DR3 catalogs in each field 
    and test for any astrometric discrepancies. That is, for every object in 
    DR3, find the nearest object in DEEP2. For each DEEP2 object matched, 
    pick DR3 object that is the closest. The surviving objects after these 
    matching process are the cross-matched set.
    """
    
    # Match cat1 to cat2 using astropy functions.
    idx_cat1_to_cat2, d2d = match_cat1_to_cat2(ra1, dec1, ra2, dec2)
    
    # Indicies of unique cat2 objects that were matched.
    cat2matched = np.unique(idx_cat1_to_cat2)
    
    # For each cat2 object matched, pick cat1 object that is the closest. 
    # Skip if the closest objects more than tol distance away.
    idx1 = [] # Place holder for indices
    idx2 = []
    tag = np.arange(ra1.size,dtype=int)
    for e in cat2matched:
        ibool = (idx_cat1_to_cat2==e)
        candidates = tag[ibool]
        dist2candidates = d2d[ibool]
        # Index of the minimum distance cat1 object
        if dist2candidates.min()<tol:
            idx1.append(candidates[np.argmin(dist2candidates)])
            idx2.append(e)
    
    # Turning list of indices into numpy arrays.
    idx1 = np.asarray(idx1)
    idx2 = np.asarray(idx2)
    
    # Return the indices of cat1 and cat2 of cross-matched objects.
    return idx1, idx2



def match_cat1_to_cat2(ra1, dec1, ra2, dec2):
    """
    "c = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)  
    catalog = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)  
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)  

    idx are indices into catalog that are the closest objects to each of the coordinates in c, d2d are the on-sky distances between them, and d3d are the 3-dimensional distances." -- astropy documentation.  

    Fore more information: http://docs.astropy.org/en/stable/coordinates/matchsep.html#astropy-coordinates-matching 
    """    
    cat1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)  
    cat2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)  
    idx, d2d, d3d = cat1.match_to_catalog_sky(cat2)
    
    return idx, d2d.degree

def closest_idx(arr, val):
    return np.argmin(np.abs(arr-val))   




##############################################################################
# The following is adpated from the URL indicated below.
# """
#     ImagingLSS
#     https://github.com/desihub/imaginglss/blob/master/imaginglss/analysis/tycho_veto.py

#     veto objects based on a star catalogue.
#     The tycho vetos are based on the email discussion at:
#     Date: June 18, 2015 at 3:44:09 PM PDT
#     To: decam-data@desi.lbl.gov
#     Subject: decam-data Digest, Vol 12, Issue 29
#     These objects takes a decals object and calculates the
#     center and rejection radius for the catalogue in degrees.
#     Note : The convention for veto flags is True for 'reject',
#     False for 'preserve'.

#     apply_tycho takes the galaxy catalog and appends a Tychoveto column
#     the code works fine for ELG and LRGs. For other galaxy type, you need to adjust it!
# """

# Import modules
import sys

def BOSS_DR9(tycho):
    bmag = tycho['BMAG']
    # BOSS DR9-11
    b = bmag.clip(6, 11.5)
    R = (0.0802 * b ** 2 - 1.86 * b + 11.625) / 60. #
    return R

def DECAM_LRG(tycho):
    vtmag = tycho['VTMAG']
    R = 10 ** (3.5 - 0.15 * vtmag) / 3600.
    return R

DECAM_ELG = DECAM_LRG

def DECAM_QSO(tycho):
    vtmag = tycho['VTMAG']
    # David Schlegel recommends not applying a bright star mask
    return vtmag - vtmag

def DECAM_BGS(tycho):
    vtmag = tycho['VTMAG']
    R = 10 ** (2.2 - 0.15 * vtmag) / 3600.
    return R

def radec2pos(ra, dec):
    """ converting ra dec to position on a unit sphere.
        ra, dec are in degrees.
    """
    pos = np.empty(len(ra), dtype=('f8', 3))
    ra = ra * (np.pi / 180)
    dec = dec * (np.pi / 180)
    pos[:, 2] = np.sin(dec)
    pos[:, 0] = np.cos(dec) * np.sin(ra)
    pos[:, 1] = np.cos(dec) * np.cos(ra)
    return pos

def tycho(filename):
    """
    read the Tycho-2 catalog and prepare it for the mag-radius relation
    """
    dataf = fits.open(filename)
    data = dataf[1].data
    tycho = np.empty(len(data),
        dtype=[
            ('RA', 'f8'),
            ('DEC', 'f8'),
            ('VTMAG', 'f8'),
            ('VMAG', 'f8'),
            ('BMAG', 'f8'),
            ('BTMAG', 'f8'),
            ('VARFLAG', 'i8'),
            ])
    tycho['RA'] = data['RA']
    tycho['DEC'] = data['DEC']
    tycho['VTMAG'] = data['MAG_VT']
    tycho['BTMAG'] = data['MAG_BT']
    vt = tycho['VTMAG']
    bt = tycho['BTMAG']
    b = vt - 0.09 * (bt - vt)
    v = b - 0.85 * (bt - vt)
    tycho['VMAG']=v
    tycho['BMAG']=b
    dataf.close()
    return tycho


def txts_read(filename):
    obj = np.loadtxt(filename)
    typeobj = np.dtype([
              ('RA','f4'), ('DEC','f4'), ('COMPETENESS','f4'),
              ('rflux','f4'), ('rnoise','f4'), ('gflux','f4'), ('gnoise','f4'),
              ('zflux','f4'), ('znoise','f4'), ('W1flux','f4'), ('W1noise','f4'),
              ('W2flux','f4'), ('W2noise','f4')
              ])
    nobj = obj[:,0].size
    data = np.zeros(nobj, dtype=typeobj)
    data['RA'][:] = obj[:,0]
    data['DEC'][:] = obj[:,1]
    data['COMPETENESS'][:] = obj[:,2]
    data['rflux'][:] = obj[:,3]
    data['rnoise'][:] = obj[:,4]
    data['gflux'][:] = obj[:,5]
    data['gnoise'][:] = obj[:,6]
    data['zflux'][:] = obj[:,7]
    data['znoise'][:] = obj[:,8]
    data['W1flux'][:] = obj[:,9]
    data['W1noise'][:] = obj[:,10]
    data['W2flux'][:] = obj[:,11]
    data['W2noise'][:] = obj[:,12]
    #datas = np.sort(data, order=['RA'])
    return data

def veto(coord, center, R):
    """
        Returns a veto mask for coord. any coordinate within R of center
        is vet.
        Parameters
        ----------
        coord : (RA, DEC)
        center : (RA, DEC)
        R     : degrees
        Returns
        -------
        Vetomask : True for veto, False for keep.
    """
    from sklearn.neighbors import KDTree
    pos_stars = radec2pos(center[0], center[1])
    R = 2 * np.sin(np.radians(R) * 0.5)
    pos_obj = radec2pos(coord[0], coord[1])
    tree = KDTree(pos_obj)
    vetoflag = ~np.zeros(len(pos_obj), dtype='?')
    arg = tree.query_radius(pos_stars, r=R)
    arg = np.concatenate(arg)
    vetoflag[arg] = False
    return vetoflag



def apply_tycho(objgal, tychofn,galtype='LRG'):
    # reading tycho star catalogs
    tychostar = tycho(tychofn)
    #
    # mag-radius relation
    #
    if galtype == 'LRG' or galtype == 'ELG':    # so far the mag-radius relation is the same for LRG and ELG
        radii = DECAM_LRG(tychostar)
    else:
        sys.exit("Check the apply_tycho function for your galaxy type")
    #
    #
    # coordinates of Tycho-2 stars
    center = (tychostar['RA'], tychostar['DEC'])
    #
    #
    # coordinates of objects (galaxies)
    coord = (objgal['ra'], objgal['dec'])
    #
    #
    # a 0.0 / 1.0 array (1.0: means the object is contaminated by a Tycho-2 star, so 0.0s are good)
    tychomask = (~veto(coord, center, radii)).astype('f4')
    objgal = rec.append_fields(objgal, ['TYCHOVETO'], data=[tychomask], dtypes=tychomask.dtype, usemask=False)
    return objgal


