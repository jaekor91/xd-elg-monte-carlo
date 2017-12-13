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
model.fmax_MC = mag2flux(16.)
model.fcut = mag2flux(24.5) # After noise addition, we make a cut at 24.5.

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


# Area
MC_AREA = 100 # In sq. deg.
print "\n"


print "Generating intrinsic sample proportional to simulation area of %d" % MC_AREA
model.set_area_MC(MC_AREA)
start = time.time()
model.gen_sample_intrinsic_mag()
print "Time for generating sample: %.2f seconds" % (time.time() - start)
print "\n"

tag = "NDM-obiwan-samples"


# grz-fluxes and redz and OII and weights
i = 2
gflux = model.gflux0[i]
rflux = model.rflux0[i]
zflux = model.zflux0[i]
redz = model.redz0[i]
OII = model.oii0[i]
weights = model.iw0[i]



# Save the samples
np.savez("NDM-obiwan-sample.npz", gflux = gflux, rflux = rflux, zflux = zflux, redz = redz, OII = OII, weights = weights)
# To load use: data = np.load("NDM-obiwan-sample.npz")

# Parameterization
mu_g = flux2asinh_mag(gflux, band="g")
mu_r = flux2asinh_mag(rflux, band="r")
mu_z = flux2asinh_mag(zflux, band="z")

mu_gz = mu_g - mu_z
mu_gr = mu_g - mu_r

ibool = (gflux>0) & (rflux >0) & (zflux>0)
gmag = flux2mag(gflux[ibool])
rmag = flux2mag(rflux[ibool])
zmag = flux2mag(zflux[ibool])

gr = gmag-rmag
rz = rmag-zmag



# Asinh color scatter
plt.scatter(mu_gz, mu_gr, s=1, edgecolor="none", alpha=1, c="black")
plt.xlabel("asinh g-z")
plt.ylabel("asinh g-r")
plt.axis("equal")
plt.xlim([-2, 5])
plt.savefig(tag+"-asinh-mag-scatter.png", dpi=200, bbox_inches="tight")
plt.close()



# mag color scatter
plt.scatter(rz, gr, s=1, edgecolor="none", alpha=1., c="black")
plt.xlabel("r-z")
plt.ylabel("g-r")
plt.axis("equal")
plt.xlim([-4, 4])
plt.savefig(tag+"-mag-scatter.png", dpi=200, bbox_inches="tight")
plt.close()



# grz histogram
bins = np.arange(15, 27, 0.1)
plt.hist(gmag, bins=bins, histtype="step", color="green", lw=2, label="gmag")
plt.hist(rmag, bins=bins, histtype="step", color="red", lw=2, label="rmag")
plt.hist(zmag, bins=bins, histtype="step", color="purple", lw=2, label="zmag")
plt.legend(loc="upper left", fontsize=15)
plt.savefig(tag+"-mag-hist.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()

plt.hist(gmag, bins=bins, histtype="step", color="green", lw=2, label="gmag", weights=weights[ibool])
plt.hist(rmag, bins=bins, histtype="step", color="red", lw=2, label="rmag", weights=weights[ibool])
plt.hist(zmag, bins=bins, histtype="step", color="purple", lw=2, label="zmag", weights=weights[ibool])
plt.legend(loc="upper left", fontsize=15)
plt.savefig(tag+"-mag-hist-weighted.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()


# Redshift 
bins = np.arange(0, 2, 0.01)
plt.hist(redz, bins=bins, histtype="step", color="black", lw=2, label="redz")
plt.legend(loc="upper left", fontsize=15)
plt.savefig(tag+"-redz.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()

plt.hist(redz, bins=bins, histtype="step", color="black", lw=2, label="redz", weights=weights)
plt.legend(loc="upper left", fontsize=15)
plt.savefig(tag+"-redz-weighted.png", dpi=200, bbox_inches="tight")
# plt.show()
plt.close()



# OII histogram
bins = np.arange(-10, 100, 1)
plt.hist(OII, bins=bins, histtype="step", color="black", lw=2, label="OII")
plt.xlabel("Flux in 1e-17")
plt.legend(loc="upper right", fontsize=15)
plt.savefig(tag+"-OII.png", dpi=200, bbox_inches="tight")
plt.close()


plt.hist(OII, bins=bins, histtype="step", color="black", lw=2, label="OII", weights=weights)
plt.xlabel("Flux in 1e-17")
plt.legend(loc="upper right", fontsize=15)
plt.savefig(tag+"-OII-weighted.png", dpi=200, bbox_inches="tight")
plt.close()