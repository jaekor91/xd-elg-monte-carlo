import numpy as np
from scipy.stats import multivariate_normal

def summed_gm(pos, mus_fit, covar_fit, amps_fit):
    """
    Return a value evaluated with Gaussian mixture given means, covariances and relative amplitudes (that sum up to one).
    """
    func_val = 0.
    for i in range(amps_fit.size):
        func_val += amps_fit[i]*multivariate_normal.pdf(pos,mean=mus_fit[i], cov=covar_fit[i])    
    return func_val



import numpy as np
from scipy.stats import multivariate_normal

def summed_gm(pos, mus_fit, covar_fit, amps_fit):
    """
    Return a value evaluated with Gaussian mixture given means, covariances and relative amplitudes (that sum up to one).
    """
    func_val = 0.
    for i in range(amps_fit.size):
        func_val += amps_fit[i]*multivariate_normal.pdf(pos,mean=mus_fit[i], cov=covar_fit[i])    
    return func_val

def inverse_cdf_gm(cvs, Xrange, Yrange, Amps, Covs, Means, xspacing=1e-3, yspacing=1e-3, gridnumber = 1e2):
    """Given parameters for a gaussian mixture and a confidence interval volume, 
    returns the corresponding probability density level.
    
    Requires:
    - Amps, Means, Covs: Parametesr for a gaussian mixture
    - cvs: Confidence volume(s) in a list format.
    - grid-spacing: Grid-spacing to try
    - Xrange, Yrange: Ranges in which to compute the guassians.
    
    Return:
    - probability density lvel
    
    Example:
    --------
    """
    x = np.arange(Xrange[0], Xrange[1], xspacing)
    y = np.arange(Yrange[0], Yrange[1], yspacing)
    
    X,Y = np.meshgrid(x,y)
    Z = summed_gm(np.transpose(np.array([Y,X])), Means, Covs, Amps) # evaluation of the function on the grid
    pdf_sorted = np.sort(Z.flatten())

    # Prob grid 
    prob_grid = np.arange(0, pdf_sorted[-1],pdf_sorted[-1]/1e3)
    
    # Calculating cumulative function (that is probability volume within the boundary)
    cdf = np.zeros(prob_grid.size)
    for i in range(cdf.size):
        cdf[i] = np.sum(pdf_sorted[pdf_sorted>prob_grid[i]])* xspacing * yspacing
    
    pReturn = []
    for i in range(len(cvs)):
        cv = cvs[i]
        pReturn.append(prob_grid[find_nearest_idx(cdf,cv)])

    return pReturn
        
def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

        
    
    
