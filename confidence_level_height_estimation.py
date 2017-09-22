import numpy as np
from scipy.stats import multivariate_normal

def summed_gm(pts, mus, covar, amps):
    """
    pts: [Nsample, ND]

    Return PDF fo GMM specified by the parameters. 
    """
    func_val = np.zeros(pts.shape[0])
    for i in range(amps_fit.size):
        func_val += amps_fit[i]*multivariate_normal.pdf(pts, mean=mus[i], cov=covar[i])    
    return func_val

def inverse_cdf_2D(cvs, X, Y, PDF):
    """
    Given grid pts X, Y and PDF, return the PDF values that correspond to probability mass values cvs.
    cv of 0.98 means there are 98% probability mass within the contour.
    """
    
    pdf_sorted = np.sort(PDF.flatten())

    # Prob grid 
    prob_grid = np.arange(0, pdf_sorted[-1], pdf_sorted[-1]/float(1e3))
    
    # Calculating cumulative function (that is probability volume within the boundary)
    cdf = np.zeros(prob_grid.size)
    for i in range(cdf.size):
        cdf[i] = np.sum(pdf_sorted[pdf_sorted>prob_grid[i]]) # * xspacing * yspacing
    
    pReturn = []
    for i in range(len(cvs)):
        cv = cvs[i]
        pReturn.append(prob_grid[find_nearest_idx(cdf,cv)])

    return pReturn


        
def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

        
    
    
