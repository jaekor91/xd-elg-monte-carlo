from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

import numpy as np
from astropy.io import ascii, fits
from astropy.wcs import WCS
from xd_elg_utils import *
from model_class import *
import sys
import matplotlib.pyplot as plt
import time

category = ["NonELG", "NoZ", "ELG"]

# 0: Full F34 data
# 1: F3 data only
# 2: F4 data only
# 3-7: CV1-CV5: Sub-sample F34 into five-fold CV sets.
# 8-10: Magnitude changes. For power law use full data. Not used. 
# g in [22.5, 23.5], [22.75, 23.75], [23, 24]. 

sub_sample_name = ["Full", "F3", "F4", "CV1", "CV2", "CV3", "CV4", "CV5", "Mag1", "Mag2", "Mag3"] # No need to touch this
NK_list = [1]#, 3, 4, 5, 6, 7]
Niter = 1

j = 0 
instance_model = model3(j)        

# fg, fr, fz = instance_model.var_x, instance_model.var_y, instance_model.gmag
# X = np.array([fg, fr, fz]).T # X inputs
fNoZ = 0.25

x, y, z = instance_model.var_x, instance_model.var_y, instance_model.gmag
X = np.array([x, y, z]).T # X inputs

iELG_DESI = instance_model.iELG & (instance_model.oii > 8) & (instance_model.red_z>0.6) & (instance_model.red_z<1.6)
Y = np.logical_or(instance_model.iNoZ, iELG_DESI) # Label
ws = instance_model.w # sample weights
ws[instance_model.iNoZ] *= fNoZ 
field = instance_model.field


#----- Number of cv
num_cv = 10

#----- Creating test vs. training sets; train and test the performance
area_total = 2.12
area_test = 2.12 * 1/float(num_cv)
print "CV: P_last; Precision"
precision_list = []
kf = KFold(n_splits=num_cv)
cv_counter = 0
for train, test in kf.split(X):
    cv_counter +=1    
    X_train, Y_train, w_train = X[train], Y[train], ws[train]
    X_test, Y_test, w_test = X[test], Y[test], ws[test]
    
    # Construct and train a new classifier
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=50, n_jobs=-1)
    rnd_clf.fit(X_train, Y_train, sample_weight=w_train)
    
    # Apply the classifer
    y_pred_rf = rnd_clf.predict_proba(X_test)[:,1]

    # Find the probability threshold where desired number density of 2,100 is reached
    num_desired = 2400
    prob_threshold = 1.
    dp = 0.001

    num_counter = 0
    while num_counter <= num_desired:
        num_counter = np.sum(w_test[y_pred_rf > prob_threshold])/(area_test)
        prob_threshold -= dp

    cfm = confusion_matrix(Y_test, (y_pred_rf > prob_threshold), sample_weight=w_test)
    TP, FP = cfm[1, 1], cfm[0, 1]
    precision = (TP/float(TP+FP))
    print "%d: %.3f, %.3f" % (cv_counter, prob_threshold, precision)
    precision_list.append(precision)
    
print "Precision mean/std: %.3f/%.3f" % (np.mean(precision_list), np.std(precision_list))





#----- Creating test vs. training set using Field splits.
# Field split 
if2 = (field==2)
if34 = ~if2

# Divide training vs. test according to fields
X_test, Y_test, w_test = X[if2], Y[if2], ws[if2]
X34, Y34, w34 = X[if34], Y[if34], ws[if34]
area_total = 1.42 # Field 3 and 4 
area_test = 0.7
print "CV: P_last; Precision"
precision_list = []
kf = KFold(n_splits=num_cv)
cv_counter = 0
for train, test in kf.split(X34):
    cv_counter +=1    
    X_train, Y_train, w_train = X34[train], Y34[train], w34[train]
#     X_test, Y_test, w_test = X[test], Y[test], ws[test] # We will actually use only 
    
    # Construct and train a new classifier
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=50, n_jobs=-1)
    rnd_clf.fit(X_train, Y_train, sample_weight=w_train)
    
    # Apply the classifer
    y_pred_rf = rnd_clf.predict_proba(X_test)[:,1]

    # Find the probability threshold where desired number density of 2,100 is reached
    num_desired = 2400
    prob_threshold = 1.
    dp = 0.001

    num_counter = 0
    while num_counter <= num_desired:
        num_counter = np.sum(w_test[y_pred_rf > prob_threshold])/(area_test)
        prob_threshold -= dp

    cfm = confusion_matrix(Y_test, (y_pred_rf > prob_threshold), sample_weight=w_test)
    TP, FP = cfm[1, 1], cfm[0, 1]
    precision = (TP/float(TP+FP))
    print "%d: %.3f, %.3f" % (cv_counter, prob_threshold, precision)
    precision_list.append(precision)
    
print "Precision mean/std: %.3f/%.3f" % (np.mean(precision_list), np.std(precision_list))







#----- Using only Field 2 data
# Field split 
if2 = (field==2)

# Divide training vs. test according to fields
X2, Y2, w2 = X[if2], Y[if2], ws[if2]
area_total = 0.7 # Field 2
area_test = area_total / float(num_cv)
print "CV: P_last; Precision"
precision_list = []
kf = KFold(n_splits=num_cv)
cv_counter = 0
for train, test in kf.split(X2):
    cv_counter +=1    
    X_train, Y_train, w_train = X2[train], Y2[train], w2[train]
    X_test, Y_test, w_test = X2[test], Y2[test], w2[test] # We will actually use only 
    
    # Construct and train a new classifier
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=50, n_jobs=-1)
    rnd_clf.fit(X_train, Y_train, sample_weight=w_train)
    
    # Apply the classifer
    y_pred_rf = rnd_clf.predict_proba(X_test)[:,1]

    # Find the probability threshold where desired number density of 2,100 is reached
    num_desired = 2400
    prob_threshold = 1.
    dp = 0.001

    num_counter = 0
    while num_counter <= num_desired:
        num_counter = np.sum(w_test[y_pred_rf > prob_threshold])/(area_test)
        prob_threshold -= dp

    cfm = confusion_matrix(Y_test, (y_pred_rf > prob_threshold), sample_weight=w_test)
    TP, FP = cfm[1, 1], cfm[0, 1]
    precision = (TP/float(TP+FP))
    print "%d: %.3f, %.3f" % (cv_counter, prob_threshold, precision)
    precision_list.append(precision)
    
print "Precision mean/std: %.3f/%.3f" % (np.mean(precision_list), np.std(precision_list))