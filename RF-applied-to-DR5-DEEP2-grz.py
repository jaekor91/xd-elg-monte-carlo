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

print "Importing DEEP2-DR5 F234 dataset"
model = model3(0)

# X values
areas = np.load("spec-area-DR5-matched.npy")
print "area", areas

fNoZ = 0.25 # How much weight we are willing to give.
x1, x2, x3, x4, x5 = model.gflux, model.rflux, model.zflux, model.w1_flux, model.w2_flux
X0 = np.array([x1, x2, x3]).T # X inputs

# Target
iELG_DESI = model.iELG & (model.oii > 8) & (model.red_z>0.6) & (model.red_z<1.6)
Y0 = np.logical_or(model.iNoZ, iELG_DESI) # Label
ws0 = model.w # sample weights
ws0[model.iNoZ] *= fNoZ 

# Get the field information as well.
field0 = model.field

print "Total Ndensity in each field and total (g<24.25)"
print "Weigthed f2/f3/f4: %.1f, %.1f, %.1f" % (np.sum(ws0[field0==2])/areas[0], np.sum(ws0[field0==3])/areas[1], np.sum(ws0[field0==4])/areas[2])
print "Raw f2/f3/f4: %.1f, %.1f, %.1f" % (np.sum(field0==2)/areas[0], np.sum(field0==3)/areas[1], np.sum(field0==4)/areas[2])
print "\n"

# Importing DEEP2-DR5 F234 dataset
# Fraction of unmatched objects with g [17.0, 24.2]: 6.97 percent
# We multiply the correction to the weights before training.
# area [ 0.70405384  0.8495537   0.57206704]
# Total Ndensity in each field and total (g<24.25)
# Weigthed f2/f3/f4: 28096.8, 32132.8, 38365.5
# Raw f2/f3/f4: 27372.9, 31677.8, 37247.4


print "Random grid-search for optimal hyper-parameters"
# Random search
np.random.seed(25)
# Hyper-parameters to try
total_try  = 50
max_depths = np.random.uniform(low=7, high=15, size=total_try).astype(int)
max_leaves = np.random.uniform(100, 1000, size=total_try).astype(int)

fig = plt.figure()
plt.scatter(max_depths, max_leaves, s =50,  edgecolors="black")
# plt.show()
plt.close()


print "Best setting seems to be 600/10."

print "Load the calibration dataset"
def load_DR5_calibration():
    """
    Note g-magnitude cut
    """
    A = np.sum(np.load("DR5-calibration-sweeps-areas.npy"))
    data = np.load("DR5-calibration-sweeps.npy")
    g = data["g"]
    data = data[g > mag2flux(24.)]
    g = data["g"]    
    r = data["r"]
    z = data["z"]
    w1 = data["w1"]
    w2  = data["w2"]
    return g, r, z, w1, w2, A
g, r, z, w1, w2, A = load_DR5_calibration()
X_cal = np.array([g, r, z]).T
print "Ndensity (Area): %d (%.2f)" % ((g.size)/A, A)

# Load the calibration dataset
# Ndensity (Area): 31272 (161.25)


print "Train on F234 and test on CV-fold; F2, F3, adn F4 seperately; and F234 combined."

# General strategy: 
# - For each set of hyper-parameters, perform 10 fold cross-validation.
# - Pick the best set of hyper-parameters
# - Train the model on the full training data set.
# - Get the variable imporatance
# - Plot the region in grz space...

# Cross-validation setting 
num_cv = 10
kf = KFold(n_splits=num_cv)

# Area corresponding to the whole data set
area_total = np.sum(areas)
area_test = area_total * 1/float(num_cv)

# Randomly shuffling data
shuffle = np.random.permutation(Y0.size)
X = X0[shuffle]
Y = Y0[shuffle]
ws = ws0[shuffle]
field = field0[shuffle]

# Recording test result. (Ndensity, Precision) x num_cv
# cv: 0
# F2: 3
# F3: 6
# F4: 9
# F234: 12
results_arr = np.zeros((total_try, 15, num_cv), dtype=np.float)


# Performing random search 
N_estimators = 50
for i, max_depth in enumerate(max_depths):
    cv_counter = 0
    max_leave = max_leaves[i]
    print "Max_depth, max_leave: %d, %d" % (max_depth, max_leave)
    print "CV: Last prob, precision, Ndensity"
    for train, test in kf.split(X):
        # Test/train split for CV
        X_train, Y_train, w_train = X[train], Y[train], ws[train]
        X_test, Y_test, w_test = X[test], Y[test], ws[test]

        # Construct and train a new classifier
        rnd_clf = RandomForestClassifier(n_estimators=N_estimators, max_leaf_nodes=max_leave, max_depth=max_depth, n_jobs=-1)
        rnd_clf.fit(X_train, Y_train, sample_weight=w_train)

        # Find the probability threshold where desired number density of 2400 is reached based on the calibration dataset.
        num_desired = 2400
        prob_threshold = 1.
        dp = 1e-3
        y_cal_rf = rnd_clf.predict_proba(X_cal)[:,1] # Predict on the calibration data set
        num_counter = 0
        while num_counter <= num_desired:
            num_counter = np.sum(y_cal_rf > prob_threshold)/A
            prob_threshold -= dp

        # Evaluating the selection on the various test sets.
        # cv: 0
        # F2: 3
        # F3: 6
        # F4: 9
        # F234: 12
        
        # cv
        y_pred_rf = rnd_clf.predict_proba(X_test)[:,1]
        cfm = confusion_matrix(Y_test, (y_pred_rf > prob_threshold), sample_weight=w_test)
        TP, FP = cfm[1, 1], cfm[0, 1]
        precision = (TP/float(TP+FP))
        Ndensity = np.sum(w_test[y_pred_rf>prob_threshold])/area_test
        Ndensity_raw = np.sum(y_pred_rf>prob_threshold)/area_test
        results_arr[i, 0, cv_counter] = Ndensity
        results_arr[i, 1, cv_counter] = Ndensity_raw        
        results_arr[i, 2, cv_counter] = precision
        print "%d: %.3f, %.3f, %.2f" % (cv_counter, prob_threshold, precision, Ndensity)
        
        # F2-F4:
        for f in range(2, 5):
            X_test = X[field==f]
            Y_test = Y[field==f]
            w_test = ws[field==f]
            y_pred_rf = rnd_clf.predict_proba(X_test)[:,1]
            cfm = confusion_matrix(Y_test, (y_pred_rf > prob_threshold), sample_weight=w_test)
            TP, FP = cfm[1, 1], cfm[0, 1]
            precision = (TP/float(TP+FP))
            Ndensity = np.sum(w_test[y_pred_rf>prob_threshold])/areas[f-2]
            Ndensity_raw = np.sum(y_pred_rf>prob_threshold)/areas[f-2]
            results_arr[i, 3*(f-1)+0, cv_counter] = Ndensity
            results_arr[i, 3*(f-1)+1, cv_counter] = Ndensity_raw        
            results_arr[i, 3*(f-1)+2, cv_counter] = precision

        # F234:
        X_test = X
        Y_test = Y
        w_test = ws
        y_pred_rf = rnd_clf.predict_proba(X_test)[:,1]
        cfm = confusion_matrix(Y_test, (y_pred_rf > prob_threshold), sample_weight=w_test)
        TP, FP = cfm[1, 1], cfm[0, 1]
        precision = (TP/float(TP+FP))
        Ndensity = np.sum(w_test[y_pred_rf>prob_threshold])/area_total
        Ndensity_raw = np.sum(y_pred_rf>prob_threshold)/area_total
        results_arr[i, 12+0, cv_counter] = Ndensity
        results_arr[i, 12+1, cv_counter] = Ndensity_raw        
        results_arr[i, 12+2, cv_counter] = precision
        
        # Update the counter
        cv_counter +=1    
    
    Ndensity_mean = np.mean(results_arr[i, 0, :])
    Ndensity_std = np.std(results_arr[i, 0, :])
    Precision_mean = np.mean(results_arr[i, 1, :])
    Precision_std = np.std(results_arr[i, 1, :])
    print "CV: Precision mean/std: %.3f/%.3f" % (Precision_mean, Precision_std)
    print "CV: Ndensity mean/std: %.2f/%.2f" % (Ndensity_mean, Ndensity_std)
    print "\n"
    
# np.save("RF-applied-DEEP2-F234-grz.npy", results_arr)    

results_arr = np.load("RF-applied-DEEP2-F234-grz.npy")


# cv: 0
# F2: 3
# F3: 6
# F4: 9
# F234: 12
precision_list = []
Ndensity_list = []
test_sets = ["CV", "F2", "F3", "F4", "F234"]
for i in range(total_try):
    print "Max depths/leaves: %d/%d" % (max_depths[i], max_leaves[i])
    for j in range(5):
        Ndensity_mean = np.mean(results_arr[i, 3*j, :])
        Ndensity_std = np.std(results_arr[i, 3*j, :])
        Ndensity_raw_mean = np.mean(results_arr[i, 3*j+1, :])
        Ndensity_raw_std = np.std(results_arr[i, 3*j+1, :])    
        Precision_mean = np.mean(results_arr[i, 3*j+2, :])
        Precision_std = np.std(results_arr[i, 3*j+2, :])
        print "%s: Ndensity mean/std: %.1f/%.1f" % (test_sets[j], Ndensity_mean, Ndensity_std)
        print "%s: Ndensity Raw mean/std: %.1f/%.1f" % (test_sets[j], Ndensity_raw_mean, Ndensity_raw_std)        
        print "%s: Precision mean/std: %.3f/%.3f" % (test_sets[j], Precision_mean, Precision_std)
        print "\n"
        
        if j == 0:
            precision_list.append(Precision_mean)
            Ndensity_list.append(Ndensity_mean)

idx_best = np.argmax(precision_list)
print "Best depth/max leaves: %d, %d" % (max_depths[idx_best], max_leaves[idx_best])
print precision_list[idx_best], Ndensity_list[idx_best]
print "\n"

i = idx_best
print "Max depths/leaves: %d/%d" % (max_depths[i], max_leaves[i])
for j in range(5):
    Ndensity_mean = np.mean(results_arr[i, 3*j, :])
    Ndensity_std = np.std(results_arr[i, 3*j, :])
    Ndensity_raw_mean = np.mean(results_arr[i, 3*j+1, :])
    Ndensity_raw_std = np.std(results_arr[i, 3*j+1, :])    
    Precision_mean = np.mean(results_arr[i, 3*j+2, :])
    Precision_std = np.std(results_arr[i, 3*j+2, :])
    print "%s: Ndensity mean/std: %.1f/%.1f" % (test_sets[j], Ndensity_mean, Ndensity_std)
    print "%s: Ndensity Raw mean/std: %.1f/%.1f" % (test_sets[j], Ndensity_raw_mean, Ndensity_raw_std)        
    print "%s: Precision mean/std: %.3f/%.3f" % (test_sets[j], Precision_mean, Precision_std)
    print "\n"

# Best depth/max leaves: 11, 694
# 0.769085426905 1860.15059707


# Max depths/leaves: 11/694
# CV: Ndensity mean/std: 1860.2/113.3
# CV: Ndensity Raw mean/std: 1833.3/96.8
# CV: Precision mean/std: 0.769/0.016


# F2: Ndensity mean/std: 1540.1/15.3
# F2: Ndensity Raw mean/std: 1499.3/11.4
# F2: Precision mean/std: 0.895/0.007


# F3: Ndensity mean/std: 1980.5/24.3
# F3: Ndensity Raw mean/std: 1946.4/13.2
# F3: Precision mean/std: 0.920/0.004


# F4: Ndensity mean/std: 2849.2/34.9
# F4: Ndensity Raw mean/std: 2620.5/18.1
# F4: Precision mean/std: 0.904/0.004


# F234: Ndensity mean/std: 2068.4/19.6
# F234: Ndensity Raw mean/std: 1979.7/9.8
# F234: Precision mean/std: 0.908/0.003

fig = plt.figure()

plt.scatter(max_depths, max_leaves, s = 200,  edgecolors="black", c=np.asarray(precision_list), cmap="seismic")
plt.colorbar()


# plt.show()
plt.close()                



print "With the best hyper-parameter setting re-train on the full dataset"

# Area corresponding to the whole data set
area_total = np.sum(areas)
area_test = area_total * 1/float(num_cv)

# Randomly shuffling data
shuffle = np.random.permutation(Y0.size)
X = X0[shuffle]
Y = Y0[shuffle]
ws = ws0[shuffle]
field = field0[shuffle]

N_estimators = 500
# Construct and train a new classifier
rnd_clf = RandomForestClassifier(n_estimators=N_estimators, max_leaf_nodes=max_leaves[idx_best], max_depth=max_depths[idx_best], n_jobs=-1)
rnd_clf.fit(X, Y, sample_weight=ws)

# Find the probability threshold where desired number density of 2400 is reached based on the calibration dataset.
num_desired = 2400
prob_threshold = 1.
dp = 1e-3
y_cal_rf = rnd_clf.predict_proba(X_cal)[:,1] # Predict on the calibration data set
num_counter = 0
while num_counter <= num_desired:
    num_counter = np.sum(y_cal_rf > prob_threshold)/A
    prob_threshold -= dp

print "Last probability threshold: %.3f" % prob_threshold

var_names = ["g", "r", "z", "w1", "w2"]
print "Variable importance"
for vn, score in zip(var_names, rnd_clf.feature_importances_):
    print "%s: %.3f" % (vn, score)

# With the best hyper-parameter setting re-train on the full dataset
# Last probability threshold: 0.632
# Variable importance
# g: 0.306
# r: 0.427
# z: 0.267


print "Plotting g-r, r-z, g selection region."
# Comping up with g-r, r-z, g grid
gmag = np.linspace(22, 24, 41, endpoint=True)
# print gmag
gr = np.linspace(-3, 3, 601, endpoint=True)
# print gr
rz = np.linspace(-3, 3, 601, endpoint=True)
# print rz

grv, rzv, gmagv = np.meshgrid(gr, rz, gmag)

# Turn them into fluxes.
rmagv = gmagv - grv
rflux = mag2flux(rmagv)
gflux = mag2flux(gmagv)
zflux = mag2flux(rmagv-rzv)

samples = np.array([gflux.flatten(), rflux.flatten(), zflux.flatten()]).T
y_samples_rf = rnd_clf.predict_proba(samples)[:, 1]
iselected = (y_samples_rf > prob_threshold)
samples_selected = samples[iselected]
gflux, rflux, zflux = samples_selected[:, 0], samples_selected[:, 1], samples_selected[:, 2]

gmag = flux2mag(gflux)
rmag = flux2mag(rflux)
zmag = flux2mag(zflux)

gr = gmag - rmag
rz = rmag - zmag




for g_tmp in np.unique(gmag):
    ibool = (gmag == g_tmp)
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(rz[ibool], gr[ibool], edgecolors="none", c="green", alpha=0.5, s=5)
    plt.title(r"$g = %.2f$ " % g_tmp, fontsize=25)
    plt.xlabel(r"$r-z$", fontsize=25)
    plt.ylabel(r"$g-r$", fontsize=25)
    plt.axis("equal")
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.savefig("RF-DEEP2-grz-only-boundary-g%d.png" % (g_tmp * 100), dpi = 250, bbox_inches = "tight")
#     plt.show()
    plt.close()







print "With the best hyper-parameter setting train on F34 and apply to Field 2 dataset."

# Dividing into F34 and F2 data
iF2 = field0 == 2
X_F2 = X0[iF2]
Y_F2 = Y0[iF2]
ws_F2 = ws0[iF2]

# F34
X_F34 = X0[~iF2]
Y_F34 = Y0[~iF2]
ws_F34 = ws0[~iF2]


N_estimators = 500
# Construct and train a new classifier
rnd_clf = RandomForestClassifier(n_estimators=N_estimators, max_leaf_nodes=max_leaves[idx_best], max_depth=max_depths[idx_best], n_jobs=-1)
rnd_clf.fit(X_F34, Y_F34, sample_weight=ws_F34)


# Find the probability threshold where desired number density of 2400 is reached based on the calibration dataset.
num_desired = 2400
prob_threshold = 1.
dp = 1e-3
y_cal_rf = rnd_clf.predict_proba(X_cal)[:,1] # Predict on the calibration data set
num_counter = 0
while num_counter <= num_desired:
    num_counter = np.sum(y_cal_rf > prob_threshold)/A
    prob_threshold -= dp

print "Last probability threshold: %.3f" % prob_threshold


print "Performance on DEEP2 F2"
y_pred_rf = rnd_clf.predict_proba(X_F2)[:, 1]
Y_test = Y_F2
w_test = ws_F2

cfm = confusion_matrix(Y_test, (y_pred_rf > prob_threshold), sample_weight=w_test)
TP, FP = cfm[1, 1], cfm[0, 1]
precision = (TP/float(TP+FP))
Ndensity = np.sum(w_test[y_pred_rf>prob_threshold])/areas[0]
Ndensity_raw = np.sum(y_pred_rf>prob_threshold)/areas[0]
print "%.3f, %.1f, %.1f" % (precision, Ndensity, Ndensity_raw)


var_names = ["g", "r", "z", "w1", "w2"]
print "Variable importance"
for vn, score in zip(var_names, rnd_clf.feature_importances_):
    print "%s: %.3f" % (vn, score)


# With the best hyper-parameter setting train on F34 and apply to Field 2 dataset.
# Last probability threshold: 0.643
# Performance on DEEP2 F2
# 0.728, 1373.8, 1391.9
# Variable importance
# g: 0.313
# r: 0.426
# z: 0.262





print "With the best hyper-parameter setting train on F2 and apply to Field 2 dataset."
# Here cross validation performance is necessary.
# Cross-validation setting 
num_cv = 10
kf = KFold(n_splits=num_cv)


# Dividing into F34 and F2 data
iF2 = field0 == 2
X_F2 = X0[iF2]
Y_F2 = Y0[iF2]
ws_F2 = ws0[iF2]

# Performing random search 
N_estimators = 50
cv_counter = 0
max_depth, max_leave = max_depths[idx_best], max_leaves[idx_best]
print "Max_depth, max_leave: %d, %d" % (max_depth, max_leave)
print "CV: Last prob, precision, Ndensity"

Ndensity_list = []
Ndensity_w_list = []
precision_list = []
print "Performance on DEEP2 F2"
for train, test in kf.split(X_F2):
    # Test/train split for CV
    X_train, Y_train, w_train = X_F2[train], Y_F2[train], ws_F2[train]
    X_test, Y_test, w_test = X_F2[test], Y_F2[test], ws_F2[test]

    # Construct and train a new classifier
    rnd_clf = RandomForestClassifier(n_estimators=N_estimators, max_leaf_nodes=max_leave, max_depth=max_depth, n_jobs=-1)
    rnd_clf.fit(X_train, Y_train, sample_weight=w_train)

    # Find the probability threshold where desired number density of 2400 is reached based on the calibration dataset.
    num_desired = 2400
    prob_threshold = 1.
    dp = 1e-3
    y_cal_rf = rnd_clf.predict_proba(X_cal)[:,1] # Predict on the calibration data set
    num_counter = 0
    while num_counter <= num_desired:
        num_counter = np.sum(y_cal_rf > prob_threshold)/A
        prob_threshold -= dp
    
    y_pred_rf = rnd_clf.predict_proba(X_test)[:, 1]
    w_test = w_test

    cfm = confusion_matrix(Y_test, (y_pred_rf > prob_threshold), sample_weight=w_test)
    TP, FP = cfm[1, 1], cfm[0, 1]
    precision = (TP/float(TP+FP))
    Ndensity = np.sum(w_test[y_pred_rf>prob_threshold])/areas[0] * num_cv
    Ndensity_raw = np.sum(y_pred_rf>prob_threshold)/areas[0] * num_cv
    print "%.3f, %.1f, %.1f" % (precision, Ndensity, Ndensity_raw)
    
    Ndensity_list.append(Ndensity_raw)
    Ndensity_w_list.append(Ndensity)
    precision_list.append(precision)

print "Last probability threshold: %.3f" % prob_threshold

var_names = ["g", "r", "z", "w1", "w2"]
print "Variable importance"
for vn, score in zip(var_names, rnd_clf.feature_importances_):
    print "%s: %.3f" % (vn, score)




# With the best hyper-parameter setting train on F2 and apply to Field 2 dataset.
# Max_depth, max_leave: 11, 694
# CV: Last prob, precision, Ndensity
# Performance on DEEP2 F2
# 0.618, 1312.8, 1448.8
# 0.647, 1364.4, 1619.2
# 0.677, 1434.5, 1718.6
# 0.697, 1576.7, 1377.7
# 0.759, 1176.5, 1349.3
# 0.593, 1549.0, 1505.6
# 0.716, 1674.7, 1548.2
# 0.663, 1613.0, 1519.8
# 0.814, 1060.3, 1306.7
# 0.753, 1366.7, 1463.0
# Last probability threshold: 0.596
# Variable importance
# g: 0.305
# r: 0.385
# z: 0.310



# Number density 1485.68183567 119.038208034
# Weighted number density 1412.85898273 186.846445199
# Precision 0.693720428182 0.0647815037129

