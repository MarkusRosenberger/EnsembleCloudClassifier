#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mrosenberger
"""

import os
import numpy as np
import pandas as pd
import csv
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics as skm

import Analysis as ana
import Modelfunctions as mfunc

ens_dir = '<path_to_directory>'
model_name = '<name-of_model_directory'
n_classes = 30
p_thresh = 0.5

# load observations and images
with open('GroundTruth_test.csv', mode ='r') as file:
    gt_test = np.array([line for line in csv.reader(file)]).astype(int)

with h5py.File('img_test.nc', mode='r') as f:
    img_test = f['Full_period'][:]/255

# load model ensemble
model_vec = [tf.keras.models.load_model(os.path.join(ens_dir, m, model_name), custom_objects={'Brier_Score':mfunc.Brier_Score()})
             for m in np.sort(os.listdir(ens_dir))]

n_members = len(model_vec)

'''
Member based (MB) predictions, i.e. scores for each member separately
'''

predictions_MB = np.array([model(img_test) for model in model_vec]) # probabilities member-wise
predictions_MB_binary = np.where(predictions_MB >= p_thresh, 1, 0) # convert to binary format (one-hot encoded)

MLCM_MB = np.array([ana.MLCM(gt_test, p, thresh = p_thresh, return_plot = False, types = None, from_logits = False) for p in predictions_MB])

MCC_MB = np.zeros((n_members, n_classes))
prec_MB = np.zeros((n_members, n_classes))
rec_MB = np.zeros((n_members, n_classes))

for p in range(n_members):
    output_dict_MB = [ana.calculate_measures(MLCM_MB[p], index = i, measures = ['MCC', 'Precision', 'Recall'], do_bootstrap=False) for i in range(n_classes)]
    MCC_MB[p] = np.array([o['MCC'] for o in output_dict_MB])
    prec_MB[p] = np.array([o['Precision'] for o in output_dict_MB])
    rec_MB[p] = np.array([o['Recall'] for o in output_dict_MB])


# round average member-based MLCM because:
#   floats don't make much sense in a confidence matrix
#   bootstrapping algorithm works only with integer input
MLCM_MBmean_rounded = np.round(np.mean(MLCM_MB, axis = 0))

'''
Average based (AB) predictions, i.e. score calculated from 
mean of predicted probabilities of ensemble members
'''

predictions_AB = np.mean(predictions_MB, axis = 0) # calculate mean of probabilities
predictions_AB_binary = np.where(np.mean(predictions_MB, axis = 0) >= p_thresh, 1, 0) # convert to binary format (one-hot encoded)

# calcualte mean and statistics
MLCM_AB_mean = ana.MLCM(gt_test, predictions_AB, thresh = p_thresh, return_plot = False, types = None, from_logits = False)
    
output_dict_AB = [ana.calculate_measures(MLCM_AB_mean, index = i, measures = ['MCC', 'Precision', 'Recall'], do_bootstrap=False) for i in range(n_classes)] # ['Accuracy', 'F1', 'MCC', 'Precision', 'Recall']
MCC_AB_mean = np.array([o['MCC'] for o in output_dict_AB])
prec_AB_mean = np.array([o['Precision'] for o in output_dict_AB])
rec_AB_mean = np.array([o['Recall'] for o in output_dict_AB])

'''
AB macro-average, no weights
'''
Prec_AB_macro = np.nanmean(prec_AB_mean)
Rec_AB_macro = np.nanmean(rec_AB_mean)
MCC_AB_macro = np.nanmean(MCC_AB_mean)

'''
Bootstrapping of MB predictions
'''

output_dict_MB = [ana.calculate_measures(MLCM_MBmean_rounded, index = i, measures = ['MCC', 'Precision', 'Recall'], do_bootstrap=True) for i in range(n_classes)]

MCC_MB_bt = np.array([o['bt_MCC'] for o in output_dict_MB])
rec_MB_bt = np.array([o['bt_Recall'] for o in output_dict_MB])
prec_MB_bt = np.array([o['bt_Precision'] for o in output_dict_MB])

'''
Majority Voting (MV) of binary MB predictions
to get single forecast vector from ensemble runs
at least 2 members have to predict the same class
'''

predictions_MB_binary_combined = np.sum(predictions_MB_binary, axis = 0)
predictions_MV = []

for p in predictions_MB_binary_combined:
    # split in height levels
    p_l = p[:10]
    p_m = p[10:20]
    p_h = p[20:30]

    # at least 2 members have to predict the same class
    class_l = np.where(p_l == np.maximum(np.max(p_l), 2))[0]    
    class_m = np.where(p_m == np.maximum(np.max(p_m), 2))[0]  
    class_h = np.where(p_h == np.maximum(np.max(p_h), 2))[0]  

    predictions_temp = np.zeros(n_classes)
    
    predictions_temp[class_l] += 1
    predictions_temp[class_m+10] += 1
    predictions_temp[class_h+20] += 1

    predictions_MV.append(predictions_temp)
    
predictions_MV = np.array(predictions_MV)

# calculate measures
MLCM_MV = ana.MLCM(gt_test, predictions_MV, thresh = p_thresh, return_plot = False, types = None, from_logits = False)

output_dict_MV = [ana.calculate_measures(MLCM_MV, index = i, measures = ['MCC', 'Precision', 'Recall'], do_bootstrap=False) for i in range(n_classes)]
MCC_MV = np.array([o['MCC'] for o in output_dict_MV])
prec_MV = np.array([o['Precision'] for o in output_dict_MV])
rec_MV = np.array([o['Recall'] for o in output_dict_MV])

'''
AB micro-average
'''

# calculate True Positives (TP), False Negatives (FN), False Positives (FP), True Negatives (TN)
# separately for each class and sum them 

TP_vec = []
FN_vec = []
FP_vec = []
TN_vec = []

for index in range(n_classes):

    TP_vec.append(MLCM_AB_mean[index, index])
    FN_vec.append(np.sum(MLCM_AB_mean[index])-TP_vec[-1])
    FP_vec.append(np.sum(MLCM_AB_mean[:,index])-TP_vec[-1])
    
    main_diagonal = np.eye(len(MLCM_AB_mean))*MLCM_AB_mean
    TN_vec.append(np.sum(main_diagonal)-TP_vec[-1])

TP = np.sum(TP_vec)
FN = np.sum(FN_vec)
FP = np.sum(FP_vec)
TN = np.sum(TN_vec)

# calculate measures
Prec_AB_micro = ( TP / (TP + FP) )
Rec_AB_micro = ( TP / (TP + FN) )

num = TP*TN - FP*FN
denom = np.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))

MCC_AB_micro = ( num/denom )

'''
MB micro-average
'''

Prec_MB_micro = []
Rec_MB_micro = []
MCC_MB_micro = []

# same as above but for each member separately
for MLCM in MLCM_MB:
    TP_vec = []
    FN_vec = []
    FP_vec = []
    TN_vec = []
    
    for index in range(n_classes):
    
        TP_vec.append(MLCM[index, index])
        FN_vec.append(np.sum(MLCM[index])-TP_vec[-1])
        FP_vec.append(np.sum(MLCM[:,index])-TP_vec[-1])
        
        main_diagonal = np.eye(len(MLCM))*MLCM
        TN_vec.append(np.sum(main_diagonal)-TP_vec[-1])
    
    TP = np.sum(TP_vec)
    FN = np.sum(FN_vec)
    FP = np.sum(FP_vec)
    TN = np.sum(TN_vec)
    
    Prec_MB_micro.append( ( TP / (TP + FP) ))
    Rec_MB_micro.append( ( TP / (TP + FN) ))
    
    num = TP*TN - FP*FN
    denom = np.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))
    
    MCC_MB_micro.append( ( num/denom ))

'''
MV micro-average
'''

# again same as above
TP_vec = []
FN_vec = []
FP_vec = []
TN_vec = []

for index in range(n_classes):

    TP_vec.append(MLCM_MV[index, index])
    FN_vec.append(np.sum(MLCM_MV[index])-TP_vec[-1])
    FP_vec.append(np.sum(MLCM_MV[:,index])-TP_vec[-1])
    
    main_diagonal = np.eye(len(MLCM_MV))*MLCM_MV
    TN_vec.append(np.sum(main_diagonal)-TP_vec[-1])

TP = np.sum(TP_vec)
FN = np.sum(FN_vec)
FP = np.sum(FP_vec)
TN = np.sum(TN_vec)

Prec_MB_majority_micro = ( TP / (TP + FP) )
Rec_MB_majority_micro = ( TP / (TP + FN) )

num = TP*TN - FP*FN
denom = np.sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN))

MCC_MB_majority_micro = ( num/denom )

'''
Subset Accuracy calculated with binary predictions
'''

subset_acc_AB = skm.accuracy_score(gt_test, predictions_AB_binary)
subset_acc_MV = skm.accuracy_score(gt_test, predictions_MV)

subset_acc_MB = []
for preds in predictions_MB_binary:
    subset_acc_MB.append(skm.accuracy_score(gt_test, preds))

'''
Hamming Loss, again from binary predictions
'''

Hamming_AB = skm.hamming_loss(gt_test, predictions_AB_binary)
Hamming_MV = skm.hamming_loss(gt_test, predictions_MV)

Hamming_MB = []
for preds in predictions_MB_binary:
    Hamming_MB.append(skm.hamming_loss(gt_test, preds))
    

'''
Plot of MB statistics and bootstrap results as boxplots
weigthed macro-average added on the right edge
'''

xax = range(n_classes)
fs = 12
labels = ['Precision', 'Recall', 'MCC']
colors = ['tab:red', 'tab:purple', 'tab:green']
items = ['a)', 'b)', 'c)']

fig, axs = plt.subplots(3,1, figsize = (9,10), sharex = False,
                        gridspec_kw = dict(hspace=0.33)
                        )

# MB scores of model & bootstrap
dist_vec = [prec_MB, rec_MB, MCC_MB]
dist_vec_bt = [prec_MB_bt.T, rec_MB_bt.T, MCC_MB_bt.T]

# Heydarian et al.(2022) suggest to use row-wise sums of MLCM as weights
MLCM_wgts_MB = np.sum(MLCM_MBmean_rounded, axis = 1)[:n_classes]
    
# alpha_indices geben stark augmentierte Klassen an
# residual_indices die anderen
# Wird benötigt, um die beiden im Plot voneinander unterscheiden zu können
# To discriminate between highly augmented and less augmented classes
alpha_indices = np.array([ 3,  4, 11, 12, 14, 15, 18, 19, 23, 24, 25, 26, 27, 28, 29]) # highly augmented
residual_indices = np.array([0, 1, 2, 5, 6, 7, 8, 9, 10, 13, 16, 17, 20, 21, 22]) # less augmentation

for index, (ax, m) in enumerate(zip(axs, dist_vec)):
  
    # check for NaNs in distribution and mask them
    mask = ~np.isnan(m) 
    filtered = [d[ma] for (d, ma) in zip(m.T, mask.T)]
 
    # Plotting of model and bootstrap values
    ax.boxplot([filtered[a] for a in alpha_indices], positions = alpha_indices, whis = (0,100), patch_artist = True, boxprops = dict(facecolor = colors[index], alpha = .5), medianprops = dict(color = 'k') ) 
    ax.boxplot([filtered[r] for r in residual_indices], positions = residual_indices, whis = (0,100), patch_artist = True, boxprops = dict(facecolor = colors[index], alpha = 1), medianprops = dict(color = 'k'))
    ax.boxplot(dist_vec_bt[index], positions = xax, whis = (0,100))
        
    # calculate weighted average of min, mean, and max statistics over all classes and plot them
    m_metrics = [np.nanmin(m, axis = 0), np.nanmean(m, axis = 0), np.nanmax(m, axis = 0)]
    weighted_avg = [np.ma.average(np.ma.array(m_metrics[i], mask=np.isnan(m_metrics[i])), weights=MLCM_wgts_MB) for i in range(3)]
    
    ax.scatter(n_classes + 1, weighted_avg[1], c = 'k', marker = '_', clip_on = False)
    ax.fill_between([n_classes + .6, n_classes + 1.4], weighted_avg[0], weighted_avg[2], fc = colors[index], alpha = .3, clip_on = False)
    
    ax.axvline(9.5, ls = '--', lw = .3, c = 'k')
    ax.axvline(19.5, ls = '--', lw = .3, c = 'k')
    ax.set_xticks(ticks = range(n_classes), labels = list(range(10))*3, fontsize = fs)
    ax.set_yticks(ticks = [.0, .2, .4, .6, .8, 1.], labels = [.0, .2, .4, .6, .8, 1.], fontsize = fs)
    
    if index == 2:
        ax.set_xlabel('Cloud class', fontsize = fs)
    
    ax.set_ylabel(labels[index], fontsize = fs)
    ax.set_ylim(0,1)
    ax.set_xlim(-.5, n_classes + 2)
    ax.tick_params('y', left = True, right = True, labelleft = True, labelright = True)

    ax.text(x = .175, y = 1.07, s = r'C$_L$', ha = 'center', va = 'center', fontsize = fs, transform=ax.transAxes)
    ax.text(x = .50, y = 1.07, s = r'C$_M$', ha = 'center', va = 'center', fontsize = fs, transform=ax.transAxes)
    ax.text(x = .825, y = 1.07, s = r'C$_H$', ha = 'center', va = 'center', fontsize = fs, transform=ax.transAxes)
    ax.text(x = -.09, y = 1.13, s = items[index], ha = 'center', va = 'center', fontsize = fs+1, fontweight = 'bold', transform=ax.transAxes)

# save
plt.savefig('MB_Metrics_boxplot.png', bbox_inches = 'tight', dpi = 500)
plt.close()

'''
Reliability, Resolution, Uncertainty, Brier Skill Score
AB and MB, micro-average and macro-average
'''

n_bins = 10

'''
AB
'''
REL_AB = []
RES_AB = []
UNC_AB = []

for i in range(n_classes):
    out = ana.ReliabilityDiagram(predictions_AB[:,i], gt_test[:,i], n_bins = n_bins, return_plot = False) 
    REL_AB.append(out['Reliability'])
    RES_AB.append(out['Resolution'])
    UNC_AB.append(out['Uncertainty'])

'''
MB
'''
REL_MB = []
RES_MB = []
UNC_MB = []
BSS_MB = []

# iterate over classes ...
for i in range(n_classes):
    y_true = gt_test[:,i]
    
    REL_temp = []
    RES_temp = []
    UNC_temp = []
    BSS_temp = []
    
    # ... and ensemble members
    for predictions in predictions_MB:
        y_pred = predictions[:,i]
        
        out = ana.ReliabilityDiagram(y_pred, y_true, n_bins = n_bins, return_plot = False) 
        REL = out['Reliability']
        RES = out['Resolution']
        UNC = out['Uncertainty']

        REL_temp.append(REL)
        RES_temp.append(RES)
        UNC_temp.append(UNC)
        BSS_temp.append((RES-REL)/UNC)


    REL_MB.append(REL_temp)
    RES_MB.append(RES_temp)
    UNC_MB.append(UNC_temp)
    BSS_MB.append(BSS_temp)


REL_MB = np.array(REL_MB)
RES_MB = np.array(RES_MB)
UNC_MB = np.array(UNC_MB)
BSS_MB = np.array(BSS_MB)

'''
BSS MB micro-average
'''

BSS_MB_micro = []

UNC_m = np.sum(UNC_MB[:,0]) # independent of ensemble member

# iterate over members
for m in range(n_members):
    REL_m = np.sum(REL_MB[:,m])
    RES_m = np.sum(RES_MB[:,m])

    BSS_MB_micro.append((RES_m - REL_m)/UNC_m)   

BSS_MB_micro_mean = np.mean(BSS_MB_micro)
BSS_MB_mean = np.mean(BSS_MB, axis = 1)

'''
BSS AB micro-average
'''
BSS_AB_micro = (np.sum(RES_AB) - np.sum(REL_AB))/np.sum(UNC_AB)

'''
BSS MB macro-average, with and without weights
'''
BSS_MB_mean_macro_noweight = np.ma.average(np.ma.array(BSS_MB_mean, mask=np.isnan(BSS_MB_mean)), weights=None)
BSS_MB_mean_macro_weights = np.ma.average(np.ma.array(BSS_MB_mean, mask=np.isnan(BSS_MB_mean)), weights=MLCM_wgts_MB)

score_table = np.zeros(shape = (4,9))

# AB micro:
score_table[0,0] = Prec_AB_micro
score_table[1,0] = Rec_AB_micro
score_table[2,0] = MCC_AB_micro
score_table[3,0] = BSS_AB_micro


# AB macro no weights
score_table[0,1] = Prec_AB_macro # Prec_AB_macro
score_table[1,1] = Rec_AB_macro # Rec_AB_macro
score_table[2,1] = MCC_AB_macro # MCC_AB_macro
BSS_AB_macro = (np.array(RES_AB) - np.array(REL_AB))/np.array(UNC_AB)
score_table[3,1] = np.ma.average(np.ma.array(BSS_AB_macro, mask=np.isnan(BSS_AB_macro)), weights=None) # BSS_AB_macro

# AB macro weighted
MLCM_wgts_AB = np.sum(MLCM_AB_mean, axis = 1)[:n_classes]
score_table[0,2] = np.ma.average(np.ma.array(prec_AB_mean, mask=np.isnan(prec_AB_mean)), weights=MLCM_wgts_AB) # Prec_AB_macro
score_table[1,2] = np.ma.average(np.ma.array(rec_AB_mean, mask=np.isnan(rec_AB_mean)), weights=MLCM_wgts_AB) # Rec_AB_macro
score_table[2,2] = np.ma.average(np.ma.array(MCC_AB_mean, mask=np.isnan(MCC_AB_mean)), weights=MLCM_wgts_AB) # MCC_AB_macro
BSS_AB_macro = (np.array(RES_AB) - np.array(REL_AB))/np.array(UNC_AB)
score_table[3,2] = np.ma.average(np.ma.array(BSS_AB_macro, mask=np.isnan(BSS_AB_macro)), weights=MLCM_wgts_AB) # BSS_AB_macro

# MB micro
score_table[0,3] = np.nanmean(Prec_MB_micro) # Prec_MB_micro
score_table[1,3] = np.nanmean(Rec_MB_micro) # Rec_MB_micro
score_table[2,3] = np.nanmean(MCC_MB_micro) # MCC_MB_micro
score_table[3,3] = BSS_MB_micro_mean # BSS_MB_micro

# MB macro no weights
score_table[0,4] = np.ma.average(np.ma.array(np.nanmean(prec_MB, axis = 0), mask=np.isnan(np.nanmean(prec_MB, axis = 0))), weights=None) # Prec_MB_macro
score_table[1,4] = np.ma.average(np.ma.array(np.nanmean(rec_MB, axis = 0), mask=np.isnan(np.nanmean(rec_MB, axis = 0))), weights=None) # Rec_MB_macro
score_table[2,4] = np.ma.average(np.ma.array(np.nanmean(MCC_MB, axis = 0), mask=np.isnan(np.nanmean(MCC_MB, axis = 0))), weights=None) # MCC_MB_macro
score_table[3,4] = BSS_MB_mean_macro_noweight # BSS_MB_macro

# MB macro weighted
score_table[0,5] = np.ma.average(np.ma.array(np.nanmean(prec_MB, axis = 0), mask=np.isnan(np.nanmean(prec_MB, axis = 0))), weights=MLCM_wgts_MB) # Prec_MB_macro
score_table[1,5] = np.ma.average(np.ma.array(np.nanmean(rec_MB, axis = 0), mask=np.isnan(np.nanmean(rec_MB, axis = 0))), weights=MLCM_wgts_MB) # Rec_MB_macro
score_table[2,5] = np.ma.average(np.ma.array(np.nanmean(MCC_MB, axis = 0), mask=np.isnan(np.nanmean(MCC_MB, axis = 0))), weights=MLCM_wgts_MB) # MCC_MB_macro
score_table[3,5] = BSS_MB_mean_macro_weights # BSS_MB_macro


# MV micro
score_table[0,6] = Prec_MB_majority_micro
score_table[1,6] = Rec_MB_majority_micro
score_table[2,6] = MCC_MB_majority_micro

# MV macro no weights
score_table[0,7] = np.ma.average(np.ma.array(prec_MV, mask=np.isnan(prec_MV)), weights=None) # Prec_MB_majority_macro
score_table[1,7] = np.ma.average(np.ma.array(rec_MV, mask=np.isnan(rec_MV)), weights=None) # Rec_MB_majority_macro
score_table[2,7] = np.ma.average(np.ma.array(MCC_MV, mask=np.isnan(MCC_MV)), weights=None) # MCC_MB_majority_macro

# AB macro weighted
MLCM_wgts_MV = np.sum(MLCM_MV, axis = 1)[:30]
score_table[0,8] = np.ma.average(np.ma.array(prec_MV, mask=np.isnan(prec_MV)), weights=MLCM_wgts_MV) # Prec_MB_majority_macro
score_table[1,8] = np.ma.average(np.ma.array(rec_MV, mask=np.isnan(rec_MV)), weights=MLCM_wgts_MV) # Rec_MB_majority_macro
score_table[2,8] = np.ma.average(np.ma.array(MCC_MV, mask=np.isnan(MCC_MV)), weights=MLCM_wgts_MV) # MCC_MB_majority_macro

print(pd.DataFrame(score_table, index = ['Precision', 'Recall', 'MCC', 'BSS'],
                   columns = ['AB micro', 'AB macro', 'AB macro (w)', 'MB micro', 'MB macro',
                              'MB macro (w)', 'MV micro', 'MV macro', 'MV macro (w)']))

print('Subset Accuracy AB = {:.6f}'.format(subset_acc_AB))
print('Subset Accuracy MB = {:.6f}'.format(np.mean(subset_acc_MB)))
print('Subset Accuracy MV = {:.6f}'.format(subset_acc_MV))

print('Hamming Loss AB = {:.6f}'.format(Hamming_AB))
print('Hamming Loss MB = {:.6f}'.format(np.mean(Hamming_MB)))
print('Hamming Loss MV = {:.6f}'.format(Hamming_MV))


