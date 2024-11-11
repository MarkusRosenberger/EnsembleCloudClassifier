#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mrosenberger
"""

import os
import numpy as np
import csv
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

import Analysis as ana
import Modelfunctions as mfunc

# directory, in which the model runs are stored
# ens_dir = '<path_to_directory>'
ens_dir = '/srvfs/home/mrosenberger/CNNResults/Icos/Einzelbilder/FinalTraining/v18_adddenselayer_shuffledinput_shuffledvalidation_BSmeanLoss_2resblocks_finaltraining/ModelRuns/'

# load observations and images
with open('GroundTruth_outofsample.csv', mode ='r') as file:
    gt_oos = np.array([line for line in csv.reader(file)]).astype(int)

with open('GroundTruth_test.csv', mode ='r') as file:
    gt_test = np.array([line for line in csv.reader(file)]).astype(int)

with open('GroundTruth_raw.csv', mode ='r') as file:
    gt_raw = np.array([line for line in csv.reader(file)]).astype(int)

with h5py.File('img_outofsample.nc', mode='r') as f:
    img_oos = f['Full_period'][:]/255

with h5py.File('img_test.nc', mode='r') as f:
    img_test = f['Full_period'][:]/255

# number of observations per class in raw dataset
# only necessary for plot
obs_per_class_raw = np.sum(gt_raw, axis = 0)

# array of model runs
model_vec = [tf.keras.models.load_model(os.path.join(ens_dir, m, 'discriminator_0200epochs'), custom_objects={'Brier_Score':mfunc.Brier_Score()})
             for m in np.sort(os.listdir(ens_dir))]

# array of predicted probabilities of each model
# evaluated on test data and out-of-sample data
predictions_MB = np.array([model(img_test) for model in model_vec]) 
predictions_MB_oos = np.array([model(img_oos) for model in model_vec]) 

# MLCM calculated using mean of predicted probabilities of all ensemble members --> Average Based
MLCM_AB = ana.MLCM(gt_test, np.mean(predictions_MB, axis = 0), thresh = 0.5, return_plot = False, from_logits = False)
MLCM_AB_oos = ana.MLCM(gt_oos, np.mean(predictions_MB_oos, axis = 0), thresh = 0.5, return_plot = False, from_logits = False)

# P, R, MCC for each class; no bootstrap
Prec_test = []
Rec_test = []
MCC_test = []

Prec_oos = []
Rec_oos = []
MCC_oos = []

for i in range(30):

    # test data
    output = ana.calculate_measures(MLCM_AB, index = i, measures = ['Precision', 'Recall', 'MCC'], do_bootstrap=False)
    Prec_test.append(output['Precision'])
    Rec_test.append(output['Recall'])
    MCC_test.append(output['MCC'])

    # out of sample data
    output = ana.calculate_measures(MLCM_AB_oos, index = i, measures = ['Precision', 'Recall', 'MCC'], do_bootstrap=False)
    Prec_oos.append(output['Precision'])
    Rec_oos.append(output['Recall'])
    MCC_oos.append(output['MCC'])

# Plot
fs = 11
items = ['a)', 'b)', 'c)']

fig, ax = plt.subplots(3,1, figsize = (8,12), sharex = True, sharey = True)
fig.subplots_adjust(hspace = 0.1)

p = ax[0].scatter(obs_per_class_raw, Prec_test, marker = 'x')
ax[0].scatter(obs_per_class_raw, Prec_oos, marker = 'o')

ax[0].set_ylabel('Precision')

ax[1].scatter(obs_per_class_raw, Rec_test, marker = 'x')
ax[1].scatter(obs_per_class_raw, Rec_oos, marker = 'o')
ax[1].set_ylabel('Recall')

h1 = ax[2].scatter(obs_per_class_raw, MCC_test, marker = 'x', label = 'Test data')
h2 = ax[2].scatter(obs_per_class_raw, MCC_oos, marker = 'o', label = 'Out-of-sample data')
ax[2].set_ylabel('MCC')

ax[2].set_xlabel('Number of obervations in raw data')
ax[2].legend(handles = [h1, h2])

for axi, axs in enumerate(ax):
    axs.text(x = -.09, y = 1.03, s = items[axi], ha = 'center', va = 'center', fontsize = fs+1, fontweight = 'bold', transform=axs.transAxes)

plt.savefig('TestvsOos.png', bbox_inches = 'tight', dpi = 500)
