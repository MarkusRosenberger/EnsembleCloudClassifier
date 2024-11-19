#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mrosenberger
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import matplotlib.pyplot as plt
import csv
import h5py
import Analysis as ana
import Modelfunctions as mfunc

# define number of cores to use in parallel
tf.config.threading.set_intra_op_parallelism_threads(num_threads = 16)
tf.config.threading.set_inter_op_parallelism_threads(1)

# define some hyperparameters 
n_classes = 30
n_residual_layers = 2
n_residual_blocks = 2
n_filters = [16, 32, 64, 128]
initial_lr = 1e-4
weight_decay = 1e-3
EPOCHS = 2
BATCH_SIZE = 16

# create directory for logs if it doesn't exist already
log_dir= os.path.join(os.getcwd(), 'logs')
if not os.path.isdir(log_dir): os.makedirs(log_dir)

# load observations and images
with open('GroundTruth_train.csv', mode ='r') as file:
        gt_train = np.array([line for line in csv.reader(file)]).astype(int)

with open('GroundTruth_val.csv', mode ='r') as file:
        gt_val = np.array([line for line in csv.reader(file)]).astype(int)

with h5py.File('img_train.nc', mode='r') as f:
    img_train = f['Full_period'][:]/255

with h5py.File('img_val.nc', mode='r') as f:
    img_val = f['Full_period'][:]/255


# initialise model
model = mfunc.make_residual_model(input_shape = np.shape(img_train[0]), n_classes = n_classes, n_residual_layers = n_residual_layers, n_filters = n_filters, n_resblocks = n_residual_blocks)

# define loss function
loss_fn = mfunc.Brier_Score(from_logits = False)

# define gradient descent method
optimizer = tf.keras.optimizers.Adam(learning_rate = initial_lr, weight_decay=weight_decay) 

# compile model
model.compile(optimizer=optimizer,
    loss=loss_fn,
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    loss_weights=None
   )

# define learning rate schedule
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.85,
    patience=7,
    verbose=0,
    mode='auto',
    min_delta=0.01,
    cooldown=0,
    min_lr=0,
)

# define early stopping condition
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=8,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0
)

# create dataset for training and validation
ds = tf.data.Dataset.from_tensor_slices((img_train, gt_train))
ds = (ds.shuffle(len(img_train))
      .map(mfunc.shuffle_directions, num_parallel_calls = AUTOTUNE)
      .cache()
      .batch(BATCH_SIZE)
      .prefetch(AUTOTUNE)
      )

ds_val = tf.data.Dataset.from_tensor_slices((img_val, gt_val))
ds_val = (ds_val.shuffle(len(img_val))
      .map(mfunc.shuffle_directions, num_parallel_calls = AUTOTUNE)
      .cache()
      .batch(BATCH_SIZE)
      .prefetch(AUTOTUNE)
      )


# train model
history = model.fit(ds, epochs = EPOCHS,
          validation_data = ds_val, callbacks=[reduce_lr, early_stopping])

# save model
model.save('Model_{:04d}epochs'.format(EPOCHS))

# save training history in log files
epochs_until_stop = len(history.history['loss'])

# Training loss 
df = pd.DataFrame(data={'Epoch': np.arange(epochs_until_stop)+1, 'Loss': history.history['loss']}, dtype = 'float32')
df.to_csv(os.path.join(log_dir, 'trainloss_log.txt'), sep=',', index = False)

# Validation loss, recall, and precision
df = pd.DataFrame(data={'Epoch': np.arange(epochs_until_stop)+1, 'Loss': history.history['val_loss'],
                        'Precision': history.history['val_precision'], 'Recall': history.history['val_recall']}, dtype = 'float32')
df.to_csv(os.path.join(log_dir, 'val_log.txt'), sep=',', index = False)

# Training precision and recall
df = pd.DataFrame(data={'Epoch': np.arange(epochs_until_stop)+1,
                        'Precision': history.history['precision'], 'Recall': history.history['recall']}, dtype = 'float32')
df.to_csv(os.path.join(log_dir, 'trainacc_log.txt'), sep=',', index = False)

# Learing rate
df = pd.DataFrame(data={'Epoch': np.arange(epochs_until_stop)+1,
                        'Learning Rate': history.history['lr']}, dtype = 'float32')
df.to_csv(os.path.join(log_dir, 'lr_log.txt'), sep=',', index = False)

# MLCM
# plot absolute values, per-class precision, and per-class recall
predictions = model(img_val)

MLCM, fig1, fig2, fig3 = ana.MLCM(ground_truth = gt_val, predicted_classes= predictions, types = [0,1,2,3,4,5,6,7,8,9]*3, from_logits=False, cmap = 'Reds')
fig1.savefig(os.path.join(log_dir, 'MLCM_absolute.png'), bbox_inches = 'tight', dpi = 300)
fig2.savefig(os.path.join(log_dir, 'MLCM_recall.png'), bbox_inches = 'tight', dpi = 300)
fig3.savefig(os.path.join(log_dir, 'MLCM_precision.png'), bbox_inches = 'tight', dpi = 300)
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)

# reliability diagram for each class
types = [r'C$_L$ = 0', r'C$_L$ = 1', r'C$_L$ = 2', r'C$_L$ = 3', r'C$_L$ = 4', r'C$_L$ = 5', r'C$_L$ = 6', r'C$_L$ = 7', r'C$_L$ = 8', r'C$_L$ = 9', 
         r'C$_M$ = 0', r'C$_M$ = 1', r'C$_M$ = 2', r'C$_M$ = 3', r'C$_M$ = 4', r'C$_M$ = 5', r'C$_M$ = 6', r'C$_M$ = 7', r'C$_M$ = 8', r'C$_M$ = 9', 
         r'C$_H$ = 0', r'C$_H$ = 1', r'C$_H$ = 2', r'C$_H$ = 3', r'C$_H$ = 4', r'C$_H$ = 5', r'C$_H$ = 6', r'C$_H$ = 7', r'C$_H$ = 8', r'C$_H$ = 9']

for i in range(len(types)):

    fig, _ = ana.ReliabilityDiagram(predictions[:,i], gt_val[:,i], n_bins = 10, title_string = types[i]) 
    plt.savefig(os.path.join(log_dir, 'ReliabilityDiagram' + '_' + types[i] + '.png'), bbox_inches = 'tight', dpi = 300)
    plt.close()

