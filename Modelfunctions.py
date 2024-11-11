#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mrosenberger
"""

import tensorflow as tf
from keras import layers

##########################################################################################################################

'''
Brier Score as loss function
'''
class Brier_Score(tf.keras.losses.Loss):
    def __init__(self, from_logits = False, **kwargs):
        super().__init__(**kwargs)

        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        
        # convert to multi-label probabilities if necessary 
        if self.from_logits == True:
            y_pred = tf.keras.activations.sigmoid(y_pred)
        
        else:
            y_pred = tf.cast(y_pred, dtype = 'float32')
        
        y_true = tf.cast(y_true, dtype = 'float32')
        loss = tf.reduce_sum((y_true - y_pred)**2, axis = 1)
    
        return tf.reduce_mean(loss)
        
    def get_config(self):
        config = super().get_config()
        config.update({'from_logits':self.from_logits})
        return config

    @classmethod
    def from_config(cls, config):
        
        return cls(**config)
    
###########################################################################################################################

'''
random sub-image shuffling at each epoch
'''
def shuffle_directions(image, label):

    # randomly shuffle directions and aplly to image
    shuffle_indices = tf.random.shuffle([0,1,2,3])
    image_shuffled = [image[shuffle_indices[0]], image[shuffle_indices[1]], image[shuffle_indices[2]], image[shuffle_indices[3]]]
    
    return (image_shuffled, label)

###########################################################################################################################

'''
Stacked Layer consisting of:
    -) 2D convolution, kernel size = 5, stride = 2
    -) Batch normalization
    -) Leaky ReLU activation
'''
class stacked_layer(layers.Layer):
    def __init__(self, n_filters, kernel_size = 5, strides = 2):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.convlayer = layers.Conv2D(self.n_filters, kernel_size = self.kernel_size, strides = self.strides, padding='same', kernel_initializer = tf.keras.initializers.glorot_normal(seed = 42))
        self.batchnorm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU()
        
    def call(self, x):
            
        x = self.convlayer(x)
        x = self.batchnorm(x)
        x = self.activation(x)    
        
        return x

#####################################################################################################################################################

'''
Residual layer consisting of:
    -) 2D convolution, kernel size = 3, stride = 1
    -) Layer Normalization
    -) Leaky ReLU activation
    -) 2D convolution, kernel size = 3, stride = 1
    -) Add Input
    -) Layer Normalization
    -) Leaky ReLU activation
    -) Dropout with ratio 0.2
'''

class residual_layer(layers.Layer):     
    def __init__(self, n_layers, n_filters):
        super().__init__()
        self.n_layers = n_layers # number of convolutional layers
        self.n_filters = [n_filters]*self.n_layers # number of filters
        self.kernel_size = [3]*self.n_layers # kernel size
        self.strides = [1]*self.n_layers # stride
        self.convlayer = [layers.Conv2D(filters = f, kernel_size = k, strides = s, padding = 'same', kernel_initializer = tf.keras.initializers.glorot_normal(seed = 42)) for (f, k, s) in zip(self.n_filters, self.kernel_size, self.strides)]
        self.layernorm = layers.LayerNormalization()
        self.activation = layers.LeakyReLU()
        self.dropout = layers.Dropout(0.2)
        self.add = layers.Add()
    
    def call(self, x):
        residual = x
        for l in range(self.n_layers):
            x = self.convlayer[l](x)
            
            if l == self.n_layers -1:
                x = self.add([x, residual])
                x = self.layernorm(x)
                x = self.activation(x)  
                x = self.dropout(x)   
        
            else:
                x = self.layernorm(x)
                x = self.activation(x)  
                
        return x

#####################################################################################################################################################

'''
Build model using above defined classes
'''

def make_residual_model(input_shape = [4, 64, 100, 3], n_classes = 30, n_residual_layers = 2, n_filters = [16, 32, 64, 128], n_resblocks = 1):   
    
    in_img = layers.Input(shape = input_shape, name='Discriminator_Input_Layer') # Input Layer
    
    x = layers.Conv2D(8, (7, 7), strides=(2, 2), padding='same', kernel_initializer = tf.keras.initializers.glorot_normal(seed = 42))(in_img) # First convolution
    x = layers.LeakyReLU()(x)
    
    for f in n_filters:
        x = stacked_layer(n_filters = f)(x)
        
        if n_residual_layers == 0:
            pass
        else:
            for i in range(n_resblocks):
                x = residual_layer(n_residual_layers, f)(x)

    # split dirtections and flatten each one
    x0, x1, x2, x3 = tf.unstack(x, axis = 1)
    
    x0 = layers.Flatten(name = 'Flatten_1')(x0)
    x1 = layers.Flatten(name = 'Flatten_2')(x1)
    x2 = layers.Flatten(name = 'Flatten_3')(x2)
    x3 = layers.Flatten(name = 'Flatten_4')(x3)

    # first dense layer for each direction separately
    out0 = tf.keras.layers.Dense(n_classes, activation = 'sigmoid', kernel_initializer = tf.keras.initializers.glorot_normal(seed = 42), name = 'Pre-classification_1')(x0)
    out1 = tf.keras.layers.Dense(n_classes, activation = 'sigmoid', kernel_initializer = tf.keras.initializers.glorot_normal(seed = 42), name = 'Pre-classification_2')(x1)
    out2 = tf.keras.layers.Dense(n_classes, activation = 'sigmoid', kernel_initializer = tf.keras.initializers.glorot_normal(seed = 42), name = 'Pre-classification_3')(x2)
    out3 = tf.keras.layers.Dense(n_classes, activation = 'sigmoid', kernel_initializer = tf.keras.initializers.glorot_normal(seed = 42), name = 'Pre-classification_4')(x3)
    
    # combine directions again
    output = layers.Concatenate(name = 'Concatenate')([out0, out1, out2, out3])
    
    # calculate final predictions
    output = layers.Dense(n_classes, activation = 'sigmoid', kernel_initializer = tf.keras.initializers.glorot_normal(seed = 42), name = 'Final_Classification')(output)
    
    # combine everything into a model
    model = tf.keras.models.Model(in_img, output, name='CloudClassifier')
    
    return model

#####################################################################################################################################################

