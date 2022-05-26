#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras import layers
import os


def mix_source_target(x_source, y_source, x_target, y_target, mix_prop  = 0.0):
    
    seed_prev = np.random.get_state()[1][0]
    np.random.seed(0)
    # shuffle target sample orders
    n_target = y_target.shape[0]
    index_shuffle = np.random.choice(n_target, n_target, replace = False)
    x_target = x_target[index_shuffle]
    y_target = y_target[index_shuffle]
    
    
    # delete mix_prop samples from target and add them in source
    
    y_unique = np.unique(y_source)
    n_classes = y_unique.shape[0]
    
    xs_list, xt_list = [], []
    ys_list, yt_list = [], []
    is_target_list = []
    
    for i in range(n_classes):
        xsi, ysi = x_source[y_source == i], y_source[y_source == i]
        xti, yti = x_target[y_target == i], y_target[y_target == i]
        n_mix = int(ysi.shape[0] * mix_prop)
        is_target_i = np.array([0] * xsi.shape[0] + [1] * n_mix)
        xsi = np.concatenate([xsi, xti[:n_mix]], axis = 0)
        xti = xti[n_mix:]
        ysi = np.concatenate([ysi, yti[:n_mix]], axis = 0)
        yti = yti[n_mix:]
        is_target_list.append(is_target_i)
        xs_list.append(xsi)
        xt_list.append(xti)
        ys_list.append(ysi)
        yt_list.append(yti)
        
    x_source = np.concatenate(xs_list, axis = 0)
    x_target = np.concatenate(xt_list, axis = 0)
    y_source = np.concatenate(ys_list, axis = 0)
    y_target = np.concatenate(yt_list, axis = 0)
    is_target = np.concatenate(is_target_list, axis = 0)
    
    
    # shuffle source sample orders
    n_source = y_source.shape[0]
    index_shuffle = np.random.choice(n_source, n_source, replace = False)
    x_source = x_source[index_shuffle]
    y_source = y_source[index_shuffle]
    is_target = is_target[index_shuffle]
    
    np.random.seed(seed_prev)
    
    return x_source, y_source, x_target, y_target, is_target
    



def final_classifier(x, y, weights):
    # setting the parameters 
    lr = 1e-2                                   
    optimizer = keras.optimizers.Adam(lr)        
    batch_size = 250                             
    epochs = 20                            
    latent_dimension = 50                        
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
    model = keras.models.Sequential([
       layers.Conv2D(32, (3, 3), activation = 'relu'),
       layers.MaxPool2D((2,2)),
       layers.Flatten(),
       layers.Dense(100, activation='relu', ),
       layers.Dense(latent_dimension,),
       layers.Dense(np.unique(y).shape[0],),
       ])
    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    model.fit(x, y, batch_size = batch_size, epochs = epochs, verbose = 1, sample_weight=weights)
    return model



def TS(data, test_size = 0.2):
    x_target, y_target = data
    Ttr_x, Tts_x, Ttr_y, Tts_y = train_test_split(x_target, y_target,\
                                                  test_size = test_size, random_state=0)
    return Ttr_x, Ttr_y, Tts_x, Tts_y
    
    


def BREEDS(mix_prop = 0.0, seed = 0, test_size = 0.2, subsample = -1, path = '/Users/smaity/projects/unlabeled-transfer-learning/exponential-tilt/breeds/breeds.npy'):
    
#     mix_prop = 0.0; seed = 0; test_size = 0.2; subsample = 6000
    
    with open(path, 'rb') as f:
        Str_x = np.load(f)
        Str_y = np.load(f)
        Sts_x = np.load(f)
        Sts_y = np.load(f)
        Ttr_x = np.load(f)
        Ttr_y = np.load(f)
        Tts_x = np.load(f)
        Tts_y = np.load(f)
    del f
    
    # source data
    x_source = np.concatenate([Str_x, Sts_x], axis = 0)
    y_source = np.concatenate([Str_y, Sts_y], axis = 0)
    
    # target data
    x_target = np.concatenate([Ttr_x, Tts_x], axis = 0)
    y_target = np.concatenate([Ttr_y, Tts_y], axis = 0)
    
    # Subsample source and target    
    if subsample < 1:
        ns = x_source.shape[0]
        nt = x_target.shape[0]
    else:
        ns = int(subsample)
        nt = int(subsample)
    
    x_source, y_source = subsample_xy(x_source, y_source, ns)
    x_target, y_target = subsample_xy(x_target, y_target, nt)
    
    # mix some target samples in source sample
    x_source, y_source, x_target, y_target, is_target = mix_source_target(x_source, y_source, x_target, y_target,\
                                                               mix_prop = mix_prop)
    # train test split for source and target 
    Str_x, Sts_x, Str_y, Sts_y, Str_it, Sts_it = train_test_split(x_source, y_source, is_target,\
                                                  test_size = test_size, random_state=0)
    Ttr_x, Tts_x, Ttr_y, Tts_y = train_test_split(x_target, y_target,\
                                                  test_size = test_size, random_state=0)
      
        
    return Str_x, Str_y, Str_it, Ttr_x, Ttr_y, Sts_x, Sts_y, Sts_it, Tts_x, Tts_y








def subsample_xyit(x, y, it, n):
    xl, yl, itl = [], [], []
    y_unique, n_y = np.unique(y, return_counts=True)
    p_y = n_y / y.shape[0]
    n_s = np.floor(p_y * n)
    n_classes = y_unique.shape[0]
    for i in range(n_classes):
        ni = int(n_s[i])
        xi = x[y == i]
        xl.append(xi[:ni])
        yi = y[y == i]
        yl.append(yi[:ni])
        iti = it[y == i]
        itl.append(iti)
    xs = np.concatenate(xl, axis = 0)
    ys = np.concatenate(yl, axis = 0)
    its = np.concatenate(itl, axis = 0)
    ns = xs.shape[0]
    
    index = np.random.choice(ns, ns, replace = False)
    return xs[index], ys[index], its[index]

def subsample_xy(x, y, n):
    xl, yl = [], []
    y_unique, n_y = np.unique(y, return_counts=True)
    p_y = n_y / y.shape[0]
    n_s = np.floor(p_y * n)
    n_classes = y_unique.shape[0]
    for i in range(n_classes):
        ni = int(n_s[i])
        xi = x[y == i]
        xl.append(xi[:ni])
        yi = y[y == i]
        yl.append(yi[:ni])
        
    xs = np.concatenate(xl, axis = 0)
    ys = np.concatenate(yl, axis = 0)

    ns = xs.shape[0]
    
    seed_prev = np.random.get_state()[1][0]
    np.random.seed(123)
    index = np.random.choice(ns, ns, replace = False)
    np.random.seed(seed_prev)
    return xs[index], ys[index]
    
def pca_plot_breeds(data, index):
    Str_x, Str_y, Ttr_x, Ttr_y, Sts_x, Sts_y, Tts_x, Tts_y = data
    pca = PCA(n_components=2)
    pc = pca.fit(np.concatenate([Str_x, Ttr_x], axis = 0))
    xi = Str_x[index]
    yi = Str_y[index]
    pci = pc.transform(xi)
    pcit = pc.transform(Ttr_x)
    pcis = pc.transform(Str_x)
    n_classes = np.unique(yi).shape[0]
    row = int(np.sqrt(n_classes))
    for j in range(row):
        for i in range(row):
            idx = row * i + j
            pcii = pci[yi == idx]
            pciit = pcit[Ttr_y == idx]
            pciis = pcis[Str_y == idx]
            ax = plt.subplot(row, row, idx + 1)
            plt.scatter(pcii[:, 0], pcii[:, 1], marker = 'x', color = 'r', alpha = 0.5)
            plt.scatter(pciit[:2000, 0], pciit[:2000, 1], marker = 'o', color = 'g', alpha = 0.01)
            plt.scatter(pciis[:2000, 0], pciis[:2000, 1], marker = 'o', color = 'y', alpha = 0.01)
            plt.axis('off')
            ax.title.set_text(f'class: {idx}')
            
            
