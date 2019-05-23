# -*- coding: utf-8 -*-
"""
Created on Fri May 17 21:59:01 2019

@author: founder
"""

import tensorflow as tf
import numpy as np
import random
from sklearn.preprocessing import StandardScaler


def function(data):

    tf.reset_default_graph()
    
    tf.set_random_seed(1)   
    np.random.seed(1)
    
    # Hyper Parameters
    epoch = 40000
    BATCH_SIZE = 32         
    NumOfLine = 16
    LR_G = 0.0001          
    LR_D = 0.0001           
    N_IDEAS = 5           
    NumOfF = data.shape[1]            

    BATCH_SIZE = data.shape[0]
    if BATCH_SIZE > 32:
        BATCH_SIZE = 32
    NumOfLine = int(BATCH_SIZE / 2)     

    #ss = StandardScaler()
    #data = ss.fit_transform(data)
    
    def Cwork(d):
        clist = random.sample(range(BATCH_SIZE), NumOfLine)
        datause = np.zeros(shape=(NumOfLine, NumOfF))
        j = 0
        for c in clist:
            datause[j] = d[c]
            j = j + 1
            
        return datause
    
    
    with tf.variable_scope('Generator'):                           
        G_in = tf.placeholder(tf.float32, [None, N_IDEAS])         
        G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
        G_out = tf.layers.dense(G_l1, NumOfF)             
    
    with tf.variable_scope('Discriminator'):
        real_f = tf.placeholder(tf.float32, [None, NumOfF], name='real_in')
        D_l0 = tf.layers.dense(real_f, 128, tf.nn.relu, name='l')
        p_real = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')             
        # reuse layers for generator
        D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='l', reuse=True)          
        p_fake = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True) 
    
    
    D_loss = -tf.reduce_mean(tf.log(p_real) + tf.log(1-p_fake))           
    G_loss = tf.reduce_mean(tf.log(1-p_fake))
    
    
    train_D = tf.train.AdamOptimizer(LR_D).minimize(
        D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
    train_G = tf.train.AdamOptimizer(LR_G).minimize(
        G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())                  

    
    for step in range(epoch):
        dataused = Cwork(data)
        G_ideas = np.random.randn(int(data.shape[0]/2), 5)
        G_paintings, pa0, Dl = sess.run([G_out, p_real, D_loss, train_D, train_G],  
                                        {G_in: G_ideas, real_f: dataused})[:3]
        
        if step == epoch - 36:  
            G_out_final = G_paintings
             
        if step >= epoch - 35:
            G_out_final = np.vstack((G_paintings, G_out_final))
    
    #G_out_final = ss.inverse_transform(G_out_final)   
    
    value_G = G_out_final
    value_G = np.vstack((data, value_G))
    
    return value_G
