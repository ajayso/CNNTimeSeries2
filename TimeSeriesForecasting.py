# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:51:44 2019

@author: Ajay Solanki
"""

#Imports

import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Flatten, Dense,Activation
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.utils.np_utils import to_categorical
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import pandas as pd
import quandl 



class CNN_TS:
    # Load the data, text and labels
    def load_data(self):
        quandl.ApiConfig.api_key ="Td2oA_m_SYUdi1X9Htdi" # Please replace with API key can get one from quandl.com
        data = quandl.get("NSE/NIFTY_AUTO")
        data = data.dropna()
        print(data.shape)
        openp  = data.Open
        openp  = openp.values
        highp = data.High
        highp = highp.values
        lowp= data.Low
        lowp = lowp.values
        closep = data.Close
        closep = closep.values
        volumep  = data['Shares Traded']
        volumep = volumep.values
        WINDOW = 30
        EMB_SIZE = 5
        STEP = 1
        FORECAST = 1
        X, Y = [], []
        for i in range(0, len(data), STEP): 
            try:
                
                o = openp[i:i+WINDOW]
                h = highp[i:i+WINDOW]
                l = lowp[i:i+WINDOW]
                c = closep[i:i+WINDOW]
                v = volumep[i:i+WINDOW]
        
                o = (np.array(o) - np.mean(o)) / np.std(o)
                h = (np.array(h) - np.mean(h)) / np.std(h)
                l = (np.array(l) - np.mean(l)) / np.std(l)
                c = (np.array(c) - np.mean(c)) / np.std(c)
                v = (np.array(v) - np.mean(v)) / np.std(v)
                
                if (i+WINDOW+FORECAST >= len(data)):
                    break
                
        
                x_i = closep[i:i+WINDOW]
                y_i = closep[i+WINDOW+FORECAST]  
        
                last_close = x_i[-1]
                next_close = y_i
        
                if last_close < next_close:
                    y_i = [1, 0]
                else:
                    y_i = [0, 1] 
        
                x_i = np.column_stack((o, h, l, c, v))
                
        
            except Exception as e:
                print("Error---" + e)
                break
        
            X.append(x_i)
            Y.append(y_i)
        
        X, Y = np.array(X), np.array(Y)
        X_train, X_test, Y_train, Y_test = self.create_Xt_Yt(X, Y)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))
 
        self.X_train,self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test
        print(len(self.X_train))
        
    
   
    
    def Train_Build_Model(self):
  
        WINDOW = 30
        EMB_SIZE = 5
        STEP = 1
        FORECAST = 1    
     
        model = Sequential()
        model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE),
                        nb_filter=16,
                        filter_length=4,
                        border_mode='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Convolution1D(nb_filter=8,
                        filter_length=4,
                        border_mode='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dense(64))
        model.add(Dense(2))
        model.add(Activation('softmax'))
             
        model.summary()
        self.model = model
    
    
    
    def execute(self):
        opt = Nadam(lr=0.002)
        self.model.compile(
                optimizer = opt,
                loss = 'categorical_crossentropy',
                metrics=['acc'])
                
        self.history = self.model.fit(self.X_train, self.Y_train,
                       epochs = 100,
                       batch_size = 128,
                       validation_data = (self.X_test, self.Y_test)
                       )
                       
        self.model.save_weights('cnn_ts.h5')
    
    def plot(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
    
    def shuffle_in_unison(self,a, b):
        # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b
    
    def create_Xt_Yt(self,X, y, percentage=0.9):
        p = int(len(X) * percentage)
        X_train = X[0:p]
        Y_train = y[0:p]
         
        #X_train, Y_train = self.shuffle_in_unison(X_train, Y_train)
     
        X_test = X[p:]
        Y_test = y[p:]
    
        return X_train, X_test, Y_train, Y_test
   
        

        
        
        
cnn_ts = CNN_TS()
cnn_ts.load_data()

cnn_ts.Train_Build_Model()
cnn_ts.execute()
cnn_ts.plot()
