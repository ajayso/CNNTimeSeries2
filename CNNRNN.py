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
from keras.layers import Embedding, Flatten, Dense,SimpleRNN,LSTM
from keras import layers
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



class CNN_RNN:
    # Load the data, text and labels
    def load_data(self):
        data_dir = os.getcwd()
        fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
        f = open(fname)
        data = f.read()
        f.close()
        lines = data.split('\n')
        header = lines[0].split(',')
        lines = lines[1:]
        float_data = np.zeros((len(lines), len(header) - 1))
        for i, line in enumerate(lines):
            values = [float(x) for x in line.split(',')[1:]]
            float_data[i, :] = values
        mean = float_data[:20000].mean(axis=0)
        float_data -= mean
        std = float_data[:200000].std(axis=0)
        float_data /= std
        self.data =float_data
        
    def generator(self, data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)
    
            samples = np.zeros((len(rows),
                               lookback // step,
                               data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]
            yield samples, targets
    
   
    
    def Train_Execute_Model(self):
        lookback = 1440
        step = 6
        delay = 144
        batch_size = 128
        float_data = self.data
        
        train_gen =self.generator(float_data,
                              lookback=lookback,
                              delay=delay,
                              min_index=0,
                              max_index=200000,
                              shuffle=True,
                              step=step, 
                              batch_size=batch_size)
        val_gen = self.generator(float_data,
                            lookback=lookback,
                            delay=delay,
                            min_index=200001,
                            max_index=300000,
                            step=step,
                            batch_size=batch_size)
        test_gen = self.generator(float_data,
                             lookback=lookback,
                             delay=delay,
                             min_index=300001,
                             max_index=None,
                             step=step,
                             batch_size=batch_size)
        
        # This is how many steps to draw from `val_gen`
        # in order to see the whole validation set:
        val_steps = (300000 - 200001 - lookback) // batch_size
        
        # This is how many steps to draw from `test_gen`
        # in order to see the whole test set:
        test_steps = (len(float_data) - 300001 - lookback) // batch_size
        
        model = Sequential()
        model.add(layers.Conv1D(32, 5, activation='relu',
                                input_shape=(None, float_data.shape[-1])))
        model.add(layers.MaxPooling1D(3))
        model.add(layers.Conv1D(32, 5, activation='relu'))
        
        model.add(layers.GRU(32, dropout = 0.2, recurrent_dropout = 0.2, input_shape=(None, float_data.shape[-1])))
        model.add(layers.Dense(1))
        model.summary()
        
        model.compile(optimizer=RMSprop(), loss='mae')
        history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
        self.history = history
    
    
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
        
    
        

        
        
        
rnnstep2 = CNN_RNN()
rnnstep2.load_data()
rnnstep2.Train_Execute_Model()
#rnnstep2.plot()
