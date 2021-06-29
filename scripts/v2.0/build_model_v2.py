#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:19:49 2020

@author: rafaelsilva
"""

import shutil
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras import Sequential
from sklearn.externals import joblib 
from sklearn.model_selection import KFold

tf.random.set_seed(91195003)
np.random.seed(91195003)
#for an easy reset backend session state
tf.keras.backend.clear_session()

shutil.rmtree('models') 
os.mkdir('models')
shutil.rmtree('plots') 
os.mkdir('plots')


# Read csv file
def load_data(file):
    df = pd.read_csv(file, encoding='utf-8', index_col='Index')
    return df

# Prepare data
def data_prepare(df, test_size=0.2, norm_range=(-1,1)):
    # target column
    target = df['criteria_final']
    # features columns
    features = df.drop(columns=['criteria_final'])
    
    # normalize data
    scaler = MinMaxScaler(feature_range=norm_range)
    normalized_features = scaler.fit_transform(features)
    new_features = pd.DataFrame(normalized_features, columns=features.columns, index=features.index)
    
    # one hot enconde
    ohe = OneHotEncoder()
    values = target.values.reshape(-1, 1)
    transformed = ohe.fit_transform(values).toarray()
    new_target = pd.DataFrame(transformed, index=target.index, columns=ohe.get_feature_names())
    
    # join all data again
    new_df = new_features.join(new_target)
    
    # split data
    train, test = train_test_split(new_df, test_size=test_size)
    
    columns_target = ['x0_0','x0_1','x0_2','x0_3']
    
    Y_train = pd.DataFrame(train, columns=columns_target)
    Y_test = pd.DataFrame(test, columns=columns_target)
    
    X_train = train.drop(columns=columns_target)
    X_test = test.drop(columns=columns_target)
    
    return X_train.values, X_test.values, Y_train.values, Y_test.values, scaler, ohe

# Create model
def build_model(input_dim, output_dim, drop_out):
    model = Sequential()
    model.add(Dense(int(nr_neurons*2), input_dim=input_dim, activation='relu', name='first_layer'))
    model.add(Dropout(drop_out, name='drop_out_1'))
    model.add(Dense(int(nr_neurons*4), activation='relu', name='hidden_layer_1'))
    model.add(Dropout(drop_out, name='drop_out_2'))
    model.add(Dense(int(nr_neurons*2), activation='relu', name='hidden_layer_2'))
    model.add(Dropout(drop_out, name='drop_out_3'))
    model.add(Dense(nr_neurons, activation='relu', name='hidden_layer_3'))
    model.add(Dropout(drop_out, name='drop_out_4'))
    model.add(Dense(int(nr_neurons/2), activation='relu', name='hidden_layer_4'))
    model.add(Dropout(drop_out, name='drop_out_5'))
    model.add(Dense(int(nr_neurons/4), activation='relu', name='hidden_layer_6'))
    model.add(Dropout(drop_out, name='drop_out_6'))
    model.add(Dense(output_dim, activation='softmax', name='output_layer'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    tf.keras.utils.plot_model(model,'neuralNetwork.png', show_shapes=True)

    return model

def plot_accuracy(history, count):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('plots/plot_acc_' + count + '.png')
    plt.show()
        
def plot_loss(history, count):
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.savefig('plots/plot_loss_' + count + '.png')
    plt.show()
        
#=====================================================================================
#
#                            Main Execution
#
#=====================================================================================

nr_neurons = 128
epochs = 200
droup_out = 0.4
batch_size = 128
verbose = 1
patience = 20
n_splits = 15
seed = 91195003

# File
file_path = '../../datasets/final/dataset_labelled.csv'

# Load data
df = load_data(file_path)

# Prepare data
X_train, X_test, Y_train, Y_test, scaler, ohe = data_prepare(df)

# KFold
cvscores = []
count = 0;
for train_index,test_index in KFold(n_splits).split(X_train):
    print('--- Begin Cross Validation Nº' + str(count) +' ---')
    x_train, x_test = X_train[train_index], X_train[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]

    # Create model
    model = build_model(X_train.shape[1], Y_train.shape[1], droup_out)

    # Fit model
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=verbose, mode='auto')
    #checkpoint = tf.keras.callbacks.ModelCheckpoint('models/model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', verbose=0, monitor='val_loss', save_best_only=True, mode='auto')  
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=patience, mode='auto', min_lr=0.00005)

    print('Fit the model, please wait...')
    history = model.fit(x_train, y_train, 
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size, 
                        verbose=0, 
                        callbacks=[early_stop,lr])

    # Plot accuracy
    plot_accuracy(history, str(count))
    # Plot loss
    plot_loss(history, str(count))
    count = count + 1;
    
    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=verbose)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)

# Final score
print("Final score: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Scaler save / OneHot Save
minMaxScaler_filename = 'minMaxScaler.pkl'
oneHotEncoder_filename = 'oneHotEncoder.pkl'

joblib.dump(scaler, minMaxScaler_filename)
joblib.dump(ohe, oneHotEncoder_filename)
