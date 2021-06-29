#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:19:49 2020

@author: rafaelsilva
"""

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


tf.random.set_seed(91195003)
np.random.seed(91195003)
#for an easy reset backend session state
tf.keras.backend.clear_session()


# Read csv file
def load_data(file):
    df = pd.read_csv(file, encoding='utf-8', index_col='Index')
    return df

# Prepare data
def data_prepare(df, test_size=0.1):
    # target column
    target = df['criteria_final']
    # features columns
    features = df.drop(columns=['criteria_final'])
    return features, target

# Normalize data
def normalize_data(df, norm_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=norm_range)
    normalized_df = scaler.fit_transform(df.values)
    new_df = pd.DataFrame(normalized_df, columns=df.columns, index=df.index)
    return new_df, scaler

# One hot enconde
def onehot_encoder(df):
    ohe = OneHotEncoder()
    values = df.values.reshape(-1, 1)
    transformed = ohe.fit_transform(values).toarray()
    y = pd.DataFrame(transformed, index=df.index, columns=ohe.get_feature_names())
    return y, ohe

# Create model
def build_model(input_dim, output_dim, drop_out):
    # Neural network
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
    
    print(model.summary())

    tf.keras.utils.plot_model(model,'neuralNetwork.png', show_shapes=True)

    return model

def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('plot_acc.png')
    plt.show()
        
def plot_loss(history):
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.savefig('plot_loss.png')
    plt.show()
        
#=====================================================================================
#
#                            Main Execution
#
#=====================================================================================

nr_neurons = 128
epochs = 500
droup_out = 0.4
batch_size = 128
verbose = 1
patience = 30

# File
file_path = 'dataset_labelled.csv'

# Load data
df = load_data(file_path)

# Prepare data
features, target = data_prepare(df)

# Normalize data
X, scaler = normalize_data(features)

# One hot enconde target column
y, ohe = onehot_encoder(target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Create model
model = build_model(X.shape[1], y.shape[1], droup_out)

# Fit model
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=patience, mode='auto', min_lr=0.00005)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=verbose, mode='auto')

history = model.fit(X_train, y_train, 
                    validation_data = (X_test, y_test), 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    verbose=verbose, 
                    callbacks=[lr,early_stop])

print('History:', history.history.keys)

# Plot accuracy
plot_accuracy(history)

# Plot loss
plot_loss(history)

# Score of model
scores = model.evaluate(X_test, y_test, verbose=verbose)
print('Evaluation %s: %s' %(model.metrics_names, str(scores)))

# Model save
model.save('model_mqrcs.h5')

# Scaler save / OneHot Save
minMaxScaler_filename = 'minMaxScaler.pkl'
oneHotEncoder_filename = 'oneHotEncoder.pkl'

joblib.dump(scaler, minMaxScaler_filename)
joblib.dump(ohe, oneHotEncoder_filename)
