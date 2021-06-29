#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:42:11 2020

@author: rafaelsilva
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf


tf.random.set_seed(91195003)
np.random.seed(91195003)
#for an easy reset backend session state
tf.keras.backend.clear_session()


'''
Read csv file
'''
def load_data(file):
    df = pd.read_csv(file, encoding='utf-8', index_col='Index')
    return df

'''
Prepare dataset
'''
def prepare_data(df):
    # Drop unwanted data
    drop_columns = ['Day of month', 'Month (number)']
    df_aux = df.drop(columns=drop_columns, inplace=False)
    
    return new_df

'''
Normalize dataset
'''
def normalize_data(df, norm_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=norm_range)
    
    # Normalize data
    normalized_df = scaler.fit_transform(df.values)
    
    columns = ['Cases']
    new_df = pd.DataFrame(normalized_df,columns=columns,index=df.index)
    
    return new_df, scaler

'''
Creating a supervised problem
'''
def to_supervised(df,timesteps):
    data = df.values
    X, y = list(), list()
    # Iterate over the trainnig_set to create X and y
    dataset_size = len(data)
    for curr_pos in range(dataset_size):
        # end of the input sequence is the curr_pos + the number of timesteps of the input sequence
        input_index = curr_pos + timesteps
        # end of the labels is the end of input_seq + 1
        label_index = input_index + 7
        # if we have enougth data for this sequence
        if label_index < dataset_size:
            X.append(data[curr_pos:input_index, :])
            y.append(data[input_index:label_index, 0])
    
    return np.array(X).astype('float32'), np.array(y).astype('float32')

'''
Split into training and validations sets
'''
def split_dataset(training,perc=10):
    training_idx = np.arange(0, int(len(training)*(100-perc)/100))
    val_idx = np.arange(len(len(training)*(100-perc)/100+1), len(training))
    return training_idx, val_idx
    

'''
Buid the model 
'''
def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred)))
    

def build_model(timesteps, features, n_neurons=64, activation='tanh', dropout_rate=0.4):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(int(n_neurons*2), return_sequences=True, input_shape=(timesteps,features)))
    
    model.add(tf.keras.layers.LSTM(int(n_neurons*4), return_sequences=True, dropout=dropout_rate))
    
    model.add(tf.keras.layers.LSTM(int(n_neurons*8), return_sequences=True, dropout=dropout_rate))
    
    model.add(tf.keras.layers.LSTM(int(n_neurons*4), return_sequences=False, dropout=dropout_rate))
    
    model.add(tf.keras.layers.Dense(int(n_neurons*2), activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(int(n_neurons*4), activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(int(n_neurons*4), activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(int(n_neurons*2), activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(7, activation='linear'))
    
    model.compile(
            loss=rmse,
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['mae',rmse]
        )
    
    print(model.summary())
    
    tf.keras.utils.plot_model(model,'TimeSeriesModel.png',show_shapes=True)
    
    return model

'''
Forecast
'''
def forecast(model, df, timesteps, multisteps, scaler):
    input_seq = df[-timesteps:].values
    inp = input_seq
    predictions = list()
    for step in range(1,multisteps+1):
        inp = inp.reshape(1,timesteps,1)
        yhat = model.predict(inp, verbose=verbose)
        yhat_inversed = scaler.inverse_transform(yhat)
        predictions.append(yhat_inversed[0][0])
        inp = np.append(inp[0],yhat)
        inp = inp[-timesteps:]
        
    return predictions


'''
Plot datset
'''
def plot_data(data, title):
    data.plot(figsize=(30,20), linewidth=5, fontsize=30)
    plt.title(title, fontsize=30)
    plt.ylabel('Cases', fontsize=30)
    plt.xlabel('Days', fontsize=30)
    plt.show()
    
def plot_forecast(data, predictions, title):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(data)), data, color='green', label='Confirmed')
    plt.plot(range(len(data)-1, len(data)+len(predictions)-1), predictions, color='red', label='Prediction')
    plt.title(title)
    plt.ylabel('Cases')
    plt.xlabel('Days')
    plt.legend()
    plt.show()


'''
Vars
'''

timesteps = 3
univariate = 1
multisteps = 1
cv_splits = 3
epochs = 50
batch_size = 64
verbose = 1


# Files
file = '../datasets/timeseries_final.csv'

# Save data read
df = load_data(file)
df.index.name = 'date'

# Drop columns 'Day of month' & 'Month (number)'
drop_columns = ['Day of month', 'Month (number)']
df_aux = df.drop(columns=drop_columns, inplace=False)

test = df_aux.loc[:'16/4']

# Prepare Data for Normalization
#total_confirm = prepare_data(data_confirmed)
#total_deaths = prepare_data(data_deaths)
#total_recovered = prepare_data(data_recovered)


# Plot some data
#plot_data(total_confirm, 'Confirmed Cases - Covid-19')
#plot_data(total_deaths, 'Deaths Cases - Covid-19')
#plot_data(total_recovered, 'Recovered Cases - Covid-19')


# Copy data
#total_confirm_aux = total_confirm.copy()

# Normalization
#normalized_confirm, scaler_confirm = normalize_data(total_confirm_aux)
#normalized_deaths, scaler_deaths = normalize_data(total_deaths)
#normalized_recovered, scaler_recovered = normalize_data(total_recovered)


# Testing
#X_confirm, y_confirm = to_supervised(normalized_confirm, timesteps)
#X_deaths, y_deaths = to_supervised(normalized_deaths, timesteps)
#X_recovered, y_recovered = to_supervised(normalized_deaths, timesteps)


'''
Experiment the model
'''
#lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=50, min_lr=0.00005)

'''
Fit the best model with all data and then predict 7 days
'''
#model_confirm = build_model(timesteps, univariate, 'confirm')


#model_confirm.fit(X_confirm, y_confirm, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=verbose, callbacks=[lr])
