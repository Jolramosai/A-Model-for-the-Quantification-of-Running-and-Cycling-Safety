#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:52:05 2020

@author: rafaelsilva
"""

import tensorflow as tf
import numpy as np
import flask
import pandas as pd
from flask import Flask, request
from tensorflow.python.keras.models import load_model
from sklearn.externals import joblib


tf.random.set_seed(91195003)
np.random.seed(91195003)
#for an easy reset backend session state
tf.keras.backend.clear_session()


app = Flask(__name__)

# Load model
model = load_model('model_mqrcs.h5')

# Load scaler
scaler = joblib.load('minMaxScaler.pkl')

# Load data
file_path = 'dataset_labelled.csv'
df = pd.read_csv(file_path, encoding='utf-8', index_col='Index')

# Drop column - criteria_final
features = df.drop(columns=['criteria_final'])

# Normalize data
def normalize_data(df):
    normalized_df = scaler.fit_transform(df.values)
    new_df = pd.DataFrame(normalized_df, columns=df.columns, index=df.index)
    return new_df

# Data normalized
features_normalized = normalize_data(features)

# request model prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        h = request.args['h']
        d = request.args['d']
        m = request.args['m']
    elif request.method == 'POST':
        h = request.args['h']
        d = request.args['d']
        m = request.args['m']

    value = h + '/' + d + '/' + m
    features_value = features_normalized.loc[features_normalized.index == value]

    if features_value.empty:
        data = {'criteria': '-1'}
        return flask.jsonify(data)
    else:
        # Predictions
        predictions = model.predict(features_value.values, verbose=1)
        # Converting predictions to label
        result = np.argmax(predictions)

        # Senda data
        data = {'criteria': str(result)}
        return flask.jsonify(data)

# start Flask server
if __name__ == "__main__":
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)
