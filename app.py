# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 22:27:05 2022

@author: user
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('main.html', prediction_text='Average body weight of shrimp is {} gram'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
