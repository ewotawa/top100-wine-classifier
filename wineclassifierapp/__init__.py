import os
from flask import Flask, render_template, url_for, flash, redirect, request
# ML Imports
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import pickle

app = Flask(__name__)

##################
### LOAD MODEL ###
##################

filename = 'wineclassifierapp/wine_classifier/SVMClassifier.pkl'
with open(filename, 'rb') as file:
    wine_classifier = pickle.load(file)

#################
### VIEWS ###
#################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():

    class_descriptions = ['Dessert & Fortified', 'Red', 'Rosé | Rosado', 'Sparkling', 'White']

    # collect form details
    winelabel = request.args.get('winelabel')
    winereview = request.args.get('winereview')
    winestyle = request.args.get('winestyle')

    # convert winestyle to descriptive value for output
    style_input_list = ['0_dessertfortified', '1_red', '2_roserosado', '3_sparkling', '4_white']
    style_input_index = style_input_list.index(winestyle)
    style_input = class_descriptions[style_input_index]

    # append wine review to an array for processing in the model.
    wr_arr = []
    wr_arr.append(winereview)

    # run the model to return the results. Results returned as an array.
    prediction = wine_classifier.predict(wr_arr)
    pred_int = prediction[0]

    # find the index of the max value in the classifier array.
    max_style = class_descriptions[pred_int]

    return render_template('result.html', winelabel = winelabel, winereview = winereview, winestyle = winestyle, style_input = style_input,
                                          max_style = max_style)
