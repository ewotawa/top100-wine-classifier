import os
from flask import Flask, render_template, url_for, flash, redirect, request
# ML Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# text vectorization workaround
import re
import string

app = Flask(__name__)

##################
### LOAD MODEL ###
##################

# text vectorization workaround
@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    output = tf.strings.regex_replace(lowercase,
                                      '[%s]' % re.escape(string.punctuation),
                                      '')
    return output

# import the model
basedir = os.path.abspath(os.path.dirname(__file__))
wine_classifier = tf.keras.models.load_model(basedir + '/wine_classifier')

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

    # run the model to return the results. Results returned as tuple embeeded in a list.
    class_outer_list = wine_classifier.predict(wr_arr)

    # extract tuple from inside list
    class_tuple = class_outer_list[0]

    # convert tuple to list.
    class_list = list(class_tuple)

    # find the index of the max value in the classifier array.
    max_result = np.amax(class_list)
    max_index = class_list.index(max_result)
    max_style = class_descriptions[max_index]

    # create version of the classifier array to store
    classifier_dict = {
        "Dessert & Fortified": class_list[0],
        "Red": class_list[1],
        "Rosé | Rosado": class_list[2],
        "Sparkling": class_list[3],
        "White": class_list[4]
    }

    return render_template('result.html', winelabel = winelabel, winereview = winereview, winestyle = winestyle, style_input = style_input,
                                          max_style = max_style, classifier_dict = classifier_dict)
