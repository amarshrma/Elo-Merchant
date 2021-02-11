from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
import Final as final


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    data = pd.read_csv('train_EDA.csv')
    cards = data['card_id']
    
    feature_1 = [1,2,3,4,5]
    feature_2 = [1,2,3]
    feature_3 = [0,1]
    return flask.render_template('index.html', cards = cards, feature_1 = feature_1, feature_2 = feature_2, feature_3 = feature_3)


@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('finalModel.pkl')
    to_predict_list = request.form.to_dict()
    card_id = request.form.getlist('card_id')
    date = pd.to_datetime(request.form['firstactivemonth'], errors = 'coerce')
    f1 = request.form.getlist('feature_1')
    f2 = request.form.getlist('feature_2')
    f3 = request.form.getlist('feature_3')
    data = {'card_id' : card_id,
            'first_active_month' : date,
            'feature_1' : f1,
            'feature_2' : f2,
            'feature_3' : f3
    }
    data = pd.DataFrame(data)
    prediction = final.final_fun_1(data)
    print("Loyalty Score", prediction)
    return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

