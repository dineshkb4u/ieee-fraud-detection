from flask import Flask, jsonify, request, render_template, session, redirect
import pandas as pd
import numpy as np
import os,gc
import xgboost as xgb
import time
import joblib
import flask
from io import StringIO
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    xg_mdl = joblib.load('xgb_fe.pkl')
    x_t = request.files['x_test']
    xt = str(x_t.read(),'utf-8')
    data = StringIO(xt)
    x_test = pd.read_csv(data)
    print(x_test.head())
    x_TrxID = x_test.pop("TransactionID")
    y_pred_test = xg_mdl.predict_proba(x_test)
    submission = {}
    submission.update(dict(zip(x_TrxID.values,y_pred_test)))
    submission = pd.DataFrame.from_dict(submission, orient="index").reset_index()
    submission.columns = ["TransactionID", "isFraud-No", "isFraud-yes"]
    print(submission.head())
    return submission.to_html(header="true", table_id="table")
    #return render_template('output.html', tables = [submission.to_html(classes='data')], header = "true") 
    #return jsonify({'prediction probabilities': submission.to_json(orient="index")})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
