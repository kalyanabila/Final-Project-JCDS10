from flask import Flask, render_template, request
# import jsonify
import requests
import joblib
import numpy as np
import pandas as pd
import json
import sklearn
import matplotlib
app = Flask(__name__)
model = joblib.load('bank-model-model_SVC_tuned', mmap_mode=None)


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        CreditScore = int(request.form['CreditScore'])
        Age = int(request.form['Age'])
        Tenure = int(request.form['Tenure'])
        Balance = int(request.form['Balance'])
        NumOfProducts = int(request.form['NumOfProducts'])
        HasCrCard = int(request.form['HasCrCard'])
        IsActiveMember = int(request.form['IsActiveMember'])
        EstimatedSalary = int(request.form['EstimatedSalary'])
        Geography_Germany = request.form['Geography_Germany']
        if(Geography_Germany == 'Germany'):
            Geography_Germany = 1
            Geography_Spain= 0
            Geography_France = 0
                
        elif(Geography_Germany == 'Spain'):
            Geography_Germany = 0
            Geography_Spain= 1
            Geography_France = 0
        
        else:
            Geography_Germany = 0
            Geography_Spain= 0
            Geography_France = 1

        Gender_Male = request.form['Gender_Male']
        if(Gender_Male == 'Male'):
            Gender_Male = 1
            Gender_Female = 0
        else:
            Gender_Male = 0
            Gender_Female = 1

        prediction = model.predict([[CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary, Geography_France, Geography_Germany,Geography_Spain,Gender_Female, Gender_Male]])
        if prediction==1:
             return render_template('index.html',prediction_text="Nasabah akan berpindah bank (Churn)")
        else:
             return render_template('index.html',prediction_text="Nasabah TIDAK akan berpindah bank (NOT Churn)")


@app.route('/static')
def statics():
    return render_template('visual.html')

@app.route('/dataset')
def dataset1():
    df = pd.read_csv(r'D:\Purwadhika\Module 03\Final Project\static\Churn_Modelling.csv').sample(50)
    return df.to_html()


if __name__=="__main__":
    app.run(debug=True)
