from django.shortcuts import render
from django.http import HttpResponse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib


def home(request):
    return render(request, 'Home.html')


def donate(request):
    return render(request, 'Donate.html')


def Hospitals(request):
    return render(request, 'BestHospitals.html')


def Check(request):
    return render(request, 'PredictionPage.html')


def Prediction(request):
    
    df = pd.read_csv('Heart_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:]
    pred = {}

    if request.method == 'POST':
        age = float(request.POST.get('age',30))
        sex = float(request.POST.get('sex',0))
        cp = float(request.POST.get('cp',50))
        trestbps = float(request.POST.get('trestbps',50))
        chol = float(request.POST.get('chol',50))
        fbs = float(request.POST.get('fbs',50))
        restecg = float(request.POST.get('restecg',50))
        thalach = float(request.POST.get('thalach',50))
        exang = float(request.POST.get('exang',50))
        oldpeak = float(request.POST.get('oldpeak',50))
        slope = float(request.POST.get('slope',50))
        ca = float(request.POST.get('ca',50))
        thal = float(request.POST.get('thal',50))

        user_data = np.array(
            (age,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalach,
             exang,
             oldpeak,
             slope,
             ca,
             thal)
        ).reshape(1, 13)

        rf = RandomForestClassifier(
            n_estimators=16,
            criterion='entropy',
            max_depth=9
        )

        rf.fit(np.nan_to_num(X), Y)
        rf.score(np.nan_to_num(X), Y)
        predictions = rf.predict(user_data)

        if predictions[0] == 0:
            pred = {'Title':'Healthy','Desc': "You don't have any Heart-related Problems.","Pred":user_data}
        elif predictions[0] == 1:
            pred = {'Title':'Not-Healthy','Desc': "High chances of you, might having Heart-Related Diseases .","Pred":user_data}
    return render(request, 'PredictionOut.html', pred)


