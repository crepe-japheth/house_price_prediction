from django.shortcuts import render
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def index(request):
    if request.method == 'POST':
        area = float(request.POST.get('area'))
        bedrooms = float(request.POST.get('bedrooms'))
        bathrooms = float(request.POST.get('bathrooms'))
        stories = float(request.POST.get('stories'))
        mainroad = float(request.POST.get('mainroad'))
        guestroom = float(request.POST.get('guestroom'))
        basement = float(request.POST.get('basement'))
        hotwaterheating = float(request.POST.get('hotwaterheating'))
        airconditioning = float(request.POST.get('airconditioning'))
        parking = float(request.POST.get('parking'))
        prefarea = float(request.POST.get('prefarea'))
        furnishingstatus = float(request.POST.get('furnishingstatus'))

        my_model = joblib.load('price_index_model/house_price_index_model.joblib')

        features = pd.DataFrame(np.array([area, bedrooms, bathrooms, stories, mainroad,guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]).reshape(1, -1), columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning','parking','prefarea', 'furnishingstatus'])

        prediction = my_model.predict(features)
        context = {'prediction':prediction[0]}
        return render(request, 'index.html', context)
    context = {'prediction':None}
    return render(request, 'index.html', context)

