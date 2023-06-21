from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
app=Flask(__name__)
cors=CORS(app)
try:
    with open('LinearRegressionModel.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print("Error loading the pickle file:", e)

try:
    car = pd.read_csv('New_Car_data.csv')
except Exception as e:
    print("Error loading the CSV file:", e)

@app.route('/', methods=['GET', 'POST'])
def index():
        companies = sorted(car['company'].unique())
        car_models = sorted(car['name'].unique())
        age = sorted(car['Age'].unique(), reverse=True)
        fuel_type = car['fuel_type'].unique()

        companies.insert(0, 'Select Company')
        return render_template('index1.html', companies=companies, car_models=car_models, age=age, fuel_types=fuel_type)
        

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    age = request.form.get('age')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'Age', 'kms_driven', 'fuel_type'],data=np.array([car_model, company, age, driven, fuel_type]).reshape(1,5)))
    print(prediction)
    return str(np.round(prediction[0], 2))

@app.route('/images')
def about():
    return render_template('images.html')
    return "This Is A Car Price Prediction Page."

if __name__ == '__main__':
    app.run(debug=True)
