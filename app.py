from flask import Flask, render_template, request
app = Flask(__name__)

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix

# loading & preprocessing
data = pd.read_csv('diabetes_data.csv')
data.drop(['SkinThickness', 'DiabetesPedigreeFunction'], axis=1, inplace=True)  
X = data.drop('Outcome', axis=1)
y = data['Outcome']
# splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# SMOTE applied for class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# training a Random_Forest model
model = RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# prediction function
def diabetes_diagnosis(patient_data):
    prediction_probabilities = model.predict_proba(patient_data)
    #  Prob of class 1 = high risk of diabetes 
    prediction_probability = prediction_probabilities[0][1]
    
    if prediction_probability >= 0.5:     
        result = f"High diabetes risk detected {prediction_probability:.1%}. Please consult a healthcare professional."
    else:
        result = f"No diabetes risk detected. Probability: {prediction_probability:.1%}"
        
    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data and create df 
        input_data = {
            'Pregnancies': [int(request.form['pregnancies'])],
            'Glucose': [int(request.form['glucose'])],
            'BloodPressure': [int(request.form['blood_pressure'])],
            'Insulin': [int(request.form['insulin'])],
            'BMI': [float(request.form['bmi'])],
            'Age': [int(request.form['age'])]
        }
        patient_data = pd.DataFrame(input_data)

        # prediction based on input using the prediction function
        result = diabetes_diagnosis(patient_data)
        return render_template('index.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
