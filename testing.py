from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

data_path = os.path.join(os.path.dirname(__file__), "Training.csv")
data = pd.read_csv(data_path)
cols = data.columns[:-1]
x = data[cols]
y = data['prognosis']

print(y.unique())
dt = DecisionTreeClassifier(min_samples_split=20)
dt.fit(x, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/explore')
def explore():
    return render_template('explore.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/doctorlogin')
def doctorlogin():
    return render_template('doctorlogin.html')

@app.route('/adminlogin')
def adminlogin():
    return render_template('adminlogin.html')

@app.route('/patientlogin')
def patientlogin():
    return render_template('patientlogin.html')

@app.route('/adminprofile')
def adminDashboard():
    return render_template('admindashboard.html')

@app.route('/doctorprofile')
def doctorDashboard():
    return render_template('doctorprofile.html')

@app.route('/patientprofile')
def patientDashboard():
    return render_template('patientprofile.html')

@app.route('/patientownprofile')
def patientOwnProfile():
    return render_template('patientownprofile.html')


@app.route('/checkDisease')
def checkDisease():
    return render_template('searchDisease.html',symptoms=cols)

@app.route('/contactinfo')
def viewContactInfo():
    return render_template('viewcontactinfo.html')

@app.route('/feedback')
def feedBack():
    return render_template('feedback.html')

@app.route('/feedbackinfo')
def viewFeedBackInfo():
    return render_template('viewfeedbackinfo.html')

@app.route('/viewpatients')
def viewPatients():
    return render_template('viewpatients.html')

# Doctor Dashboard
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']

#         # Add login validation logic here (mock logic for now)
#         if username == "doctor" and password == "doctor123":
#             return redirect(url_for('doctor_dashboard'))
#         elif username == "patient" and password == "patient123":
#             return redirect(url_for('search'))
#         else:
#             return render_template('login/login.html', error="Invalid username or password")
#     return render_template('login/login.html')

# # Route for Signup Page
# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         # Handle signup logic here
#         username = request.form['username']
#         password = request.form['password']
#         role = request.form['role']  # Doctor or Patient
#         return redirect(url_for('login'))
#     return render_template('signup/signup.html')

# Symptom Checker (Patient's Dashboard)
# @app.route('/search')
# def search():
#     return render_template('search.html', symptoms=cols)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    if not selected_symptoms:
        return render_template('searchDisease.html', symptoms=cols, prediction_text="Please select at least one symptom.")

    input_data = [0] * len(cols)
    for symptom in selected_symptoms:
        if symptom in cols:
            input_data[cols.tolist().index(symptom)] = 1 

    input_data = np.array([input_data])
    prediction = dt.predict(input_data)[0]

    # Map diseases to doctor specializations
    disease_to_specialist = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist/Immunologist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist",
    "Drug Reaction": "Allergist/Immunologist",
    "Peptic ulcer disease": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Diabetes": "Endocrinologist",
    "Gastroenteritis": "Gastroenterologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedist",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Hepatologist",
    "Malaria": "Infectious Disease Specialist",
    "Chicken pox": "Infectious Disease Specialist",
    "Dengue": "Infectious Disease Specialist",
    "Typhoid": "Infectious Disease Specialist",
    "hepatitis A": "Hepatologist",
    "Hepatitis B": "Hepatologist",
    "Hepatitis C": "Hepatologist",
    "Hepatitis D": "Hepatologist",
    "Hepatitis E": "Hepatologist",
    "Alcoholic hepatitis": "Hepatologist",
    "Tuberculosis": "Pulmonologist",
    "Common Cold": "General Physician",
    "Pneumonia": "Pulmonologist",
    "Dimorphic hemmorhoids(piles)": "Proctologist",
    "Heart attack": "Cardiologist",
    "Varicose veins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Osteoarthritis": "Orthopedist",
    "Arthritis": "Rheumatologist",
    "(vertigo) Paroymsal Positional Vertigo": "Neurologist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist"
    }

    # Suggest a specialist for the predicted disease
    specialist = disease_to_specialist.get(prediction, "General Physician")

    return render_template(
        'searchDisease.html',
        symptoms=cols,
        prediction_text=f'Disease predicted by AI: {prediction}',
        specialist_text=f' {specialist}.',
        selected_symptoms=selected_symptoms
    )


if __name__ == '__main__':
    app.run(debug=True)

