from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('fish_weight_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction_result="", length1="", length2="", length3="", height="", width="")

@app.route('/predict', methods=['POST'])
def predict():
    # Convert input values to floats
    length1 = float(request.form['Length1'])
    length2 = float(request.form['Length2'])
    length3 = float(request.form['Length3'])
    height = float(request.form['Height'])
    width = float(request.form['Width'])

    # Make prediction
    prediction = model.predict([[length1, length2, length3, height, width]])

    return render_template('index.html', prediction_result=f'Predicted weight: {prediction[0]:.2f} grams', length1= length1, length2= length2, length3= length3, height= height, width= width )

if __name__ == '__main__':
    app.run(debug=True)
