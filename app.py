from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 
import joblib
app=Flask(__name__)


model = joblib.load('elasticnet_model1.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()

    feature_order = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
        'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        'PTRATIO', 'B', 'LSTAT'
    ]

    input_data = pd.DataFrame([data], columns=feature_order)

    print(input_data)

    prediction = model.predict(input_data)

    return jsonify({
        "prediction": float(prediction[0])
    })

if __name__=="__main__":
    app.run(debug=True)
