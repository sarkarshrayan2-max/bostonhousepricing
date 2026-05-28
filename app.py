from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 
import joblib
app=Flask(__name__)
model=joblib.load(open('elasticnet_model.joblib','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    new_data = np.array(list(data.values())).reshape(1, -1)
    print(new_data)
    output = model.predict(new_data)
    return jsonify({
        'prediction': float(output[0])
    })

if __name__=="__main__":
    app.run(debug=True)
