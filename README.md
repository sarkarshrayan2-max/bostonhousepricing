# bostonhousepricing

1.[Github Account](https://github.com/sarkarshrayan2-max/bostonhousepricing)

2.[Render](https://bostonhousepricing-24n2.onrender.com)

3.[GitCLI]

4.[VsCode]

5.[Postman]

6.[Docker]

'''
🏠 Boston House Price Prediction

This project is an end-to-end Machine Learning application that predicts Boston house prices based on various housing features. The model is trained using the ElasticNet Regression algorithm and deployed as a Flask web application.

🚀 Features
Predict house prices using 13 housing attributes
REST API support for programmatic predictions
User-friendly web interface built with Flask
Model saved and loaded using Joblib
Containerized using Docker
Automated deployment with GitHub Actions
Hosted on Render

🛠️ Tech Stack
Python
Pandas & NumPy
Scikit-Learn
Flask
Joblib
Docker
GitHub Actions
Render
📊 Model

The model is trained using ElasticNet Regression, which combines both L1 (Lasso) and L2 (Ridge) regularization to improve prediction performance and reduce overfitting.

📁 Project Workflow
Data Preprocessing
        ↓
Model Training (ElasticNet)
        ↓
Model Serialization (Joblib)
        ↓
Flask Web Application
        ↓
Docker Containerization
        ↓
GitHub Repository
        ↓
GitHub Actions CI/CD
        ↓
Render Deployment
🌐 API Endpoint
Predict House Price

POST /predict_api

Example JSON:

{
  "CRIM": 0.1,
  "ZN": 18,
  "INDUS": 2.3,
  "CHAS": 0,
  "NOX": 0.5,
  "RM": 6.5,
  "AGE": 65,
  "DIS": 4.0,
  "RAD": 1,
  "TAX": 300,
  "PTRATIO": 15.3,
  "B": 390,
  "LSTAT": 5.0
}

Response:

{
  "prediction": 28.88
}
🐳 Running with Docker

Build Image:

docker build -t bostonhousepricing .

Run Container:

docker run -p 5000:10000 -e PORT=10000 bostonhousepricing

Open:

http://localhost:5000
'''

Create a new enviornment.....

'''
conda create -p venv python==3.12.4 -y

To activate Enviornment use "venv\Scripts\activate"

Built image:docker build -t bostonhousepricing .

Ran container:

docker run -p 5000:10000 -e PORT=10000 bostonhousepricing

Successfully started:

Gunicorn
Listening at 0.0.0.0:10000
'''


