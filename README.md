````md
# Boston House Price Prediction with MLOps

An end-to-end machine learning application that predicts Boston house prices from 13 housing features.

The project uses ElasticNet Regression and serves predictions through a Flask web application and REST API. The MLOps branch adds reproducible training with DVC, Amazon S3 artifact storage, Docker containerization, and AWS EC2 deployment.

## Links

- [GitHub Repository](https://github.com/sarkarshrayan2-max/bostonhousepricing)
- [MLOps Branch](https://github.com/sarkarshrayan2-max/bostonhousepricing/tree/feat/dvc-aws)
- [Render Deployment](https://bostonhousepricing-24n2.onrender.com)

## Features

- Predict house prices using 13 housing attributes.
- ElasticNet Regression model.
- Flask web interface.
- REST API for programmatic predictions.
- Joblib model serialization.
- Dockerized deployment with Gunicorn.
- GitHub Actions workflow for the original Render deployment.
- Reproducible training pipeline with DVC.
- Dataset and trained model versioning through Amazon S3.
- AWS EC2 deployment using Docker.
- IAM role-based S3 access on EC2 without permanent AWS keys on the server.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib
- DVC
- Amazon S3
- Docker
- Gunicorn
- AWS EC2
- AWS IAM
- GitHub Actions
- Render

## Model

The model uses ElasticNet Regression, which combines L1 and L2 regularization.

Current evaluation metrics:

| Metric | Value |
|---|---:|
| R² Score | 0.6590 |
| Mean Absolute Error | 3.1395 |
| Root Mean Squared Error | 5.0056 |

## Project Workflow

```text
Dataset
   |
   v
Data Preprocessing
   |
   v
ElasticNet Model Training
   |
   v
Model and Metrics Artifacts
   |
   +------------------------------+
   |                              |
   v                              v
Flask API                    DVC Versioning
   |                              |
   v                              v
Docker Container           Amazon S3 Remote
   |                              |
   v                              v
Render Deployment         AWS EC2 Deployment
````

## MLOps Architecture

```text
GitHub Repository
  - Application code
  - DVC pipeline definition
  - Training parameters
  - DVC lock file
          |
          v
DVC Pipeline
  Dataset -> train.py -> model.joblib + metrics.json
          |
          v
Amazon S3
  - Versioned dataset
  - Versioned model artifact
  - Versioned metrics artifact
          |
          v
AWS EC2 with IAM Role
  - Pulls DVC artifacts securely from S3
  - Builds Docker image
  - Runs Flask API container
```

## Project Structure

```text
bostonhousepricing/
│
├── .dvc/
│   └── config
│
├── .github/
│   └── workflows/
│
├── data/
│   └── raw/
│       └── HousingData.csv.dvc
│
├── src/
│   └── train.py
│
├── artifacts/
│   └── model.joblib
│
├── templates/
│   └── home.html
│
├── app.py
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── metrics.json
├── Dockerfile
├── requirements.txt
└── README.md
```

## Input Features

The model expects these 13 fields:

```text
CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
DIS, RAD, TAX, PTRATIO, B, LSTAT
```

## DVC Pipeline

The DVC training stage is:

```text
data/raw/HousingData.csv
          |
          v
      src/train.py
          |
          v
artifacts/model.joblib + metrics.json
```

Training configuration is stored in `params.yaml`:

```yaml
train:
  test_size: 0.2
  random_state: 42

model:
  alpha: 0.01
  l1_ratio: 0.2
  max_iter: 10000
```

## Local Setup

Clone the MLOps branch:

```bash
git clone --branch feat/dvc-aws https://github.com/sarkarshrayan2-max/bostonhousepricing.git
cd bostonhousepricing
```

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install "dvc[s3]"
```

## Restore DVC Artifacts

The dataset, trained model, and metrics are stored in Amazon S3 through DVC.

```bash
dvc pull
```

Verify that the pipeline is consistent:

```bash
dvc status
```

Expected output:

```text
Data and pipelines are up to date.
```

View tracked metrics:

```bash
dvc metrics show
```

## Reproduce Training

Run the training pipeline:

```bash
dvc repro
```

Push updated model artifacts to the configured DVC remote:

```bash
dvc push
```

## Run Locally

Restore the model artifact first:

```bash
dvc pull
```

Start the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Health endpoint:

```text
http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "status": "healthy",
  "model": "model.joblib"
}
```

## API Endpoint

### Predict House Price

```text
POST /predict_api
```

### Example Request

```json
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
```

### Example PowerShell Request

```powershell
$body = @{
    CRIM = 0.1
    ZN = 18
    INDUS = 2.3
    CHAS = 0
    NOX = 0.5
    RM = 6.5
    AGE = 65
    DIS = 4.0
    RAD = 1
    TAX = 300
    PTRATIO = 15.3
    B = 390
    LSTAT = 5.0
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/predict_api" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### Example Response

```json
{
  "prediction": 30.44756746926752
}
```

## Docker

Before building the image, restore the DVC model artifact:

```bash
dvc pull
```

Build the image:

```bash
docker build -t boston-mlops-api:1 .
```

Run the container locally:

```bash
docker run -d \
  --name boston-mlops-api \
  -p 5000:10000 \
  boston-mlops-api:1
```

Test it:

```bash
curl http://127.0.0.1:5000/health
```

Stop the container:

```bash
docker stop boston-mlops-api
docker rm boston-mlops-api
```

## AWS EC2 Deployment

The AWS deployment was tested with:

* Amazon Linux 2023.
* Docker and Git installed on EC2.
* Amazon S3 configured as the DVC remote.
* An EC2 IAM role with read-only access to DVC artifacts.
* AWS Systems Manager Session Manager instead of SSH keys.
* Docker mapping host port `80` to application port `10000`.

### EC2 Deployment Commands

```bash
sudo dnf install -y git docker python3.12 python3.12-pip
sudo systemctl enable --now docker

mkdir -p ~/apps
cd ~/apps

git clone --branch feat/dvc-aws https://github.com/sarkarshrayan2-max/bostonhousepricing.git
cd bostonhousepricing

python3.12 -m venv .dvc-venv
source .dvc-venv/bin/activate

pip install --upgrade pip
pip install "dvc[s3]"

dvc pull
dvc status
```

Build and run the API:

```bash
sudo docker build -t boston-mlops-api:1 .

sudo docker run -d \
  --name boston-mlops-api \
  --restart unless-stopped \
  -p 80:10000 \
  boston-mlops-api:1
```

Verify from the EC2 instance:

```bash
curl http://127.0.0.1/health
```

Public health endpoint:

```text
http://<EC2_PUBLIC_IP>/health
```

## Branch Strategy

| Branch         | Purpose                                                       |
| -------------- | ------------------------------------------------------------- |
| `main`         | Original Flask project deployed on Render                     |
| `feat/dvc-aws` | DVC, Amazon S3, Docker, IAM, and AWS EC2 MLOps implementation |

Do not merge `feat/dvc-aws` into `main` until the Render deployment is also configured to run `dvc pull` before the Flask application starts.

## Cost Management

Stop the EC2 instance when not testing:

```text
EC2 Console
-> Instances
-> Select boston-mlops-api
-> Instance state
-> Stop instance
```

Stopping the instance retains the EBS volume and project files. The public IPv4 address may change after starting the instance again.

## Future Improvements

* MLflow experiment tracking.
* Automated DVC pipeline validation in GitHub Actions.
* Automated EC2 deployment.
* Amazon ECR for Docker image storage.
* CloudWatch logging and monitoring.
* Data validation.
* Model drift monitoring.
* Elastic IP or domain name for a stable API URL.

## Author

Shrayan Sarkar

## License

This project is licensed under the Apache 2.0 License.

````

Then run:

```powershell
git add README.md
git commit -m "Update README with DVC and AWS MLOps workflow"
git push
````

[1]: https://github.com/sarkarshrayan2-max/bostonhousepricing "GitHub - sarkarshrayan2-max/bostonhousepricing · GitHub"
