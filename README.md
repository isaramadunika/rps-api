# Rock Paper Scissors API

A production-ready FastAPI service for classifying Rock-Paper-Scissors images using a PyTorch model.

## Features

- FastAPI REST API with health and prediction endpoints
- Startup-time model loading for efficient inference
- Automatic CPU/GPU detection
- Image preprocessing with torchvision transforms
- Pydantic response models
- CORS enabled and logging configured
- Docker support for containerized deployment
- Render deployment configuration

## Folder Structure

```text
rps-api/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   └── predict.py
│   ├── services/
│   │   └── predictor.py
│   ├── models/
│   │   ├── cnn_model.py
│   │   └── model_loader.py
│   ├── utils/
│   │   └── preprocessing.py
│   └── schemas/
│       └── response.py
├── model/
│   └── model.pth
├── requirements.txt
├── Dockerfile
├── render.yaml
├── README.md
└── .gitignore
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open:

- http://localhost:8000/
- http://localhost:8000/health
- http://localhost:8000/docs

## Deploy to Render

1. Create a new Web Service on Render.
2. Connect this repository.
3. Render will use the provided render.yaml configuration.
4. Deploy the service.

## Swagger Documentation

Once the server is running, visit:

- http://localhost:8000/docs
- http://localhost:8000/redoc

## Example Requests

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

### Python

```python
import requests

url = "http://localhost:8000/predict"
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    print(response.json())
```

### JavaScript

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("http://localhost:8000/predict", {
  method: "POST",
  body: formData,
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

## Git Commands

```bash
git init
git add .
git commit -m "Initial Commit"
git branch -M main
git remote add origin https://github.com/isaramadunika/rps-api.git
git push -u origin main
```
