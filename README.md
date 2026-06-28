# Rock Paper Scissors Classifier

A modern Streamlit web application for classifying Rock-Paper-Scissors images using a trained PyTorch model.

## Features

- Upload JPG, JPEG, or PNG images
- Preview the uploaded image
- Run inference with a pretrained PyTorch model
- Display predicted class and confidence percentage
- Responsive and modern UI
- Ready for deployment on Streamlit Community Cloud

## Project Structure

```text
.
├── app.py
├── model/
│   └── model.pth
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

.venv\Scripts\activate
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. Open Streamlit Community Cloud.
3. Click New app.
4. Select the repository and branch.
5. Set the main file path to `app.py`.
6. Deploy.

## Notes

- The app loads the trained model from `model/model.pth` only once using Streamlit caching.
- Invalid or corrupted images show a user-friendly error message.
