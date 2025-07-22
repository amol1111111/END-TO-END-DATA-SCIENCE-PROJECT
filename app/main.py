from datetime import datetime
import os
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.model import load_model

app = FastAPI()

# Load model
model = load_model()

# Template folder is outside 'app' → use root-level templates
templates = Jinja2Templates(directory="templates")

# Temporary global store for form data
last_prediction = {}

# Home route → form page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    fields = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]
    return templates.TemplateResponse("form.html", {
        "request": request,
        "fields": fields
    })

# Handle form submission
@app.post("/predict-form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    Pregnancies: int = Form(...),
    Glucose: int = Form(...),
    BloodPressure: int = Form(...),
    SkinThickness: int = Form(...),
    Insulin: int = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...)
):
    input_data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }

    # Predict
    prediction = model.predict([[*input_data.values()]])[0]
    result = 1 if prediction == 1 else 0

    # Add current Indian datetime
    india_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    record = input_data.copy()
    record["Prediction"] = result
    record["Timestamp"] = india_time

    df = pd.DataFrame([record])
    file_path = "C:/Users/kadam/OneDrive/Documents/internship_IV/Diabetes_Project_1/user_data.csv"
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)

    # Store temporarily
    global last_prediction
    last_prediction = {
        "input_data": input_data,
        "prediction": result
    }

    return RedirectResponse("/result", status_code=303)

# Show result page
@app.get("/result", response_class=HTMLResponse)
def show_result(request: Request):
    return templates.TemplateResponse("result.html", {
        "request": request,
        "input_data": last_prediction.get("input_data", {}),
        "prediction": last_prediction.get("prediction", "No prediction")
    })
