from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
import json
import joblib
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from utils.utils import load_json_file, preprocess_responses

app = FastAPI()

# Завантаження файлів
questions = load_json_file("questions.json")
variables_encoding = load_json_file("variables_encoding.json")

# Завантаження моделі та скейлера
model = joblib.load("logreg_model.pkl")
scaler = joblib.load("scaler.pkl")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "questions": questions})

@app.post("/calculate_risk", response_class=HTMLResponse)
async def calculate_risk(request: Request):
    form_data = await request.form()
    responses = {key: form_data[key] for key in form_data}

    coded_responses = preprocess_responses(responses, variables_encoding)
    scaled_responses = scaler.transform([coded_responses])

    risk_probability = model.predict_proba(scaled_responses)[0][1]
    risk_percentage = int(risk_probability * 100)

    return templates.TemplateResponse("result.html", {"request": request, "risk_percentage": risk_percentage})

app.mount("/static", StaticFiles(directory="static"), name="static")
