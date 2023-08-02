#import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, Response, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Depends, FastAPI, HTTPException, status
from titanic_model.predict import make_prediction

import uvicorn
import datetime

import logging

app = FastAPI(
    title="Titanic Survial Prediction API",
    description="Suma",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def hello():
    return JSONResponse(status_code=200, content={'response': 'Hello'})

@app.post("/predict")
async def predict_survival(request: Request):
    try:
        data_in = await request.json()
        resp = make_prediction(input_data=data_in)
        survived = resp["predictions"].tolist()
        return JSONResponse(status_code=200, content={'survided': survived[0], "version": resp["version"]})
    except ValueError:
        return JSONResponse(status_code=400, content={'response': 'Invalid input'})

if __name__=="__main__":
    logging.basicConfig(filename='application.log', level=logging.INFO)

    uvicorn.run(app, host="0.0.0.0", port=5003)

