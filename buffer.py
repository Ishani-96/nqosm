from fastapi import FastAPI
import requests
from pydantic import BaseModel
from models import Value
import uvicorn

app = FastAPI()

buffer = []
window_size = 100  

def to_prediction(buffer):
    load = {
        "values": buffer
        }

    response = requests.post("http://127.0.0.1:8000/predict", json=load)
    response.raise_for_status()
    return response.json()

def receive_data(value:Value):

    global buffer
    buffer.append(value.model_dump())


    if len(buffer) >= window_size:
        
        response = to_prediction(buffer)
        return response


if __name__ == "__main__":
    uvicorn.run(app, port=8001)