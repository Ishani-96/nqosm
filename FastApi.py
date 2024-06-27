from fastapi import FastAPI
import pandas as pd
from models import Value, PredictionResponse, Data
from prediction import predict

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(data: Data):

    #import pdb
    #pdb.set_trace()

    df =  pd.DataFrame(data.model_dump()["values"])


    threshold = data.threshold
    window_size = data.window_size
    prob_threshold = data.prob_threshold


    prob, best_dist = predict(df, 'value', threshold)

    #mport pdb
   # pdb.set_trace()


    response = PredictionResponse(
        timestamp=data.values[-1].timestamp,  
        current_value=data.values[-1].value, 
        threshold=threshold,
        next_value_over_threshold=(prob > prob_threshold),
        prob=prob,
        prob_threshold=prob_threshold,
        window_size=window_size,
        distribution_type=best_dist
    )

    return response