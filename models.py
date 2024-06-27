from pydantic import BaseModel, Field
from typing import List

class Value(BaseModel):
    timestamp: str
    value: float 

class PredictionResponse(BaseModel):
    timestamp: str
    current_value: float
    threshold: float
    next_value_over_threshold: bool
    prob: float
    prob_threshold: float
    window_size: int
    distribution_type: str

class Data(BaseModel):
    values: List[Value]
    threshold: float
    window_size: int
    prob_threshold: float 