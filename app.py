#app.py

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow import keras

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

model = keras.models.load_model("export_model_Vtesting.keras")
print(model.summary())

weather_code = {
    "0": "clear",
    "1": "partly cloudy",
    "2": "overcast",
    "3": "drizzle",
    "4": "rain"
}

def convert_percentage(numbers):
    total = sum(numbers)

    # Convert each number to a percentage
    percentages = [(num / total) * 100 for num in numbers]

    # Adjust the percentages to ensure their sum equals 100
    adjusted_percentages = [round(p) for p in percentages]

    # Fix any rounding errors by adjusting the largest value
    difference = 100 - sum(adjusted_percentages)
    if difference != 0:
        max_index = adjusted_percentages.index(max(adjusted_percentages))
        adjusted_percentages[max_index] += difference

    return adjusted_percentages

def predict(temp, humi, pres):
    result = model.predict(np.array([[temp, humi, pres]])).tolist()[0]
    result_max = max(result)
    result_index = result.index(result_max)
    result_text = weather_code[str(result_index)]

    return [result_index, result_text, convert_percentage(result)]

app = FastAPI()

class Argument(BaseModel):
    temp: float
    humi: float
    pres: float

@app.get("/predict")
def route_predict(arg:Argument):
    result = predict(float(arg.temp), float(arg.humi), float(arg.pres))
    return result

@app.get('/')
def route_hello_world():
    return "Hello,World"

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)