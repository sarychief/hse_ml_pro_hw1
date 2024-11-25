from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
from io import StringIO

app = FastAPI()

with open("model.pkl", "rb") as f: # моделька
    model = pickle.load(f)

class Item(BaseModel): # описание входных данных для одного объекта
    year: int
    km_driven: int
    mileage: float
    engine: int
    max_power: float
    torque: float
    seats: int
    max_torque_rpm: float


class Items(BaseModel): # Описание коллекции объектов
    objects: List[Item]


@app.post("/predict_item") # предсказание одного семпла 
def predict_item(item: Item) -> float:
    try:
        data = pd.DataFrame([item.dict()])
        prediction = model.predict(data)[0]
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {e}")


@app.post("/predict_items") # предсказание датасета
async def predict_items(file: UploadFile) -> str:
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        predictions = model.predict(df)
        df["predicted_price"] = predictions
        output = StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки файла: {e}")
