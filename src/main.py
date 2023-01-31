from fastapi import FastAPI

from pydantic import BaseModel, validator

import pandas as pd

import json

import uvicorn

from starter.inference import inference


app = FastAPI(
    title="Exercise API",
    description="An API used for inference on the Census dataset.",
    version="1.0.0",
)

class Data(BaseModel):
    """
    Base model for input type checking.
    """
    age: float = 35
    workclass: str = "Federal-gov"
    fnlgt: float = 39207
    education: str = "HS-grad"
    education_num: float = 9
    marital_status: str = "Never-married"
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: float = 0
    capital_loss: float = 0
    hours_per_week: float = 40
    native_country: str = "United-States"

    @validator("age")
    def check_age_positive(cls, value):
        """
        The age must be positive.
        """
        if value < 0:
            raise ValueError("Age must be positive.")
        return value


@app.get("/")
def greet() -> dict[str, str]:
    """
    Greet function.
    """
    return {"message": "Hello, world!"}


@app.post("/predict")
def predict(inference_data: Data) -> dict:
    """
    Inference function.
    """
    inference_json = json.loads(inference_data.json())
    inference_json = {k: [v] for k, v in inference_json.items()}
    inference_df = pd.DataFrame(inference_json)
    result = list(int(e) for e in inference(inference_df))
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app)
