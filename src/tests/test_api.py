import pytest
import requests
import json
import subprocess
import time
import aiohttp
import httpx

from fastapi.testclient import TestClient

from main import app


CLIENT = TestClient(app)


def test_greet():
    response = CLIENT.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, world!"}


@pytest.mark.asyncio
async def test_predict():
    response = CLIENT.post(
        url="/predict",
        json={
            "age": 35,
            "workclass": "Federal-gov",
            "fnlgt": 39207,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        }
    )

    assert response.status_code == 200
    assert "result" in response.json()


def test_post_data_fail():
    data = {"age": -5}
    response = CLIENT.post("/predict/", data=json.dumps(data))
    assert response.status_code == 422
