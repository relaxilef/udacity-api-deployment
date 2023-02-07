
"""
Send a POST request to the API for testing purposes.
"""
import requests

if __name__ == "__main__":
    content = {
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

    
    URL = "http://127.0.0.1:8000/predict"
    request = requests.post(
        url=URL,
        json=content,
        timeout=600,
    )

    status_code = request.status_code

    inference = request.json()["result"]
    print(inference)
