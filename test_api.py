import requests
import json
import random

def generate_test_sample():
    """Generate a random test sample matching the dataset format"""
    return {f"A{i}": random.random() for i in range(180)}

def test_api():
    url = "http://localhost:8000/predict"
    
    # test data - using a random sample
    test_data = {
        "data": generate_test_sample(),
        "metadata": {
            "sample_id": "test_001",
            "description": "Test tabular data"
        }
    }
    
    try:
        # make the request
        response = requests.post(url, json=test_data)
        response.raise_for_status()
        
        # print results
        print("Test successful!")
        print("Status Code:", response.status_code)
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print("Test failed!")
        print("Error:", e)

if __name__ == "__main__":
    test_api()