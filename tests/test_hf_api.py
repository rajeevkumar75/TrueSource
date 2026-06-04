import os
import requests
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")

headers = {"Authorization": f"Bearer {token}"}

# Test text model
print("Testing Text Model API...")
response = requests.post(
    "https://api-inference.huggingface.co/models/rajeevkumar75/truesource-text-model",
    headers=headers,
    json={"inputs": "This is a test of the emergency broadcast system."}
)
print("Text Model Response:", response.json())

# Test image model
print("\nTesting Image Model API...")
with open("C:/Users/Rajeev kumar/Desktop/TrueSource 1/frontend/img/favicon.png", "rb") as f:
    img_data = f.read()
    
response = requests.post(
    "https://api-inference.huggingface.co/models/rajeevkumar75/truesource-image-detector",
    headers=headers,
    data=img_data
)
print("Image Model Response:", response.json())
