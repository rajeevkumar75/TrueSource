import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv

load_dotenv()
print("HF_TOKEN loaded:", bool(os.getenv("HF_TOKEN")))

from app.inference_service import warmup_models
print("Starting warmup...")
loaded = warmup_models()
print("Warmup finished. Loaded:", loaded)
