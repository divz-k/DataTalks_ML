#must have scikit-learn and fastapi installed
# how to run: uvicorn app:app --reload

# how to test: 
#   curl -X POST "http://127.0.0.1:8000/predict" \
#    -H "Content-Type: application/json" \
#    -d '{
#        "lead_source": "Facebook",
#        "number_of_courses_viewed": 4,
#        "annual_income": 50000
#    }'

# how to test alternative:
#must have requests installed
# url = "http://127.0.0.1:8000/predict"
#client = {
#    "lead_source": "organic_search",
#    "number_of_courses_viewed": 4,
#    "annual_income": 80304.0
#}
# requests.post(url, json=client).json()

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# ----- Load model -----
with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

# ----- Create API -----
app = FastAPI(title="Model API")

# Input schema
class LeadData(BaseModel):
    lead_source: str
    number_of_courses_viewed: float
    annual_income: float

@app.get("/")
def root():
    return {"message": "Model API is running!"}

@app.post("/predict")
def predict(data: LeadData):
    # Convert to list-of-dict format expected by DictVectorizer
    record = data.dict()
    preds = model.predict_proba([record])[:, 1]
    return {"prediction probability": float(preds[0])}