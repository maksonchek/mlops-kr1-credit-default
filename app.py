from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import ConfigDict, Field, FiniteFloat, StrictInt, create_model
ROOT = Path(__file__).resolve().parent
context = json.loads((ROOT / "model_context.json").read_text(encoding="utf-8"))
model = joblib.load(ROOT / "model.joblib")
fields = {}
for name in context["features"]:
    if name in context["categorical"]:
        fields[name] = (StrictInt, Field(..., ge=1 if name == "SEX" else 0, le={"SEX": 2, "EDUCATION": 6, "MARRIAGE": 3}[name]))
    elif name.startswith("PAY_") and not name.startswith("PAY_AMT"):
        fields[name] = (StrictInt, Field(..., ge=-2, le=9))
    elif name == "AGE":
        fields[name] = (StrictInt, Field(..., ge=18, le=120))
    elif name == "LIMIT_BAL" or name.startswith("PAY_AMT"):
        fields[name] = (FiniteFloat, Field(..., ge=0))
    else:
        fields[name] = (FiniteFloat, ...)
CreditProfile = create_model("CreditProfile", __config__=ConfigDict(extra="forbid"), **fields)
app = FastAPI(title="Credit default model", version="1.0", description="Учебная модель вероятности дефолта по истории карточных платежей.")
@app.get("/health")
def health():
    return {"status": "ok", "model": context["model"]}
@app.post("/predict")
def predict(profile: CreditProfile):
    frame = pd.DataFrame([profile.model_dump()], columns=context["features"])
    frame[context["categorical"]] = frame[context["categorical"]].astype(int)
    probability = float(model.predict_proba(frame)[0, 1])
    return {"default_probability": probability, "prediction": int(probability >= context["threshold"]), "threshold": context["threshold"]}
