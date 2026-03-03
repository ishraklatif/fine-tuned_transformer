"""
FastAPI backend — Question Classification API
Loads the trained PrefixTuning BERT model and exposes a /predict endpoint.
"""
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.utils import seed_all, device, LABEL_CLASSES
from src.data_manager import DataManager
from src.models import PrefixTuningForClassification

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/q4_prefix_tuning_best.pt")
MODEL_NAME = "bert-base-uncased"
PREFIX_LENGTH = 5
MAX_LENGTH = 36

seed_all(1234)

# ── Load data manager (needed to get num_classes) ─────────────────────────────
print("Loading DataManager …")
DataManager.maybe_download("data", DataManager.DATA_FILE, DataManager.DATA_URL, verbose=True)
dm = DataManager(verbose=False)
dm.read_data("data/", [DataManager.DATA_FILE])
dm.manipulate_data()

# ── Load tokeniser ────────────────────────────────────────────────────────────
print("Loading tokeniser …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} …")
model = PrefixTuningForClassification(
    model_name=MODEL_NAME,
    prefix_length=PREFIX_LENGTH,
    data_manager=dm,
).to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model weights not found at '{MODEL_PATH}'. "
        "Copy your trained .pt file to backend/models/q4_prefix_tuning_best.pt"
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Question Classifier API",
    description="Classifies questions into ABBR, DESC, ENTY, HUM, LOC, NUM using BERT prefix tuning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    question: str


class PredictResponse(BaseModel):
    question: str
    predicted_class: str
    confidence: float
    all_scores: dict[str, float]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Question Classifier API is running. POST to /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.get("/classes")
def get_classes():
    return {"classes": dm.str_classes.tolist()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    encoding = tokenizer(
        req.question,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()

    class_names = dm.str_classes.tolist()
    predicted_idx = int(torch.argmax(torch.tensor(probs)))

    return PredictResponse(
        question=req.question,
        predicted_class=class_names[predicted_idx],
        confidence=round(probs[predicted_idx], 4),
        all_scores={cls: round(p, 4) for cls, p in zip(class_names, probs)},
    )
