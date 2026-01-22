
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import os

# Versión del modelo
MODEL_VERSION = "v3.0.1"

# Configuración dinámica del modelo
MODEL_NAME = os.getenv("MODEL_NAME", "R5K_v3.pkl")
MODEL_PATH = os.path.join("models", MODEL_NAME)

# Constantes para la lógica de Neutro (del sentiment_wrapper)
LOW_THRESHOLD = 0.4
HIGH_THRESHOLD = 0.6

# Variable global para el pipeline unificado (Vectorizador + Modelo)
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global pipeline
    if not os.path.exists(MODEL_PATH):
        # Fallback para desarrollo local si no está en la carpeta /models
        local_path = MODEL_NAME 
        if not os.path.exists(local_path):
            raise RuntimeError(f"Model file {MODEL_NAME} not found in /models or root!")
        path_to_load = local_path
    else:
        path_to_load = MODEL_PATH
    
    pipeline = joblib.load(path_to_load)
    print(f"Pipeline {MODEL_NAME} loaded successfully. Version: {MODEL_VERSION}")
    yield
    # Shutdown (si es necesario)

app = FastAPI(title="Sentiment API", version=MODEL_VERSION, lifespan=lifespan)

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    prevision: str
    probabilidad: float
    model_version: str

class BatchSentimentRequest(BaseModel):
    texts: list[str]

class BatchSentimentResponse(BaseModel):
    results: list[SentimentResponse]

def get_sentiment_label(prob_pos: float):
    """Implementa la lógica del sentiment_wrapper.py"""
    if LOW_THRESHOLD < prob_pos < HIGH_THRESHOLD:
        return "Neutro"
    return "Positivo" if prob_pos >= HIGH_THRESHOLD else "Negativo"

def calculate_confidence(prob_pos: float):
    prob_neg = 1.0 - prob_pos
    return round(max(prob_pos, prob_neg), 4)

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_sentiment_batch(request: BatchSentimentRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="List of texts cannot be empty")
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        # Inferencia por lotes
        probs = pipeline.predict_proba(request.texts)
        
        results = []
        for i, prob in enumerate(probs):
            prob_pos = float(prob[1])
            prediction_label = get_sentiment_label(prob_pos)
            confidence = calculate_confidence(prob_pos)
            
            results.append(SentimentResponse(
                prevision=prediction_label,
                probabilidad=confidence,
                model_version=MODEL_VERSION
            ))

        return BatchSentimentResponse(results=results)
    except Exception as e:
        import traceback
        print(f"ERROR EN PREDICCIÓN BATCH: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        # Inferencia directa
        probs = pipeline.predict_proba([request.text])[0]
        prob_pos = float(probs[1])
        
        prediction_label = get_sentiment_label(prob_pos)
        confidence = calculate_confidence(prob_pos)

        return SentimentResponse(
            prevision=prediction_label,
            probabilidad=confidence,
            model_version=MODEL_VERSION
        )
    except Exception as e:
        import traceback
        print(f"ERROR EN PREDICCIÓN: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "model_loaded": pipeline is not None,
        "model_version": MODEL_VERSION,
        "model_file": MODEL_NAME
    }
