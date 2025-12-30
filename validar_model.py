from sentiment_model import SentimentModel

BASE_PATH_OUTPUT = "../train/output/"
MODEL_PATH = BASE_PATH_OUTPUT + "modelo_final.pkl"
VECT_PATH = BASE_PATH_OUTPUT + "vectorizador_final.pkl"

model = SentimentModel(MODEL_PATH, VECT_PATH)

# Ejemplos de prueba
texts = [
    "El servicio fue excelente y muy rápido",
    "No me gustó para nada la atención",
    "El producto llegó, sin más comentarios"
]


for text in texts:
    pred = model.predict(text)
    proba = model.predict_proba(text)

    print(f"Texto: {text}")
    print(f"Predicción: {pred}")
    print(f"Probabilidades: {proba}")
    print("-" * 60)
