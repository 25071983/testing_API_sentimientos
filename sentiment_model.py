import joblib

class SentimentModel:
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text: str) -> int:
        """
        text: string
        return: predicción (-1, 0, 1)
        """
        X_vec = self.vectorizer.transform([text])
        return int(self.model.predict(X_vec)[0])

    def predict_proba(self, text: str) -> dict:
        """
        text: string
        return: probabilidades por clase como dict
        """
        X_vec = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X_vec)[0]

        return {
            int(cls): float(p)
            for cls, p in zip(self.model.classes_, proba)
        }



