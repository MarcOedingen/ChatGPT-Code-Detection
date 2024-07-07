# Prediction interface for Cog ⚙️
# https://cog.run/python
import dill as pickle
from cog import BasePredictor, BaseModel, Input
import tiktoken
from xgboost import XGBClassifier


class Output(BaseModel):
    prediction: int
    probability: float # Probability of the prediction to b 1

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedder = pickle.load(open("./models/Vectorizer_TFIDF.pkl", "rb"))
        self.model: XGBClassifier = XGBClassifier()
        self.model.load_model("./models/XGB_TFIDF.json")

    def embed(self, tokens: list[int]) -> list[float]:
        embedded_text = self.embedder.transform([tokens])
        return embedded_text.toarray()

    def tokenize(self, text: str) -> "list[int]":
        tokens = self.tokenizer.encode(text)
        return tokens

    """
    TODO: 
    \n examples lead to faulty classification results. Tokenization goes wrong.
    
    Example: 
    def pattern(n):
        result = ''
        for i in range(1, n + 1):
            result += str(i) * i + "\n"
        return result.strip()
    """
    def predict(
        self,
        code: str = Input(description="Code to classify"),
    ) -> Output:
        """Run a single prediction on the model"""
        tokens = self.tokenize(code)
        embedded_code = self.embed(tokens)

        pred = self.model.predict(embedded_code).astype(int)
        pred_proba = self.model.predict_proba(embedded_code).astype(float)[:, 1]
        print(code)
        print({"tokens": tokens, "code": code, "prediction": pred[0], "probability": pred_proba[0]})
        return Output(prediction=pred[0], probability=pred_proba[0])


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    predictor.predict("""def pattern(n):
    result = ''
    for i in range(1, n + 1):
        result += str(i) * i + "\n"
    return result.strip()""")