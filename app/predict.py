# Prediction interface for Cog ⚙️
# https://cog.run/python

from typing import Any
import dill as pickle
from cog import BasePredictor, Input, Path
import tiktoken
import json

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedder = pickle.load(open(Path("./models/DILL_Vectorizer_TFIDF.pkl"), "rb"))
        self.model = pickle.load(open(Path("./models/XGB_TFIDF.pkl"), "rb"))

    def embed(self, tokens: "list[int]") -> "list[float]":
        embedded_text = self.embedder.transform([tokens])
        return embedded_text

    def tokenize(self, text: str) -> "list[int]":
        tokens = self.tokenizer.encode(text)
        return tokens

    def predict(
        self,
        code: str = Input(description="Code to classify"),
    ) -> dict:
        """Run a single prediction on the model"""
        tokens = self.tokenize(code)
        embedded_code = self.embed(tokens)

        pred = self.model.predict(embedded_code).astype(int)
        pred_proba = self.model.predict_proba(embedded_code).astype(float)
      
        return dict({"prediction": pred[0], "probability": pred_proba[0]})
