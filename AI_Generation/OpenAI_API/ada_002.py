from ratelimit import limits, sleep_and_retry
from openai.embeddings_utils import get_embedding


class Embedder:
    def __init__(self, engine="text-embedding-ada-002"):
        self.engine = engine

    @sleep_and_retry
    @limits(calls=200, period=60)
    def _embed(self, func):
        try:
            return get_embedding(func, engine=self.engine)
        except Exception as e:
            print(e)
            print(f" Error in API call. Please try again.")
            return None

    def _embed_all(self, df, col_name):
        return df[col_name].progress_apply(lambda x: self._embed(x))
