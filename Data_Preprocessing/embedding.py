import jsonlines
import Utility.utils as utils
from OpenAI_API.ada_002 import Embedder


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    embedder = Embedder()
    for index, row in code_data.iterrows():
        print(f"Embedding row {index}/{len(code_data)}")
        row["embedding"] = embedder._embed(row["code"])
        with jsonlines.open(
            "Final_Datasets/Paired_Embedded_Cleaned.jsonl", mode="a"
        ) as writer:
            writer.write(row.to_dict)
