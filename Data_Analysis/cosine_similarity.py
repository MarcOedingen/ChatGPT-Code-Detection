import numpy as np
import seaborn as sns
import Utility.utils as utils
import matplotlib.pyplot as plt
from openai.embeddings_utils import cosine_similarity


def _calc_cosine_similarity(vector1, vector2):
    assert len(vector1) == len(vector2)
    similarities = np.zeros((len(vector1), len(vector2)))
    for i in range(len(vector1)):
        for j in range(len(vector2)):
            similarities[i, j] = cosine_similarity(vector1[i], vector2[j])
    return similarities


def _visualize_similarity(similarities, show=False, save=False):
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        similarities,
        cmap="Blues",
    )
    plt.title("Cosine Similarity Between AI and Human Embeddings")
    plt.xlabel("AI Embeddings")
    plt.ylabel("Human Embeddings")
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig("Plots/CosineSimilarityHeatmap.pdf")


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    n_samples = 40
    assert n_samples % 2 == 0
    ai_embeddings = np.zeros(shape=(n_samples, 1536))
    human_embeddings = np.zeros(shape=(n_samples, 1536))
    counter = 0
    while counter < n_samples:
        random_task = code_data[
            code_data["task_id"] == np.random.choice(code_data["task_id"].unique())
        ]
        if len(np.unique(random_task["is_gpt"])) > 1:
            ai_embeddings[counter] = (
                random_task[random_task["is_gpt"] == True]
                .sample(1)["embedding"]
                .iloc[0]
            )
            human_embeddings[counter] = (
                random_task[random_task["is_gpt"] == False]
                .sample(1)["embedding"]
                .iloc[0]
            )
            counter += 1
    cosine_similarity = _calc_cosine_similarity(ai_embeddings, human_embeddings)
    _visualize_similarity(cosine_similarity, show=True, save=True)


main()
