import numpy as np
import Utility.utils as utils
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def _perform_t_sne(embeddings, n_components, perplexity, n_iter):
    t_sne = TSNE(
        n_components=n_components, perplexity=perplexity, n_iter=n_iter
    ).fit_transform(embeddings)
    return t_sne


def _visualize_t_sne2D(t_sne, labels, show=False, save=False):
    plt.figure(figsize=(10, 10))
    plt.scatter(t_sne[:, 0], t_sne[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("t-SNE 2D")
    if show:
        plt.show()
    if save:
        plt.savefig("Plots/tSNE2D.pdf")


def _visualize_t_sne3D(t_sne, labels, show=False, save=False):
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.scatter3D(t_sne[:, 0], t_sne[:, 1], t_sne[:, 2], c=labels)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    plt.title("t-SNE 3D")
    if show:
        plt.show()
    if save:
        plt.savefig("Plots/tSNE3D.pdf")


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    embeddings = np.stack(code_data["embedding"].values)
    perplexity = 30
    n_iter = int(3e2)
    t_sne_2D = _perform_t_sne(
        embeddings=embeddings, n_components=2, perplexity=perplexity, n_iter=n_iter
    )
    t_sne_3D = _perform_t_sne(
        embeddings=embeddings, n_components=3, perplexity=perplexity, n_iter=n_iter
    )

    labels = np.zeros(len(code_data))
    labels[code_data["is_gpt"] == True] = 1
    _visualize_t_sne2D(t_sne=t_sne_2D, labels=labels, show=True, save=True)
    _visualize_t_sne3D(t_sne=t_sne_3D, labels=labels, show=True, save=True)


main()
