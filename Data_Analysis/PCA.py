import numpy as np
import Utility.utils as utils
from sklearn.decomposition import PCA


def _perform_pca(embeddings, components):
    explained_variance = np.zeros(len(components))
    for i, component in enumerate(components):
        pca = PCA(svd_solver="auto", n_components=component)
        pca.fit(
            embeddings
        )  # TODO: check if this is correct or whether we should be using embeddings.T
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        explained_variance[i] = explained_variance[-1]
    return explained_variance


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    embeddings = np.stack(code_data["embedding"].values)
    n_components = [2, 32, 256, 512, 768, 1024, 1536]
    explained_variance = _perform_pca(embeddings=embeddings, components=n_components)
    for i in range(len(n_components)):
        print(
            f"{n_components[i]} components with {explained_variance[i] * 100}% explained variance"
        )


main()
