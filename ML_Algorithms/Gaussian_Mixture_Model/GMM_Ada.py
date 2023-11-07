import numpy as np
from sklearn import mixture
import Utility.utils as utils
from sklearn.model_selection import train_test_split

def compute_log_likelihood(x, gmm):
    return gmm.score_samples(x.reshape(1, -1)).sum()

def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    ai_embedding = np.stack(code_data[code_data["is_gpt"]]["embedding"].values)
    human_embedding = np.stack(code_data[~code_data["is_gpt"]]["embedding"].values)
    assert ai_embedding.shape[0] == human_embedding.shape[0]

    X_AI_train, X_AI_test = train_test_split(ai_embedding, test_size=0.2, random_state=42)
    X_human_train, X_human_test = train_test_split(human_embedding, test_size=0.2, random_state=42)
    X_test = np.concatenate((X_AI_test, X_human_test))
    Y_test = np.concatenate((np.ones(len(X_AI_test)), np.zeros(len(X_human_test))))

    n_components = 1
    gmm_ai = mixture.GaussianMixture(n_components=n_components, covariance_type='full', verbose=10, verbose_interval=1).fit(X_AI_train)
    gmm_human = mixture.GaussianMixture(n_components=n_components, covariance_type='full', verbose=10, verbose_interval=1).fit(X_human_train)

    ai_log_likelihood = np.zeros(len(X_test))
    human_log_likelihood = np.zeros(len(X_test))
    for l in range(len(X_test)):
        ai_log_likelihood[l] = compute_log_likelihood(X_test[l], gmm_ai)
        human_log_likelihood[l] = compute_log_likelihood(X_test[l], gmm_human)

    Y_hat = np.where(ai_log_likelihood > human_log_likelihood, 1, 0)
    metrics = utils._evaluate_model(Y_test, Y_hat)
    print(metrics)

main()