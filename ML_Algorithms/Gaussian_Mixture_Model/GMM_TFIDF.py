import numpy as np
from sklearn import mixture
import Utility.utils as utils
from sklearn.model_selection import train_test_split


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    ai_code = code_data[code_data["is_gpt"]]["code"].values
    human_code = code_data[~code_data["is_gpt"]]["code"].values
    assert ai_code.shape[0] == human_code.shape[0]

    X_AI_train, X_AI_test = train_test_split(
        utils._tfidf(ai_code), test_size=0.2, random_state=42
    )
    X_human_train, X_human_test = train_test_split(
        utils._tfidf(human_code), test_size=0.2, random_state=42
    )
    X_test = np.concatenate((X_AI_test, X_human_test))
    Y_test = np.concatenate((np.ones(len(X_AI_test)), np.zeros(len(X_human_test))))

    n_components = 1
    gmm_ai = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        verbose=10,
        verbose_interval=1,
    ).fit(X_AI_train)
    gmm_human = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        verbose=10,
        verbose_interval=1,
    ).fit(X_human_train)

    ai_log_likelihood = np.zeros(len(X_test))
    human_log_likelihood = np.zeros(len(X_test))
    for l in range(len(X_test)):
        ai_log_likelihood[l] = utils._compute_log_likelihood(X_test[l], gmm_ai)
        human_log_likelihood[l] = utils._compute_log_likelihood(X_test[l], gmm_human)

    Y_hat = np.where(ai_log_likelihood > human_log_likelihood, 1, 0)
    metrics = utils._evaluate_model(Y_test, Y_hat)
    print(metrics)


main()
