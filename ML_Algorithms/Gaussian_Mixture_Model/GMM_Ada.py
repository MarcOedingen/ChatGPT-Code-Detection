import numpy as np
from sklearn import mixture
import Utility.utils as utils

def train_eval_model(X_AI_train, X_human_train, X_test, y_test):
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
        ai_log_likelihood[l] = gmm_ai.score_samples([X_test[l]]).sum()
        human_log_likelihood[l] = gmm_human.score_samples([X_test[l]]).sum()

    logits = np.vstack((ai_log_likelihood, human_log_likelihood)).T
    y_prob = np.apply_along_axis(utils.softmax, 1, logits)[:, 0]
    y_pred = np.where(y_prob >= 0.5, 1, 0)
    y_test = np.where(y_test == True, 1, 0)
    utils.save_results(y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="GMM_Ada")
    utils.save_probas(y_test=y_test, y_prob=y_prob, file_name="GMM_Ada")
    utils.print_pretty_results(index_start=-1, file_name="GMM_Ada")


def run_on_problems(code_data, seed):
    X_train_com, X_test_com = utils._split_on_problems(
        X=code_data, seed=seed, test_size=0.2
    )

    ai_embedding = X_train_com[X_train_com["label"] == 1]["embedding"].values
    human_embedding = X_train_com[X_train_com["label"] == 0]["embedding"].values
    assert ai_embedding.shape[0] == human_embedding.shape[0]

    X_AI_train = np.stack(ai_embedding)
    X_human_train = np.stack(human_embedding)

    X_test = np.stack(X_test_com["embedding"].values)
    y_test = X_test_com["label"].values

    train_eval_model(
        X_AI_train=X_AI_train, X_human_train=X_human_train, X_test=X_test, y_test=y_test
    )


def run(dataset, seed):
    file_path = f"Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    run_on_problems(code_data=code_data, seed=seed)
