import numpy as np
from sklearn import mixture
import Utility.utils as utils
from sklearn.model_selection import train_test_split


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
    utils.save_results(y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="GMM_TFIDF")
    utils.save_probas(y_test=y_test, y_prob=y_prob, file_name="GMM_TFIDF")
    utils.print_pretty_results(index_start=-1, file_name="GMM_TFIDF")

def run_on_problems(code_data, seed):
    X_train_com, X_test_com = utils._split_on_problems(
        X=code_data, seed=seed, test_size=0.2
    )

    ai_code = X_train_com[X_train_com["is_gpt"]]["code"].values
    human_code = X_train_com[~X_train_com["is_gpt"]]["code"].values
    assert ai_code.shape[0] == human_code.shape[0]

    X_train, X_train_emb = utils._tfidf(
        np.concatenate((ai_code, human_code)), max_features=1536
    )
    X_AI_train = X_train[: len(ai_code)]
    X_human_train = X_train[len(ai_code) :]

    X_test = X_train_emb.transform(utils._tokenize(X_test_com["code"].values)).toarray()
    y_test = X_test_com["is_gpt"].values

    train_eval_model(
        X_AI_train=X_AI_train, X_human_train=X_human_train, X_test=X_test, y_test=y_test
    )


def run_on_random(code_data, seed):
    ai_code = code_data[code_data["is_gpt"]]["code"].values
    human_code = code_data[~code_data["is_gpt"]]["code"].values
    assert ai_code.shape[0] == human_code.shape[0]

    X_AI_train, X_AI_test = train_test_split(
        utils._tfidf(ai_code, max_features=1536)[0], test_size=0.2, random_state=seed
    )
    X_human_train, X_human_test = train_test_split(
        utils._tfidf(human_code, max_features=1536)[0], test_size=0.2, random_state=seed
    )
    X_test = np.concatenate((X_AI_test, X_human_test))
    y_test = np.concatenate((np.ones(len(X_AI_test)), np.zeros(len(X_human_test))))

    train_eval_model(
        X_AI_train=X_AI_train, X_human_train=X_human_train, X_test=X_test, y_test=y_test
    )


def run(dataset, split, seed):
    file_path = f"Final_Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    if split == "random":
        run_on_random(code_data=code_data, seed=seed)
    elif split == "problems":
        run_on_problems(code_data=code_data, seed=seed)
