import numpy as np
from sklearn import mixture
import Utility.utils as utils
from gensim.models import Word2Vec


def train_eval_model(X_AI_train, X_human_train, X_test, y_test, model):
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
        ai_log_likelihood[l] = utils._compute_log_likelihood_token(
            x=X_test[l], gmm=gmm_ai, model=model
        )
        human_log_likelihood[l] = utils._compute_log_likelihood_token(
            x=X_test[l], gmm=gmm_human, model=model
        )

    logits = np.vstack((ai_log_likelihood, human_log_likelihood)).T
    y_prob = np.apply_along_axis(utils.softmax, 1, logits)[:, 0]
    y_pred = np.where(y_prob >= 0.5, 1, 0)
    y_test = np.where(y_test == True, 1, 0)
    utils.save_results(y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="GMM_Word2Vec")
    utils.save_probas(y_test=y_test, y_prob=y_prob, file_name="GMM_Word2Vec")
    utils.print_pretty_results(index_start=-1, file_name="GMM_Word2Vec")


def run_on_problems(code_data, seed):
    X_train_com, X_test_com = utils._split_on_problems(
        X=code_data, seed=seed, test_size=0.2
    )

    ai_code = X_train_com[X_train_com["label"] == 1]["code"].values
    human_code = X_train_com[X_train_com["label"] == 0]["code"].values
    assert ai_code.shape[0] == human_code.shape[0]

    X_train_ai_tokens = utils._tokenize(corpus=ai_code)
    X_train_human_tokens = utils._tokenize(corpus=human_code)

    X_test = utils._tokenize(corpus=X_test_com["code"].values)
    y_test = X_test_com["label"].values

    vector_size = 1536
    window = 1
    model = Word2Vec(
        sentences=X_train_ai_tokens + X_train_human_tokens,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=8,
    )

    X_train_ai_tokens_emb = [
        model.wv[word]
        for code in X_train_ai_tokens
        for word in code
        if word in model.wv
    ]
    X_train_human_tokens_emb = [
        model.wv[word]
        for code in X_train_human_tokens
        for word in code
        if word in model.wv
    ]

    train_eval_model(
        X_AI_train=X_train_ai_tokens_emb,
        X_human_train=X_train_human_tokens_emb,
        X_test=X_test,
        y_test=y_test,
        model=model,
    )


def run(dataset, seed):
    file_path = f"Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    run_on_problems(code_data=code_data, seed=seed)
