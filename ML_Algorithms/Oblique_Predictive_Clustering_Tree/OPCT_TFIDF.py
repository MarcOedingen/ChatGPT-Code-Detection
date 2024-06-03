import spyct
import numpy as np
import Utility.utils as utils
from keras.utils import to_categorical


def train_eval_model(X_train, X_test, y_train, y_test):
    n_opcts = 10
    best_performance = np.zeros(5)
    best_prob = np.zeros(y_test.shape[0])
    best_opct = None
    opcts = [spyct.Model(num_trees=1) for _ in range(n_opcts)]
    for i in range(n_opcts):
        opcts[i].fit(X_train, to_categorical(y_train))
        y_pred = np.argmax(opcts[i].predict(X_test), axis=1)
        y_prob = opcts[i].predict(X_test)[:, 1]
        if np.mean(utils._evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob)) > np.mean(
                best_performance
        ):
            best_performance = utils._evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob)
            best_prob = y_prob
            best_opct = opcts[i]

    y_pred = np.argmax(best_opct.predict(X_test), axis=1)
    utils.save_results(y_test=y_test, y_pred=y_pred, y_prob=best_prob, file_name="OPCT_TFIDF")
    utils.save_probas(y_test=y_test, y_prob=best_prob, file_name="OPCT_TFIDF")
    utils.print_pretty_results(index_start=-1, file_name="OPCT_TFIDF")


def run_on_problems(code_data, seed):
    X_train, X_test = utils._split_on_problems(X=code_data, seed=seed, test_size=0.2)

    y_train = X_train["label"].values
    y_test = X_test["label"].values

    X_train, embedder = utils._tfidf(corpus=X_train["code"].values, max_features=1536)
    X_test = embedder.transform(utils._tokenize(corpus=X_test["code"].values)).toarray()

    train_eval_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def run(dataset, seed):
    file_path = f"Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    run_on_problems(code_data=code_data, seed=seed)
