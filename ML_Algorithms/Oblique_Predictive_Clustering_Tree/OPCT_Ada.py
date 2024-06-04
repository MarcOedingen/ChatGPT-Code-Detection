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
        if np.mean(
            utils._evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob)
        ) > np.mean(best_performance):
            best_performance = utils._evaluate_model(
                y_test=y_test, y_pred=y_pred, y_prob=y_prob
            )
            best_prob = y_prob
            best_opct = opcts[i]

    y_pred = np.argmax(best_opct.predict(X_test), axis=1)
    utils.save_results(
        y_test=y_test, y_pred=y_pred, y_prob=best_prob, file_name="OPCT_Ada"
    )
    utils.save_probas(y_test=y_test, y_prob=best_prob, file_name="OPCT_Ada")
    utils.print_pretty_results(index_start=-1, file_name="OPCT_Ada")


def run_on_problems(code_data, seed):
    X_train_com, X_test_com = utils._split_on_problems(
        X=code_data, seed=seed, test_size=0.2
    )

    y_train = X_train_com["label"].values
    y_test = X_test_com["label"].values

    X_train = np.stack(X_train_com["embedding"].values)
    X_test = np.stack(X_test_com["embedding"].values)

    train_eval_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def run(dataset, seed):
    file_path = f"Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    run_on_problems(code_data=code_data, seed=seed)
