import numpy as np
import Utility.utils as utils
from sklearn.linear_model import LogisticRegression


def train_eval_model(X_train, X_test, y_train, y_test):
    random_forest = LogisticRegression()
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    y_prob = random_forest.predict_proba(X_test)[:, 1]
    utils.save_results(y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="LR_TFIDF")
    utils.save_probas(y_test=y_test, y_prob=y_prob, file_name="LR_TFIDF")
    utils.print_pretty_results(index_start=-1, file_name="LR_TFIDF")


def run_on_problems(code_data, seed):
    X_train, X_test = utils._split_on_problems(X=code_data, seed=seed, test_size=0.2)

    y_train = X_train["is_gpt"].values
    y_test = X_test["is_gpt"].values

    X_train, embedder = utils._tfidf(corpus=X_train["code"].values, max_features=1536)
    X_test = embedder.transform(utils._tokenize(corpus=X_test["code"].values)).toarray()

    train_eval_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def run(dataset, split, seed):
    file_path = f"Final_Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    run_on_problems(code_data=code_data, seed=seed)
