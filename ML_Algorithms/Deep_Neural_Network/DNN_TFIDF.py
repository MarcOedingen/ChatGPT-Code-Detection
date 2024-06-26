import numpy as np
import Utility.utils as utils


def train_eval_model(X_train, X_test, y_train, y_test):
    DNN = utils._build_DNN(input_dim=X_train.shape[1])
    utils._fit_model(
        model=DNN,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    y_prob = np.squeeze(DNN.predict(X_test))
    y_pred = np.where(y_prob >= 0.5, 1, 0)
    y_test = np.where(y_test == True, 1, 0)
    utils.save_results(
        y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="DNN_TFIDF"
    )
    utils.save_probas(y_test=y_test, y_prob=y_prob, file_name="DNN_TFIDF")
    utils.print_pretty_results(index_start=-1, file_name="DNN_TFIDF")


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
