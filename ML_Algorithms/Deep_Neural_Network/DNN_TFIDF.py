import numpy as np
import Utility.utils as utils
from sklearn.model_selection import train_test_split


def train_eval_model(X_train, X_test, y_train, y_test):
    neural_network = utils._build_DNN(input_dim=X_train.shape[1])
    history = utils._fit_model(
        model=neural_network,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    y_prob = np.squeeze(neural_network.predict(X_test))
    y_pred = np.where(y_prob >= 0.5, 1, 0)
    y_test = np.where(y_test == True, 1, 0)
    utils.save_results(y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="DNN_TFIDF")
    utils.save_probas(y_test=y_test, y_prob=y_prob, file_name="DNN_TFIDF")
    utils.print_pretty_results(index_start=-1, file_name="DNN_TFIDF")


def run_on_problems(code_data, seed):
    X_train, X_test = utils._split_on_problems(X=code_data, seed=seed, test_size=0.2)

    y_train = X_train["is_gpt"].values
    y_test = X_test["is_gpt"].values

    X_train, embedder = utils._tfidf(corpus=X_train["code"].values, max_features=1536)
    X_test = embedder.transform(utils._tokenize(corpus=X_test["code"].values)).toarray()

    train_eval_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def run_on_random(code_data, seed):
    X_train, X_test = train_test_split(code_data, test_size=0.2, random_state=42)

    y_train = X_train["is_gpt"].values
    y_test = X_test["is_gpt"].values

    X_train, embedder = utils._tfidf(corpus=X_train["code"].values, max_features=1536)
    X_test = embedder.transform(utils._tokenize(corpus=X_test["code"].values)).toarray()

    train_eval_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def run(dataset, split, seed):
    file_path = f"Final_Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    if split == "random":
        run_on_random(code_data=code_data, seed=seed)
    elif split == "problems":
        run_on_problems(code_data=code_data, seed=seed)
