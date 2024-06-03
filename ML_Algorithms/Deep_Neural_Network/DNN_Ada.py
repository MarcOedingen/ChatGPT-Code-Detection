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
    utils.save_results(y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="DNN_Ada")
    utils.save_probas(y_test=y_test, y_prob=y_prob, file_name="DNN_Ada")
    utils.print_pretty_results(index_start=-1, file_name="DNN_Ada")


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
