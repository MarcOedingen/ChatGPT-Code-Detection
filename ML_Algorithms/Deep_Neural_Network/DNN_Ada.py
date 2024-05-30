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
    utils.save_results(y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="DNN_Ada")
    utils.save_probas(y_test=y_test, y_prob=y_prob, file_name="DNN_Ada")
    utils.print_pretty_results(index_start=-1, file_name="DNN_Ada")


def run_on_problems(code_data, seed):
    X_train_com, X_test_com = utils._split_on_problems(
        X=code_data, seed=seed, test_size=0.2
    )

    y_train = X_train_com["is_gpt"].values
    y_test = X_test_com["is_gpt"].values

    X_train = np.stack(X_train_com["embedding"].values)
    X_test = np.stack(X_test_com["embedding"].values)

    train_eval_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def run_on_random(code_data, seed):
    Y = code_data["is_gpt"].values
    X = np.stack(code_data["embedding"].values)

    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=0.2, random_state=seed
    )

    train_eval_model(X_train=XTrain, X_test=XTest, y_train=YTrain, y_test=YTest)


def run(dataset, split, seed):
    file_path = f"Final_Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    if split == "random":
        run_on_random(code_data=code_data, seed=seed)
    elif split == "problems":
        run_on_problems(code_data=code_data, seed=seed)
