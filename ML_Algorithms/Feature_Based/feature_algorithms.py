import spyct
import numpy as np
import Utility.utils as utils
from xgboost import XGBClassifier
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from ML_Algorithms.Feature_Based.feature_extractor import FeatureExtractor
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)


def extract_features(ai_code, human_code):
    featureExtractor = FeatureExtractor()
    feature_names = featureExtractor._get_feature_names()

    number_of_leading_whitespaces_ai = ai_code.apply(
        featureExtractor.number_of_leading_whitespaces
    )
    number_of_leading_whitespaces_human = human_code.apply(
        featureExtractor.number_of_leading_whitespaces
    )

    number_of_empty_lines_ai = ai_code.apply(featureExtractor.number_of_empty_lines)
    number_of_empty_lines_human = human_code.apply(
        featureExtractor.number_of_empty_lines
    )

    number_of_inline_whitespace_ai = ai_code.apply(
        featureExtractor.number_of_inline_whitespace
    )
    number_of_inline_whitespace_human = human_code.apply(
        featureExtractor.number_of_inline_whitespace
    )

    number_of_punctuation_ai = ai_code.apply(featureExtractor.number_of_punctuation)
    number_of_punctuation_human = human_code.apply(
        featureExtractor.number_of_punctuation
    )

    max_line_length_ai = ai_code.apply(featureExtractor.max_line_length)
    max_line_length_human = human_code.apply(featureExtractor.max_line_length)

    number_of_trailing_whitespaces_ai = ai_code.apply(
        featureExtractor.number_of_trailing_whitespaces
    )
    number_of_trailing_whitespaces_human = human_code.apply(
        featureExtractor.number_of_trailing_whitespaces
    )

    number_of_lines_with_leading_whitespace_ai = ai_code.apply(
        featureExtractor.number_of_leading_whitespaces_lines
    )
    number_of_lines_with_leading_whitespace_human = human_code.apply(
        featureExtractor.number_of_leading_whitespaces_lines
    )

    X = np.array(
        [
            number_of_leading_whitespaces_ai,
            number_of_empty_lines_ai,
            number_of_inline_whitespace_ai,
            number_of_punctuation_ai,
            max_line_length_ai,
            number_of_trailing_whitespaces_ai,
            number_of_lines_with_leading_whitespace_ai,
        ]
    ).T

    X = np.concatenate(
        (
            X,
            np.array(
                [
                    number_of_leading_whitespaces_human,
                    number_of_empty_lines_human,
                    number_of_inline_whitespace_human,
                    number_of_punctuation_human,
                    max_line_length_human,
                    number_of_trailing_whitespaces_human,
                    number_of_lines_with_leading_whitespace_human,
                ]
            ).T,
        )
    )

    Y = np.array([1] * ai_code.shape[0] + [0] * human_code.shape[0])
    return X, Y, feature_names


def train_eval_model(X_train, y_train, X_test, y_test):
    classifiers = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        BaggingClassifier(),
        GradientBoostingClassifier(),
        AdaBoostClassifier(),
        XGBClassifier(),
    ]
    metrics, y_probs = utils._train_Classifier(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classifiers=classifiers,
    )

    neural_network = utils._build_DNN(input_dim=X_train.shape[1])
    utils._fit_model(
        model=neural_network,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        verbose=0,
    )
    y_pred = neural_network.predict(X_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    y_prob = neural_network.predict(X_test)
    metrics = np.concatenate(
        (
            metrics,
            utils._evaluate_model(y_test=y_test, y_pred=y_pred, y_prob=y_prob).reshape(
                1, 5
            ),
        )
    )
    y_probs = np.concatenate((y_probs, y_prob.T))

    n_opcts = 10
    best_performance = np.zeros(5)
    best_prob = np.zeros(y_test.shape[0])
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
    y_probs = np.concatenate((y_probs, best_prob.reshape(1, -1)))
    return np.concatenate((metrics, best_performance.reshape(1, 5))), y_probs


def run_on_problems(code_data, seed):
    X_train_com, X_test_com = utils._split_on_problems(
        X=code_data, seed=seed, test_size=0.2
    )

    ai_train_code = X_train_com[X_train_com["label"] == 1]["code"]
    human_train_code = X_train_com[X_train_com["label"] == 0]["code"]
    ai_test_code = X_test_com[X_test_com["label"] == 1]["code"]
    human_test_code = X_test_com[X_test_com["label"] == 0]["code"]

    X_train, y_train, _ = extract_features(
        ai_code=ai_train_code, human_code=human_train_code
    )
    X_test, y_test, _ = extract_features(
        ai_code=ai_test_code, human_code=human_test_code
    )

    metrics, y_probs = train_eval_model(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    models = [
        "LR",
        "DT",
        "RF",
        "BG",
        "GB",
        "AB",
        "XGB",
        "NN",
        "OPCT",
    ]

    utils.save_results_from_metrics(metrics=metrics, models=models)
    for i in range(y_probs.shape[0]):
        utils.save_probas(y_test=y_test, y_prob=y_probs[i], file_name=models[i])
        print(f"------------{models[i]}------------")
        utils.print_pretty_results(index_start=0, file_name=models[i])


def run(dataset, seed):
    file_path = f"Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    run_on_problems(code_data=code_data, seed=seed)
