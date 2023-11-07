import spyct
import numpy as np
import Utility.utils as utils
from xgboost import XGBClassifier
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ML_Algorithms.Feature_Based.feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier

def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path)

    ai_code = code_data[code_data["is_gpt"]]["code"]
    human_code = code_data[~code_data["is_gpt"]]["code"]
    assert ai_code.shape[0] == human_code.shape[0]

    featureExtractor = FeatureExtractor()
    feature_names = featureExtractor._get_feature_names()

    tabs_ai = ai_code.apply(utils.extract_number_of_tabs)
    tabs_human = human_code.apply(utils.extract_number_of_tabs)

    empty_lines_ai = ai_code.apply(utils.extract_number_of_empty_lines)
    empty_lines_human = human_code.apply(utils.extract_number_of_empty_lines)

    inline_whitespace_ai = ai_code.apply(utils.number_of_inline_whitespace)
    inline_whitespace_human = human_code.apply(utils.number_of_inline_whitespace)

    punctuation_ai = ai_code.apply(utils.number_of_punctuation)
    punctuation_human = human_code.apply(utils.number_of_punctuation)

    length_of_lines_ai = ai_code.apply(utils.extract_length_of_lines)
    length_of_lines_human = human_code.apply(utils.extract_length_of_lines)

    max_line_length_ai = ai_code.apply(utils.extract_max_line_length)
    max_line_length_human = human_code.apply(utils.extract_max_line_length)

    number_of_trailing_whitespaces_ai = ai_code.apply(utils.extract_number_of_trailing_whitespaces)
    number_of_trailing_whitespaces_human = human_code.apply(utils.extract_number_of_trailing_whitespaces)

    number_of_leading_whitespaces_ai = ai_code.apply(utils.extract_number_of_leading_whitespaces)
    number_of_leading_whitespaces_human = human_code.apply(utils.extract_number_of_leading_whitespaces)

    complex_whitespaces_ai = ai_code.apply(utils.extract_complex_whitespaces)
    complex_whitespaces_human = human_code.apply(utils.extract_complex_whitespaces)

    X = np.array(
        [
            tabs_ai,
            empty_lines_ai,
            inline_whitespace_ai,
            punctuation_ai,
            length_of_lines_ai,
            max_line_length_ai,
            number_of_trailing_whitespaces_ai,
            number_of_leading_whitespaces_ai,
            complex_whitespaces_ai,
        ]
    ).T

    X = np.concatenate((X, np.array(
        [
            tabs_human,
            empty_lines_human,
            inline_whitespace_human,
            punctuation_human,
            length_of_lines_human,
            max_line_length_human,
            number_of_trailing_whitespaces_human,
            number_of_leading_whitespaces_human,
            complex_whitespaces_human,
        ]
    ).T))

    Y = np.array([1] * ai_code.shape[0] + [0] * human_code.shape[0])

    models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Bagging",
        "Gradient Boosting",
        "AdaBoost",
        "XGBoost",
        "Neural Network",
        "OPCTs",
    ]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    classifiers = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        BaggingClassifier(),
        GradientBoostingClassifier(),
        AdaBoostClassifier(),
        XGBClassifier(),
    ]
    metrics = utils.train_Classifier(X_train, Y_train, X_test, Y_test, classifiers)

    neural_network = utils._build_model(input_dim=X.shape[1])
    utils._fit_model(neural_network, X_train, Y_train, X_test, Y_test)
    y_pred = neural_network.predict(X_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    metrics = np.concatenate((metrics, utils._evaluate_model(Y_test, y_pred).reshape(1, 3)))

    n_opcts = 10
    best_performance = np.zeros((3))
    opcts = [spyct.Model(num_trees=1) for _ in range(n_opcts)]
    for i in range(n_opcts):
        opcts[i].fit(X_train, to_categorical(Y_train))
        y_pred = np.argmax(opcts[i].predict(X_test), axis=1)
        if np.mean(utils._evaluate_model(Y_test, y_pred)) > np.mean(best_performance):
            best_performance = utils._evaluate_model(Y_test, y_pred)
    metrics = np.concatenate((metrics, best_performance.reshape(1, 3)))

    print("Accuracy\tPrecision\tRecall")
    for i in range(len(models)):
        print(f"{models[i]}\t{np.round(metrics[i][0], 3)}\t\t{np.round(metrics[i][1], 2)}\t\t{np.round(metrics[i][2], 2)}")




main()