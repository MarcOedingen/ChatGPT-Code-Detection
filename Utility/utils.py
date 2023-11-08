import re
import jsonlines
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from Data_Preprocessing.tokenization import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score


def _visualize_hist(history, show=False, save=False):
    plt.figure(figsize=(10, 10))
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["loss"])
    plt.title("Model performance in training")
    plt.ylabel("Accuracy / Loss")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Loss"], loc="upper left")
    if show:
        plt.show()
    if save:
        plt.savefig("Plots/DNN_Ada_training.pdf")


def load_data(file_path):
    with jsonlines.open(f"{file_path}.jsonl") as reader:
        data = list(reader)
    return pd.DataFrame(data)


def save_data(data, file_name):
    data.to_json(f"{file_name}.jsonl", orient="records", lines=True)


def extract_number_of_tabs(code):
    count = 0
    for line in code.split("\n"):
        count += len(line) - len(line.lstrip())
    return count / len(code)


def number_of_inline_whitespace(code):
    lines = code.split("\n")
    count = 0
    for line in lines:
        count += line.lstrip().count(" ")
    return count / len(code)


def extract_number_of_empty_lines(code):
    return len([line for line in code.split("\n") if line == ""]) / len(
        code.split("\n")
    )


def number_of_punctuation(code):
    return len(re.findall(r"[^\w\s]", code)) / len(code)


def extract_length_of_lines(code):
    return np.sum([len(line) for line in code.splitlines()]) / len(code)


def extract_max_line_length(code):
    return np.max([len(line) for line in code.splitlines()]) / len(code)


def extract_number_of_trailing_whitespaces(code):
    return len([line for line in code.split("\n") if line.endswith(" ")]) / len(
        code.split("\n")
    )


def extract_number_of_leading_whitespaces(code):
    return len([line for line in code.split("\n") if line.startswith(" ")]) / len(
        code.split("\n")
    )


def extract_complex_whitespaces(code):
    count_space = code.count(" ")
    count_tab = extract_number_of_tabs(code)
    ratio = count_space + count_tab
    return count_space / ratio if ratio > 0 else 0


def _train_Classifier(X_train, y_train, X_test, y_test, classifiers):
    metrics = np.zeros((len(classifiers), 3))
    for i in range(len(classifiers)):
        classifiers[i].fit(X_train, y_train)
        y_pred = classifiers[i].predict(X_test)
        metrics[i][0] = accuracy_score(y_test, y_pred)
        metrics[i][1] = precision_score(y_test, y_pred)
        metrics[i][2] = recall_score(y_test, y_pred)
    return metrics


def _build_model(input_dim=1536):
    model = Sequential()
    model.add(Dense(768, input_dim=input_dim, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def _fit_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )
    return history


def _evaluate_model(y_test, y_pred):
    return np.array(
        [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
        ]
    )


def _compute_log_likelihood(x, gmm):
    return gmm.score_samples(x.reshape(1, -1)).sum()


def _compute_log_likelihood_token(x, gmm, model):
    code_vec = [model.wv[word] for word in x if word in model.wv]
    code_vec_array = np.vstack(code_vec)
    return gmm.score_samples(code_vec_array).sum()


def _tfidf(corpus):
    tokenizer = Tokenizer()
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, max_features=512)
    tokenized_corpus = [tokenizer._tokenize(code) for code in corpus]
    return vectorizer.fit_transform(tokenized_corpus).toarray()
