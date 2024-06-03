import os
import re
import tiktoken
import jsonlines
import prettytable
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def load_data(file_path):
    with jsonlines.open(f"{file_path}.jsonl") as reader:
        data = list(reader)
    return pd.DataFrame(data)


def save_data(data, file_name):
    data.to_json(f"{file_name}.jsonl", orient="records", lines=True)


def _train_Classifier(X_train, y_train, X_test, y_test, classifiers):
    metrics = np.zeros((len(classifiers), 5))
    y_probs = np.zeros((len(classifiers), len(y_test)))
    for i in range(len(classifiers)):
        classifiers[i].fit(X_train, y_train)
        y_pred = classifiers[i].predict(X_test)
        y_prob = classifiers[i].predict_proba(X_test)[:, 1]
        y_probs[i] = y_prob
        metrics[i][0] = accuracy_score(y_test, y_pred)
        metrics[i][1] = precision_score(y_test, y_pred)
        metrics[i][2] = recall_score(y_test, y_pred)
        metrics[i][3] = 2 * (metrics[i][1] * metrics[i][2]) / (
                metrics[i][1] + metrics[i][2]
        )
        metrics[i][4] = roc_auc_score(y_test, y_prob)
    return metrics, y_probs


def _build_DNN(input_dim=1536):
    model = Sequential()
    model.add(Dense(768, input_dim=input_dim, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def _fit_model(model, X_train, y_train, X_test, y_test, verbose=1):
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=verbose,
    )
    return history


def save_probas(y_test, y_prob, file_name):
    files = os.listdir("Results")
    pattern = re.compile(rf"{file_name}_(\d+)\.csv")
    max_index = -1
    for file in files:
        match = pattern.match(file)
        if match:
            index = int(match.group(1))
            max_index = max(max_index, index)
    if max_index < 1:
        df = pd.DataFrame(
            {"run_id": 1, "true_label": y_test, "y_prob": np.squeeze(y_prob)}
        )
        df.to_csv(f"Results/{file_name}_1.csv", index=False)
    else:
        df = pd.read_csv(f"Results/{file_name}_{max_index}.csv")
        df_new = pd.DataFrame(
            {"run_id": max_index + 1, "true_label": y_test, "y_prob": y_prob}
        )
        df = pd.concat([df, df_new], axis=0)
        df.to_csv(f"Results/{file_name}_{max_index + 1}.csv", index=False)
        os.remove(f"Results/{file_name}_{max_index}.csv")


def save_results(y_test, y_pred, y_prob, file_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    auc = roc_auc_score(y_test, y_prob)
    if os.path.exists(f"Results/{file_name}.npz"):
        results = np.load(f"Results/{file_name}.npz")
        accuracy = np.append(results["accuracy"], accuracy)
        precision = np.append(results["precision"], precision)
        recall = np.append(results["recall"], recall)
        f1 = np.append(results["f1"], f1)
        auc = np.append(results["auc"], auc)
    np.savez(
        f"Results/{file_name}.npz",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=auc,
    )


def save_results_from_metrics(metrics, models):
    for i, model_name in enumerate(models):
        accuracy = metrics[i][0]
        precision = metrics[i][1]
        recall = metrics[i][2]
        f1 = metrics[i][3]
        auc = metrics[i][4]
        if os.path.exists(f"Results/{model_name}.npz"):
            results = np.load(f"Results/{model_name}.npz")
            accuracy = np.append(results["accuracy"], accuracy)
            precision = np.append(results["precision"], precision)
            recall = np.append(results["recall"], recall)
            f1 = np.append(results["f1"], f1)
            auc = np.append(results["auc"], auc)
        np.savez(
            f"Results/{model_name}.npz",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc=auc,
        )


def print_pretty_results(file_name, index_start):
    results = np.load(f"Results/{file_name}.npz")
    table = prettytable.PrettyTable()
    table.field_names = ["Metric", "Mean ± Std"]
    for key in results.files:
        data = results[key.lower()]
        if data.ndim > 0:
            index_start = data.size - 1 if index_start == -1 else index_start
            mean_val = np.round(np.mean(data[index_start:]), 3)
            std_val = np.round(np.std(data[index_start:]), 3)
            value_str = f"{mean_val} ± {std_val}"
        else:
            value_str = str(np.round(data, 3))
        table.add_row([key.capitalize(), value_str])
    print(table)


def _evaluate_model(y_test, y_pred, y_prob):
    return np.array(
        [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            2 * (precision_score(y_test, y_pred) * recall_score(y_test, y_pred))
            / (
                    precision_score(y_test, y_pred) + recall_score(y_test, y_pred)
            ),
            roc_auc_score(y_test, y_prob),
        ]
    )


def _compute_log_likelihood(x, gmm):
    return gmm.score_samples(x.reshape(1, -1)).sum()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def _compute_log_likelihood_token(x, gmm, model):
    code_vec = [model.wv[word] for word in x if word in model.wv]
    code_vec_array = np.vstack(code_vec)
    return gmm.score_samples(code_vec_array).sum()


def _tfidf(corpus, max_features=1536):
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, max_features=max_features)
    return vectorizer.fit_transform(_tokenize(corpus)).toarray(), vectorizer


def _tokenize(corpus):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenized_corpus = [tokenizer.encode(code) for code in corpus]
    return tokenized_corpus


# Cluster sampling
def _split_on_problems(X, seed, test_size=0.2):
    unique_problems = X["id"].unique()
    X_Train, X_Test = train_test_split(
        unique_problems, test_size=test_size, random_state=seed
    )
    X_train = X[X["id"].isin(X_Train)]
    X_test = X[X["id"].isin(X_Test)]
    return X_train, X_test
