import os
import numpy as np
import pandas as pd
from Utility import utils
import reliability_diagrams as rd


def train_eval_model(X_train, X_test, y_train, y_test):
    neural_network = utils._build_DNN(input_dim=X_train.shape[1])
    utils._fit_model(neural_network, X_train, y_train, X_test, y_test)
    y_pred_prob = neural_network.predict(X_test)
    y_pred = np.where(y_pred_prob >= 0.5, 1, 0)
    return np.squeeze(y_pred), np.squeeze(y_pred_prob)


def run():
    file_path = f"Final_Datasets/Paired_Embedded_Cleaned_formatted"
    code_data = utils.load_data(file_path)

    X_train_com, X_test_com = utils._split_on_problems(code_data, test_size=0.2)

    X_train_ada = np.stack(X_train_com["embedding"].values)
    X_test_ada = np.stack(X_test_com["embedding"].values)

    X_train_tfidf, embedder = utils._tfidf(
        X_train_com["code"].values, max_features=1536
    )
    X_test_tfidf = embedder.transform(
        utils._tokenize(X_test_com["code"].values)
    ).toarray()

    y_train = X_train_com["is_gpt"].values
    y_test = X_test_com["is_gpt"].values

    if not os.path.exists("DNN_reliability.jsonl"):
        DNN_ada_pred, DNN_ada_prob = train_eval_model(
            X_train=X_train_ada, X_test=X_test_ada, y_train=y_train, y_test=y_test
        )
        DNN_tfidf_pred, DNN_tfidf_prob = train_eval_model(
            X_train=X_train_tfidf, X_test=X_test_tfidf, y_train=y_train, y_test=y_test
        )

        y_test = np.where(y_test, 1, 0)

        columns = [
            "task_id",
            "code",
            "is_gpt",
            "DNN_Ada_pred",
            "DNN_Ada_prob",
            "DNN_TFIDF_pred",
            "DNN_TFIDF_prob",
        ]
        df_reliability = pd.DataFrame(columns=columns)
        df_reliability["task_id"] = X_test_com["task_id"]
        df_reliability["code"] = X_test_com["code"]
        df_reliability["is_gpt"] = y_test
        df_reliability["DNN_Ada_pred"] = DNN_ada_pred
        df_reliability["DNN_Ada_prob"] = DNN_ada_prob
        df_reliability["DNN_TFIDF_pred"] = DNN_tfidf_pred
        df_reliability["DNN_TFIDF_prob"] = DNN_tfidf_prob
        df_reliability.to_json("DNN_reliability.jsonl", orient="records", lines=True)
    else:
        df_reliability = pd.read_json(
            "DNN_reliability.jsonl", orient="records", lines=True
        )
        DNN_ada_pred = df_reliability["DNN_Ada_pred"].values
        DNN_ada_prob = df_reliability["DNN_Ada_prob"].values
        DNN_tfidf_pred = df_reliability["DNN_TFIDF_pred"].values
        DNN_tfidf_prob = df_reliability["DNN_TFIDF_prob"].values

    # Plot the reliability diagram
    fig = rd.reliability_diagram(
        y_test,
        DNN_ada_pred,
        DNN_ada_prob,
        num_bins=10,
        draw_ece=True,
        draw_bin_importance="alpha",
        draw_averages=True,
        title="DNN_ADA",
        figsize=(10, 8),
        return_fig=True,
    )
    fig.savefig("Result_Analysis/Reliability/DNN_ADA_reliability.pdf")

    fig = rd.reliability_diagram(
        y_test,
        DNN_tfidf_pred,
        DNN_tfidf_prob,
        num_bins=10,
        draw_ece=True,
        draw_bin_importance="alpha",
        draw_averages=True,
        title="DNN_TFIDF",
        figsize=(10, 8),
        return_fig=True,
    )
    fig.savefig("Result_Analysis/Reliability/DNN_TFIDF_reliability.pdf")


def main():
    run()


main()
