import numpy as np
import Utility.utils as utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    ai_embeddings = np.stack(code_data[code_data["is_gpt"]]["embedding"].values)
    human_embeddings = np.stack(code_data[~code_data["is_gpt"]]["embedding"].values)
    assert ai_embeddings.shape[0] == human_embeddings.shape[0]

    Y = np.concatenate(
        (np.ones(ai_embeddings.shape[0]), np.zeros(human_embeddings.shape[0])), axis=0
    ).astype("float32")
    X = np.concatenate((ai_embeddings, human_embeddings), axis=0)

    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    neural_network = utils._build_model(input_dim=X.shape[1])
    history = utils._fit_model(neural_network, XTrain, YTrain, XTest, YTest)
    utils._visualize_hist(history, show=True, save=True)
    y_pred = neural_network.predict(XTest)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    metrics = utils._evaluate_model(YTest, y_pred)
    print(metrics)


main()
