import optuna
import numpy as np
import Utility.utils as utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def objective(trial, mean_X, cov_X, Y, mean_performance):
    alpha = trial.suggest_float("alpha", 0.0, 100.0)
    X = np.random.multivariate_normal(mean_X, alpha * cov_X, size=Y.shape[0])

    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    model = utils._build_model(input_dim=X.shape[1])
    utils._fit_model(model, XTrain, YTrain, XTest, YTest)
    y_pred = model.predict(XTest)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    accuracy = accuracy_score(YTest, y_pred)
    precision = precision_score(YTest, y_pred)
    recall = recall_score(YTest, y_pred)

    alpha_weight = 1e-4
    return np.mean([accuracy, precision, recall]) - mean_performance + alpha * alpha_weight

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

    n_runs = 10
    performance = np.zeros((n_runs))
    for i in range(n_runs):
        XTrain, XTest, YTrain, YTest = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        neural_network = utils._build_model(input_dim=X.shape[1])
        utils._fit_model(neural_network, XTrain, YTrain, XTest, YTest)
        y_pred = neural_network.predict(XTest)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        performance[i] = np.mean([accuracy_score(YTest, y_pred), precision_score(YTest, y_pred), recall_score(YTest, y_pred)])

    mean_performance = np.mean(performance)

    AI_X = X[Y==1]
    mean_AI_X = np.mean(AI_X, axis=0)
    cov_AI_X = np.cov(AI_X, rowvar=False)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, mean_X=mean_AI_X, cov_X=cov_AI_X, Y=Y, mean_performance=mean_performance), n_trials=100)
    best_alpha = study.best_params["alpha"]
    print(f"Best alpha: {best_alpha}")

    X = np.random.multivariate_normal(mean_AI_X, best_alpha * cov_AI_X, size=Y.shape[0])
    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    model = utils._build_model(input_dim=X.shape[1])
    utils._fit_model(model, XTrain, YTrain, XTest, YTest)
    y_pred = model.predict(XTest)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    metrics = utils._evaluate_model(YTest, y_pred)
    print(metrics)

main()