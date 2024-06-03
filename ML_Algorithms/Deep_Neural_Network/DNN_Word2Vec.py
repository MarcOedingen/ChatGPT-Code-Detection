import numpy as np
import Utility.utils as utils
from gensim.models import Word2Vec

def train_eval_model(X_train, X_test, y_train, y_test, y_test_emb, separators):
    neural_network = utils._build_DNN(input_dim=X_train.shape[1])
    utils._fit_model(
        model=neural_network,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test_emb,
    )
    y_prob = np.zeros(len(separators))
    start_index = 0
    for i in range(len(separators)):
        X_test_split = X_test[start_index : start_index + separators[i]]
        start_index += separators[i]
        y_prob[i] = np.mean(neural_network.predict(X_test_split))
    y_pred = np.where(y_prob >= 0.5, 1, 0)
    y_test = np.where(y_test == True, 1, 0)
    utils.save_results(y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="DNN_Word2Vec")
    utils.save_probas(y_test=y_test, y_prob=y_prob, file_name="DNN_Word2Vec")
    utils.print_pretty_results(index_start=-1, file_name="DNN_Word2Vec")


def run_on_problems(code_data, seed):
    X_train_com, X_test_com = utils._split_on_problems(
        X=code_data, seed=seed, test_size=0.2
    )

    y_train = X_train_com["label"].values
    y_test = X_test_com["label"].values

    X_train = X_train_com["code"].values
    X_test = X_test_com["code"].values

    # Tokenize the code
    X_train = utils._tokenize(corpus=X_train)
    X_test = utils._tokenize(corpus=X_test)

    vector_size = 1536
    window = 5
    model = Word2Vec(
        sentences=X_train,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=8,
    )

    X_train_emb = [model.wv[word] if word in model.wv else np.zeros(vector_size) for code in X_train for word in code]
    X_test_emb = [model.wv[word] if word in model.wv else np.zeros(vector_size) for code in X_test for word in code]

    y_train = np.repeat(y_train, [len(code) for code in X_train])
    y_test_emb = np.repeat(y_test, [len(code) for code in X_test])

    train_eval_model(
        X_train=np.array(X_train_emb),
        X_test=np.array(X_test_emb),
        y_train=y_train,
        y_test=y_test,
        y_test_emb=y_test_emb,
        separators=[len(tokens) for tokens in X_test]
    )

def run(dataset, seed):
    file_path = f"Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    run_on_problems(code_data=code_data, seed=seed)
