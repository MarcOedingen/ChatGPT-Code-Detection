import numpy as np
import Utility.utils as utils
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


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

    y_train = X_train_com["is_gpt"].values
    y_test = X_test_com["is_gpt"].values

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


def run_on_random(code_data, seed):
    X_train, X_test = train_test_split(code_data, test_size=0.2, random_state=seed)

    y_train = X_train["is_gpt"].values
    y_test = X_test["is_gpt"].values

    X_train = X_train["code"].values
    X_test = X_test["code"].values

    vector_size = 1536
    window = 1
    model = Word2Vec(
        sentences=X_train.tolist(),
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=8,
    )

    X_train_emb = [
        model.wv[word] for code in X_train for word in code if word in model.wv
    ]
    X_test_emb = [
        model.wv[word] for code in X_test for word in code if word in model.wv
    ]

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


def run(dataset, split, seed):
    file_path = f"Final_Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    if split == "random":
        run_on_random(code_data=code_data, seed=seed)
    elif split == "problems":
        run_on_problems(code_data=code_data, seed=seed)
