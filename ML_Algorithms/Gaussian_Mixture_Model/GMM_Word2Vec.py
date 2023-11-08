import numpy as np
from sklearn import mixture
import Utility.utils as utils
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from Data_Preprocessing.tokenization import Tokenizer


def main():
    file_path = "Final_Datasets/Paired_Embedded_Cleaned"
    code_data = utils.load_data(file_path=file_path)

    ai_code = code_data[code_data["is_gpt"]]["code"]
    human_code = code_data[~code_data["is_gpt"]]["code"]
    assert ai_code.shape[0] == human_code.shape[0]

    tokenizer = Tokenizer()
    ai_code_tokens = ai_code.apply(tokenizer._tokenize)
    human_code_tokens = human_code.apply(tokenizer._tokenize)
    all_code_tokens = np.concatenate((ai_code_tokens, human_code_tokens))

    X_AI_train, X_AI_test = train_test_split(
        ai_code_tokens, test_size=0.2, random_state=42
    )
    X_human_train, X_human_test = train_test_split(
        human_code_tokens, test_size=0.2, random_state=42
    )
    X_test = np.concatenate((X_AI_test, X_human_test))
    Y_test = np.concatenate((np.ones(len(X_AI_test)), np.zeros(len(X_human_test))))

    vector_size = 768
    window = 1
    model = Word2Vec(
        sentences=all_code_tokens.tolist(),
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=8,
    )

    ai_code_vecs = [
        model.wv[word] for code in X_AI_train for word in code if word in model.wv
    ]
    human_code_vecs = [
        model.wv[word] for code in X_human_train for word in code if word in model.wv
    ]

    n_components = 1
    gmm_ai = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        verbose=10,
        verbose_interval=1,
    ).fit(ai_code_vecs)
    gmm_human = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        verbose=10,
        verbose_interval=1,
    ).fit(human_code_vecs)

    ai_log_likelihood = np.zeros(len(X_test))
    human_log_likelihood = np.zeros(len(X_test))
    for l in range(len(X_test)):
        ai_log_likelihood[l] = utils._compute_log_likelihood_token(
            X_test[l], gmm_ai, model
        )
        human_log_likelihood[l] = utils._compute_log_likelihood_token(
            X_test[l], gmm_human, model
        )

    Y_hat = np.where(ai_log_likelihood > human_log_likelihood, 1, 0)
    metrics = utils._evaluate_model(Y_test, Y_hat)
    print(metrics)


main()
