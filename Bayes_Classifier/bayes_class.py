import tiktoken
import numpy as np
from Utility import utils
from collections import Counter


def classify_tokenized_code(
    tokenized_code, tokens_above_tau, prob_tk_given_H, prob_tk_given_G, P_H=0.5, P_G=0.5
):
    filtered_tokens = [token for token in tokenized_code if token in tokens_above_tau]

    log_prob_sum_H = np.sum(
        [np.log(prob_tk_given_H[token]) for token in filtered_tokens]
    )
    log_prob_sum_G = np.sum(
        [np.log(prob_tk_given_G[token]) for token in filtered_tokens]
    )

    log_prob_H_numerator = np.log(P_H) + log_prob_sum_H
    log_prob_G_numerator = np.log(P_G) + log_prob_sum_G

    log_denominator = np.logaddexp(log_prob_H_numerator, log_prob_G_numerator)

    log_prob_H = log_prob_H_numerator - log_denominator
    log_prob_G = log_prob_G_numerator - log_denominator

    return 0 if log_prob_H > log_prob_G else 1


def run_on_problems(code_data, seed):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tau = len(code_data) / int(1e3)

    X_train, X_test = utils._split_on_problems(X=code_data, seed=seed)
    human_train = X_train[X_train["label"] == 0]["code"]
    gpt_train = X_train[X_train["label"] == 1]["code"]

    tokenized_test = X_test["code"].apply(lambda x: tokenizer.encode(x))
    tokenized_human_train = human_train.apply(lambda x: tokenizer.encode(x))
    tokenized_gpt_train = gpt_train.apply(lambda x: tokenizer.encode(x))

    flat_human_tokens = [
        token for sublist in tokenized_human_train for token in sublist
    ]
    flat_gpt_tokens = [token for sublist in tokenized_gpt_train for token in sublist]

    human_counter = Counter(flat_human_tokens)
    gpt_counter = Counter(flat_gpt_tokens)

    human_counter_tau = {k: v for k, v in human_counter.items() if v > tau}
    gpt_counter_tau = {k: v for k, v in gpt_counter.items() if v > tau}

    tokens_above_tau = set(human_counter_tau.keys()).intersection(
        set(gpt_counter_tau.keys())
    )
    total_human_tokens = sum(human_counter_tau.values())
    total_gpt_tokens = sum(gpt_counter_tau.values())
    prob_tk_given_H = {
        token: human_counter_tau[token] / total_human_tokens
        for token in tokens_above_tau
    }
    prob_tk_given_G = {
        token: gpt_counter_tau[token] / total_gpt_tokens for token in tokens_above_tau
    }

    predictions = tokenized_test.apply(
        lambda x: classify_tokenized_code(
            x, tokens_above_tau, prob_tk_given_H, prob_tk_given_G
        )
    )

    y_test = X_test["label"]
    y_pred = predictions
    y_prob = predictions
    utils.save_results(
        y_test=y_test, y_pred=y_pred, y_prob=y_prob, file_name="Bayes_Classifier"
    )
    utils.print_pretty_results(index_start=-1, file_name="Bayes_Classifier")


def run(dataset, seed):
    file_path = f"Datasets/{dataset}_Balanced_Embedded"
    code_data = utils.load_data(file_path=file_path)
    run_on_problems(code_data=code_data, seed=seed)
