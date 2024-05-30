import tiktoken
import numpy as np
from Utility import utils
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

code_data = utils.load_data("Final_Datasets/Unformatted_Balanced_Embedded")
encoder = tiktoken.get_encoding("cl100k_base")
tau = len(code_data) / int(1e3)

def replace_special_characters(token):
    return token.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")

def tokenize_and_filter_document(doc, encoder, tokens_above_threshold):
    tokenized_doc = encoder.encode(doc)
    filtered_tokens = [token for token in tokenized_doc if token in tokens_above_threshold]
    return filtered_tokens

def calculate_log_probability_sum(tokens, prob_tk_given_origin):
    log_prob_sum = 0.0
    for token in tokens:
        log_prob_sum += np.log(prob_tk_given_origin.get(token))
    return log_prob_sum


def classify_tokenized_document(tokenized_doc, tokens_above_threshold, prob_tk_given_H, prob_tk_given_G, P_H=0.5, P_G=0.5):
    # Filter the tokens based on the threshold
    filtered_tokens = [token for token in tokenized_doc if token in tokens_above_threshold]

    # Calculate the sum of log probabilities for the filtered tokens
    log_prob_sum_H = calculate_log_probability_sum(filtered_tokens, prob_tk_given_H)
    log_prob_sum_G = calculate_log_probability_sum(filtered_tokens, prob_tk_given_G)

    # Convert log probabilities of priors into log space for consistency
    log_P_H = np.log(P_H)
    log_P_G = np.log(P_G)

    # Apply Bayes' theorem in log space
    log_numerator_H = log_prob_sum_H + log_P_H
    log_numerator_G = log_prob_sum_G + log_P_G

    # Since we're in log space, use logsumexp to safely calculate log(denominator)
    # np.logaddexp safely computes the logarithm of the sum of exponentiations of the inputs
    log_denominator = np.logaddexp(log_numerator_H, log_numerator_G)

    # Calculate log probabilities for final comparison
    log_P_H_given_D = log_numerator_H - log_denominator
    log_P_G_given_D = log_numerator_G - log_denominator

    # Return the classification based on which log probability is higher
    return 0 if log_P_H_given_D > log_P_G_given_D else 1


X_train, X_test = utils._split_on_problems(X=code_data, seed=42)
human_train = X_train[~X_train['is_gpt']]['code']
gpt_train = X_train[X_train['is_gpt']]['code']
tokenized_test = X_test['code'].apply(lambda x: encoder.encode(x))
tokenized_human_train = human_train.apply(lambda x: encoder.encode(x))
tokenized_gpt_train = gpt_train.apply(lambda x: encoder.encode(x))
flat_human_tokens = [token for sublist in tokenized_human_train for token in sublist]
flat_gpt_tokens = [token for sublist in tokenized_gpt_train for token in sublist]
human_token_counts = Counter(flat_human_tokens)
gpt_token_counts = Counter(flat_gpt_tokens)
# Keep the count and tokens above the threshold for both human and GPT
tokens_count_human = {token: count for token, count in human_token_counts.items() if count > tau}
tokens_count_gpt = {token: count for token, count in gpt_token_counts.items() if count > tau}
# Create an intersection of tokens above the threshold and add their counts
tokens_above_threshold = set(tokens_count_human.keys()).intersection(set(tokens_count_gpt.keys()))
tokens_count = {token: tokens_count_human[token] + tokens_count_gpt[token] for token in tokens_above_threshold}
total_human_tokens = sum(human_token_counts.values())
total_gpt_tokens = sum(gpt_token_counts.values())
prob_tk_given_H = {token: (human_token_counts[token] / total_human_tokens) for token in tokens_above_threshold}
prob_tk_given_G = {token: (gpt_token_counts[token] / total_gpt_tokens) for token in tokens_above_threshold}

# Determine the top 50 tokens of the common tokens above the threshold
top_50_tokens = sorted(tokens_count, key=tokens_count.get, reverse=True)[:50]

# Sort the top 50 tokens by their probabilities given human and GPT
top_50_tokens.sort(key=lambda token: prob_tk_given_G[token], reverse=True)
top_50_tokens_decoded = [encoder.decode([token]) for token in top_50_tokens]

bar_width = 0.4

# Calculate positions for each group and bar
positions = np.arange(len(top_50_tokens))
human_positions = positions - bar_width / 2
gpt_positions = positions + bar_width / 2

# Plot the probabilities of the top 50 tokens as grouped bar chart for human and GPT
fig = plt.figure(figsize=(12, 8))
plt.bar(human_positions, [prob_tk_given_H[token] for token in top_50_tokens], width=bar_width, label="Human")
plt.bar(gpt_positions, [prob_tk_given_G[token] for token in top_50_tokens], width=bar_width, label="GPT")

# Set the x-ticks to be in the middle of the grouped bars
plt.xticks(positions, [replace_special_characters(encoder.decode([token])) for token in top_50_tokens], rotation=90)
plt.xlabel("Token")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig("Top_50_Tokens.pdf")

# Do the same plot but just for the tokens that exceeds discrepancies of 50% and rank them in descending order based on the difference
# Hence, the plot should show small human bars / large GPT bars on the left and large human bars / small GPT bars on the right
# Calculate discrepancies and their ratios
token_discrepancies = {
    token: abs(prob_tk_given_H[token] - prob_tk_given_G[token])
    for token in tokens_above_threshold
}

decoded_tokens_discrepancies = {encoder.decode([token]): discrepancy for token, discrepancy in token_discrepancies.items()}

# Filter out the tokens that have blankets in it, i.e., ' ' or '    '
filtered_tokens_discrepancies = {token: discrepancy for token, discrepancy in token_discrepancies.items() if ' ' not in encoder.decode([token])}

# Sort the tokens based on the discrepancy
tokens_sorted_by_discrepancy = sorted(filtered_tokens_discrepancies, key=filtered_tokens_discrepancies.get, reverse=True)[:40]

# Sort tokens from small human bars / large GPT bars to large human bars / small GPT bars
tokens_sorted_for_visual = sorted(tokens_sorted_by_discrepancy, key=lambda token: (prob_tk_given_H[token] - prob_tk_given_G[token]))

# Prepare data for plotting
human_probs = [prob_tk_given_H[token] for token in tokens_sorted_for_visual]
gpt_probs = [prob_tk_given_G[token] for token in tokens_sorted_for_visual]

# Calculate positions for each group and bar
positions = np.arange(len(tokens_sorted_for_visual))
human_positions = positions - bar_width / 2
gpt_positions = positions + bar_width / 2

# Create the plot
fig = plt.figure(figsize=(10, 6))
plt.bar(human_positions, human_probs, width=bar_width, label="Human")
plt.bar(gpt_positions, gpt_probs, width=bar_width, label="GPT")

# Set the x-ticks to be in the middle of the grouped bars
plt.xticks(positions, [replace_special_characters(encoder.decode([token])) for token in tokens_sorted_for_visual], rotation=90)
plt.xlabel("Token")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.savefig("Top_40_Tokens_Discrepancies.pdf")

# Logarithmiere alle Probabilities, also L_H = log(p_H), L_G = log(p_G) und wähle die Tokens mit größtem |L_H – L_G| aus.
# Sortiere diese Tokens nach |L_H – L_G| absteigend und wähle die ersten 40 Tokens aus.
# Plotte die Wahrscheinlichkeiten der Tokens als gruppiertes Balkendiagramm für Mensch und GPT.

# Calculate the discrepancies and their ratios

log_prob_tk_given_H = {token: np.log(prob_tk_given_H[token]) for token in tokens_above_threshold}
log_prob_tk_given_G = {token: np.log(prob_tk_given_G[token]) for token in tokens_above_threshold}

# filter tokens that contain blanks
log_prob_discrepancies = {
    token: abs(log_prob_tk_given_H[token] - log_prob_tk_given_G[token])
    for token in tokens_above_threshold if ' \n' not in encoder.decode([token] or '\n ' not in encoder.decode([token]))
}

# filter out the same tokens as for the log_prob_discrepancies from "tokens_above_threshold"
tokens_above_threshold = {token for token in tokens_above_threshold if ' \n' not in encoder.decode([token] or '\n ' not in encoder.decode([token]))}

decoded_tokens_log_discrepancies = {encoder.decode([token]): discrepancy for token, discrepancy in log_prob_discrepancies.items()}

tokens_sorted_by_log_discrepancy = sorted(tokens_above_threshold, key=log_prob_discrepancies.get, reverse=True)[:40]

# Sort tokens from small human bars / large GPT bars to large human bars / small GPT bars
tokens_sorted_for_visual = sorted(tokens_sorted_by_log_discrepancy, key=lambda token: (log_prob_tk_given_H[token] - log_prob_tk_given_G[token]))

# Prepare data for plotting
human_probs = [prob_tk_given_H[token] for token in tokens_sorted_for_visual]
gpt_probs = [prob_tk_given_G[token] for token in tokens_sorted_for_visual]

# Calculate positions for each group and bar
positions = np.arange(len(tokens_sorted_for_visual))
human_positions = positions - bar_width / 2
gpt_positions = positions + bar_width / 2

# Create the plot
fig = plt.figure(figsize=(10, 6))
plt.bar(human_positions, human_probs, width=bar_width, label="Human")
plt.bar(gpt_positions, gpt_probs, width=bar_width, label="GPT")

# Set the x-ticks to be in the middle of the grouped bars
plt.xticks(positions, [replace_special_characters(encoder.decode([token])) for token in tokens_sorted_for_visual], rotation=90)
plt.xlabel(r"Token $T_k$")
plt.ylabel(r"Probability $P(T_k | X)$")
plt.legend()
plt.tight_layout()
plt.savefig("Top_40_Tokens_Log_Discrepancies.pdf")

"""accuracies, precisions, recalls, f1_scores = [], [], [], []
avg_tokens_per_snippet = []
avg_tokens_above_threshold = []
seeds = [7784, 7570, 7592, 9466, 3606, 1143, 2892, 42, 3290, 3722]
for seed in seeds:
    X_train, X_test = utils._split_on_problems(X=code_data, seed=seed)
    human_train = X_train[~X_train['is_gpt']]['code']
    gpt_train = X_train[X_train['is_gpt']]['code']
    tokenized_test = X_test['code'].apply(lambda x: encoder.encode(x))
    avg_tokens_per_snippet.append(tokenized_test.apply(len).mean())
    tokenized_human_train = human_train.apply(lambda x: encoder.encode(x))
    tokenized_gpt_train = gpt_train.apply(lambda x: encoder.encode(x))
    flat_human_tokens = [token for sublist in tokenized_human_train for token in sublist]
    flat_gpt_tokens = [token for sublist in tokenized_gpt_train for token in sublist]
    human_token_counts = Counter(flat_human_tokens)
    gpt_token_counts = Counter(flat_gpt_tokens)
    tokens_above_threshold_human = {token for token, count in human_token_counts.items() if count > tau}
    tokens_above_threshold_gpt = {token for token, count in gpt_token_counts.items() if count > tau}
    tokens_above_threshold = tokens_above_threshold_human.intersection(tokens_above_threshold_gpt)
    avg_tokens_above_threshold.append(len(tokens_above_threshold))
    total_human_tokens = sum(human_token_counts.values())
    total_gpt_tokens = sum(gpt_token_counts.values())
    prob_tk_given_H = {token: (human_token_counts[token] / total_human_tokens) for token in tokens_above_threshold}
    prob_tk_given_G = {token: (gpt_token_counts[token] / total_gpt_tokens) for token in tokens_above_threshold}
    predictions = tokenized_test.apply(lambda x: classify_tokenized_document(x, tokens_above_threshold, prob_tk_given_H, prob_tk_given_G))
    accuracy = accuracy_score(X_test['is_gpt'], predictions) * 100
    precision = precision_score(X_test['is_gpt'], predictions) * 100
    recall = recall_score(X_test['is_gpt'], predictions) * 100
    f1 = f1_score(X_test['is_gpt'], predictions) * 100
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    print(f"Run {seed}: Accuracy: {accuracy:.2f} Precision: {precision:.2f} Recall: {recall:.2f} F1: {f1:.2f}")

print(f"Average Tokens Per Snippet: {np.mean(avg_tokens_per_snippet):.2f} +/- {np.std(avg_tokens_per_snippet):.2f}")
print(f"Average Tokens Above Threshold: {np.mean(avg_tokens_above_threshold):.2f} +/- {np.std(avg_tokens_above_threshold):.2f}")
print(f"Average Accuracy: {np.mean(accuracies):.2f} +/- {np.std(accuracies):.2f}")
print(f"Average Precision: {np.mean(precisions):.2f} +/- {np.std(precisions):.2f}")
print(f"Average Recall: {np.mean(recalls):.2f} +/- {np.std(recalls):.2f}")
print(f"Average F1 Score: {np.mean(f1_scores):.2f} +/- {np.std(f1_scores):.2f}")"""
