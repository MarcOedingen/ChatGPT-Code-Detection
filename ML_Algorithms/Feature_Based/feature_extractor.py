import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class FeatureExtractor:
    def __init__(self):
        self._feature_names = [
            "number_of_leading_whitespaces",
            "number_of_empty_lines",
            "number_of_inline_whitespace",
            "number_of_punctuation",
            "max_line_length",
            "number_of_trailing_whitespaces",
            "number_of_lines_with_leading_whitespace",
        ]

    def _get_feature_names(self):
        return self._feature_names

    def number_of_leading_whitespaces(self, code):
        count = 0
        for line in code.split("\n"):
            count += len(line) - len(line.lstrip())
        return count / len(code)

    def number_of_empty_lines(self, code):
        return len([line for line in code.split("\n") if line == ""]) / len(
            code.split("\n")
        )

    def number_of_inline_whitespace(self, code):
        lines = code.split("\n")
        count = 0
        for line in lines:
            count += line.lstrip().count(" ")
        return count / len(code)

    def number_of_punctuation(self, code):
        return len(re.findall(r"[^\w\s]", code)) / len(code)

    def max_line_length(self, code):
        return np.max([len(line) for line in code.splitlines()]) / len(code)

    def number_of_trailing_whitespaces(self, code):
        return len([line for line in code.split("\n") if line.endswith(" ")]) / len(
            code.split("\n")
        )

    def number_of_leading_whitespaces_lines(self, code):
        return len([line for line in code.split("\n") if line.startswith(" ")]) / len(
            code.split("\n")
        )

    def _plot_corrMatrix(self, features, feature_names, show=False, save=False):
        df = pd.DataFrame(features, columns=feature_names)
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.subplots(figsize=(15, 12))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            plt.savefig("Plots/correlation_matrix.pdf")
