import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self):
        self._feature_names = [
            "number_of_tabs",
            "number_of_empty_lines",
            "number_of_inline_whitespace",
            "number_of_punctuation",
            "length_of_lines",
            "max_line_length",
            "number_of_trailing_whitespaces",
            "number_of_leading_whitespaces",
            "complex_whitespaces",
        ]

    def _get_feature_names(self):
        return self._feature_names

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

