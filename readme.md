# AI Code Detection

## Introduction
This work was part of the Guided Project: ChatGPT in Software Enginerring Academic Education. Fraud Tool, or useful 
helper? An Empirical Study, see [here](https://www.archi-lab.io/projects/ss23/gp_chatgpt_ss23.html) for more details.

The goal of this work is to distinguish between AI- and human-generated code. Inital results and technical 
descriptions of the process can be found in the technical report: [here](Code_Detection_Paper.pdf). This document also 
describes the motivation and the background of the project. Further, is contains a discussion of the results and an
outlook on Future Work.

## Data
The data used for this project is a collection of Python code snippets from various sources:
- APPS Data: [here](https://github.com/hendrycks/apps); Paper: [here](https://arxiv.org/pdf/2105.09938v3.pdf)
- CodeChef Data: [here](https://www.codechef.com/); Paper: /
- CodeContests Data: [here](https://huggingface.co/datasets/deepmind/code_contests); Paper: [here](https://arxiv.org/abs/2203.07814)
- HackerEarth Data: [here](https://www.hackerearth.com/); Paper: /
- HumanEval Data: [here](https://github.com/openai/human-eval); Paper: [here](https://arxiv.org/abs/2107.03374)
- MBPPD Data: [here](https://github.com/google-research/google-research/tree/master/mbpp); Paper: [here](https://arxiv.org/abs/2108.07732)
- MTrajK Data: [here](https://github.com/MTrajK/); Paper: /

Looking forward to further data sources:
- CSES Data: [here](https://cses.fi/problemset/); Paper: /
- DS-1000 Data: [here](https://ds1000-code-gen.github.io); Paper: [here](https://arxiv.org/abs/2211.11501)
- edabit Data: [here](https://edabit.com/challenges/python3); Paper: /
- LeetCode Data: [here](https://leetcode.com/); Paper: /
- 150k PSCD Data: [here](https://www.kaggle.com/datasets/veeralakrishna/150k-python-dataset); Paper [here](https://dl.acm.org/doi/10.1145/2983990.2984041)

Since every data source has its own format (e.g. JSON, CSV, ...) and structure, the data is preprocessed and stored in
a unified format. Additionally, each source provides different formatted code snippets and test-cases. Therefore, the
data is also preprocessed for each source individually.


## Structure of the Project
```
.
├── README.md
├── requirements.txt
├── Code_Detection_Paper.pdf
├── Problem Collection
│   ├── APPS
│   ├── CodeChef
│   ├── CodeContests
│   ├── HackerEarth
│   ├── HumanEval
│   ├── MBPPD
│   ├── MTrajK
├── AI Generation
│   ├── OpenAI_API
│   │   ├── ada_002.py
│   │   ├── GPT3_5_turbo.py
├── Data Preprocessing
│   ├── Embedding
│   ├── Formatting
│   ├── Missing Values
│   ├── Outlier Detection
│   ├── Pairing
│   ├── Tokenization
├── Final Datasets
│   ├── Formatted Paired Embedded without outliers
│   ├── Unformatted Paired Embedded without outliers
├── Data Analysis
│   ├── Cosine Similarity
│   ├── PCA
│   ├── t-SNE
├── Machine Learning Algorithms
│   ├── AdaBoost
│   ├── Decision Tree
│   ├── Deep Neural Network
│   │   ├── Ada Embedding
│   │   ├── Noisy Ada Embedding
│   │   ├── TFIDF Embedding
│   ├── Gaussian Mixture Model
│   │   ├── Ada Embedding
│   │   ├── Word2Vec Embedding
│   │   ├── TFIDF Embedding
│   ├── Logistic Regression
│   ├── Oblique Decision Tree
│   ├── Random Forest
│   ├── XGBoost
├── Utility
```

## Usage
To be written ...