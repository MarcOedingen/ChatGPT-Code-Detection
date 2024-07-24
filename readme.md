# ChatGPT Code Detection: Techniques for Uncovering The Source of Code

## Introduction

This repository contains all experiments from the paper [Oedingen2024] and allows for the reproduction of the results. Additionally, we provide the data used for the experiments, allowing for further research of detecting AI-generated code.

[Oedingen2024]: Oedingen, M., Engelhardt, R. C., Denz, R., Hammer, M., & Konen, W. (2024). ChatGPT Code Detection: Techniques for Uncovering the Source of Code. arXiv preprint arXiv:2405.15512.
https://arxiv.org/abs/2405.15512

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

### Dataset Description

The dataset consists of four relevant columns:

- `id`: The unique identifier of the code snippet, which is a combination of the source and the task id of the code snippet from its original source.
- `source`: The source of the code snippet.
- `code`: The Python code snippet.
- `label`: The label of the code snippet, which is `1` if the code snippet is GPT-generated and `0` if the code snippet is human-written.
- `embedding`: The 'Ada' embedding of the code snippet.

In total, the dataset contains 40158 code snippets with 20079 GPT-generated and 20079 human-written code snippets.
The dataset is balanced, meaning that the number of GPT-generated code snippets and human-written code snippets is equal for each `id`.
In our paper, we used a more strict filtering of duplicates of the code snippets, which resulted in a dataset of 31448 code snippets.

Download the dataset without embeddings from [here](https://th-koeln.sciebo.de/s/XZRR45yzO0rRuj3) and with embeddings from [here](https://th-koeln.sciebo.de/s/5kh6qOhEcO5ueFV).
Unzip the downloaded file and place it in the `Dataset` folder to use it for the experiments.

## Structure of the Project

```
├── Bayes_Classifier
│   ├── bayes_class.py
├── ML_Algorithms
│   ├── Decision_Tree
│   │   ├── DT_Ada.py
│   │   ├── DT_TFIDF.py
│   ├── Deep_Neural_Network
│   │   ├── DNN_Ada.py
│   │   ├── DNN_TFIDF.py
│   │   ├── DNN_Word2Vec.py
│   ├── Feature_Based
│   │   ├── feature_algorithms.py
│   │   ├── feature_extractor.py
│   ├── Gaussian Mixture Model
│   │   ├── GMM_Ada.py
│   │   ├── GMM_TFIDF.py
│   │   ├── GMM_Word2Vec.py
│   ├── Logistic_Regression
│   │   ├── LR_Ada.py
│   │   ├── LR_TFIDF.py
│   ├── Oblique_Predictive_Clustering_Tree
│   │   ├── OPCT_Ada.py
│   │   ├── OPCT_TFIDF.py
│   ├── Random_Forest
│   │   ├── RF_Ada.py
│   │   ├── RF_TFIDF.py
│   ├── eXtreme_Gradient_Boosting
│   │   ├── XGB_Ada.py
│   │   ├── XGB_TFIDF.py
├── Datasets
│   ├── Unformatted_Balanced.jsonl
│   ├── Unformatted_Balanced_Embedded.jsonl
├── Utility
│   ├── utils.py
├── Models
│   ├── DNN_Ada.pkl
│   ├── XGB_TFIDF.pkl
│   ├── Vectorizer_TFIDF.pkl
├── Results
├── main.py
├── requirements.txt
├── README.md
```

## Usage

To run the experiments, you need to install the required packages. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

After installing the required packages, you can run the experiments by executing the `main.py` file:

```bash
python main.py --dataset <dataset> --embedding <embedding> --algorithm <algorithm> --seed <seed>
```

Parameters:

- `dataset`: The dataset to use for the experiments. Possible values are `Formatted` or `Unformatted`.
- `embedding`: The embedding to use for the experiments. Possible values are `Ada`, `TFIDF`, or `Word2Vec`.
- `algorithm`: The algorithm to use for the experiments. Possible values are `DT`, `DNN`, `features`, `GMM`, `LR`, `OPCT`, `RF`, or `XGB`.
- `seed`: The seed to use for the experiments, i.e, the distribution of problems in the training and test set.

## Application

### Getting Started

1. Install [Cog](https://github.com/replicate/cog) and [Docker](https://www.docker.com/)
2. Build cog docker container

```bash
cd app
cog build -t xgb-tfidf-model
```

3. Run docker-compose

```bash
docker-compose up
```

4. Go to `http://localhost:3000` to see UI
5. Go to `http://localhost:5002/docs` to see the Swagger UI
6. Curl predictions endpoint directly

```bash
curl http://localhost:5002/predictions -X POST \
--header "Content-Type: application/json" \
--data '{"input": {"code": "hello"}}'
```

### Tech Stack

- Python 3.8
- [Cog](https://github.com/replicate/cog) for building model prediction API container
- Docker
- SvelteKit

## Reference

If you use this code or data, please cite the following paper:

```

@misc{oedingen2024chatgpt,
title={ChatGPT Code Detection: Techniques for Uncovering the Source of Code},
author={Marc Oedingen and Raphael C. Engelhardt and Robin Denz and Maximilian Hammer and Wolfgang Konen},
year={2024},
eprint={2405.15512},
archivePrefix={arXiv},
primaryClass={cs.LG}
}

```

```

```
