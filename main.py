import argparse


def main():
    parser = argparse.ArgumentParser(description="Run the ML algorithms on the dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to use",
        choices=["Formatted", "Unformatted"],
        default="Unformatted",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="The embeddings to use",
        choices=["TFIDF", "Word2Vec", "Ada"],
        default="TFIDF",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        help="The algorithm to run",
        choices=["bayes", "DNN", "DT", "features", "GMM", "LR", "OPCT", "RF", "XGB"],
        default="DT",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The seed to use",
        default=42
    )

    args = parser.parse_args()
    if args.algorithm == "features":
        from ML_Algorithms.Feature_Based.feature_algorithms import run

        run(dataset=args.dataset, seed=args.seed)
    elif args.algorithm == "bayes":
        from Bayes_Classifier.bayes_class import run

        run(dataset=args.dataset, seed=args.seed)
    if args.embedding == "Ada":
        if args.algorithm == "DNN":
            from ML_Algorithms.Deep_Neural_Network.DNN_Ada import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "RF":
            from ML_Algorithms.Random_Forest.RF_Ada import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "LR":
            from ML_Algorithms.Logistic_Regression.LR_Ada import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "GB":
            from ML_Algorithms.Gradient_Boosting.GB_Ada import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "DT":
            from ML_Algorithms.Decision_Tree.DT_Ada import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "XGB":
            from ML_Algorithms.eXtreme_Gradient_Boosting.XGB_Ada import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "OPCT":
            from ML_Algorithms.Oblique_Predictive_Clustering_Tree.OPCT_Ada import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "GMM":
            from ML_Algorithms.Gaussian_Mixture_Model.GMM_Ada import run

            run(dataset=args.dataset, seed=args.seed)
    elif args.embedding == "TFIDF":
        if args.algorithm == "DNN":
            from ML_Algorithms.Deep_Neural_Network.DNN_TFIDF import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "RF":
            from ML_Algorithms.Random_Forest.RF_TFIDF import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "LR":
            from ML_Algorithms.Logistic_Regression.LR_TFIDF import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "GB":
            from ML_Algorithms.Gradient_Boosting.GB_TFIDF import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "DT":
            from ML_Algorithms.Decision_Tree.DT_TFIDF import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "XGB":
            from ML_Algorithms.eXtreme_Gradient_Boosting.XGB_TFIDF import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "OPCT":
            from ML_Algorithms.Oblique_Predictive_Clustering_Tree.OPCT_TFIDF import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "GMM":
            from ML_Algorithms.Gaussian_Mixture_Model.GMM_TFIDF import run

            run(dataset=args.dataset, seed=args.seed)
    elif args.embedding == "Word2Vec":
        if args.algorithm == "DNN":
            from ML_Algorithms.Deep_Neural_Network.DNN_Word2Vec import run

            run(dataset=args.dataset, seed=args.seed)
        elif args.algorithm == "GMM":
            from ML_Algorithms.Gaussian_Mixture_Model.GMM_Word2Vec import run

            run(dataset=args.dataset, seed=args.seed)


if __name__ == "__main__":
    main()
