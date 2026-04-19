from config.model import HPOptions, small_model, Prompts

hp_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending hyperparametrs for a model to a user

        hyperparameters list: 
        Logistic Regression: 
            - C: [0.01, 0.1, 1, 10, 100]
            - penalty: ["l1", "l2", "elasticnet"]
            - solver: ["lbfgs", "saga"]
            - max_iter: [100, 1000, 10000]

        Random Forest Classifier: 
            - n_estimators: [10, 100, 1000, 10000]
            - max_depth: [1, 2, 3, 4, 5]
            - min_samples_split: [2, 3, 4, 5, 6]
            - min_samples_leaf: [1, 2, 3, 4, 5]
            - max_features: ["auto", "sqrt", "log2"]
            - bootstrap: [True, False]
            - criterion: ["gini", "entropy"]

        XGBoost Classifier: 
            - n_estimators: [10, 100, 1000, 10000]
            - max_depth: [1, 2, 3, 4, 5]
            - min_samples_split: [2, 3, 4, 5, 6]
            - min_samples_leaf: [1, 2, 3, 4, 5]
            - max_features: ["auto", "sqrt", "log2"]
            - booster: ["gbtree", "gblinear"]
            - subsample: [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            - colsample_bytree: [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            - learning_rate: [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
            - gamma: [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
            - reg_alpha: [0, 0.1, 0.2, 0.3, 0.4, 0.5]
            - reg_lambda: [0, 0.1, 0.2, 0.3, 0.4, 0.5]

        SVM: 
            - C: [0.01, 0.1, 1, 10, 100]
            - kernel: ["linear", "poly", "rbf", "sigmoid"]
            - degree: [2, 3, 4, 5]
            - gamma: [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
            - coef0: [0, 0.1, 0.2, 0.3, 0.4, 0.5]
            - tol: [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
            - max_iter: [100, 1000, 10000]

        LightGBM Classifier: 
            - n_estimators: [10, 100, 1000, 10000]
            - max_depth: [1, 2, 3, 4, 5]
            - min_samples_split: [2, 3, 4, 5, 6]
            - min_samples_leaf: [1, 2, 3, 4, 5]
            - max_features: ["auto", "sqrt", "log2"]
            - booster: ["gbtree", "gblinear"]
            - subsample: [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            - colsample_bytree: [0.5, 0.6, 0.7, 0.8, 0.9, 1]
            - learning_rate: [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
            - gamma: [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
            - reg_alpha: [0, 0.1, 0.2, 0.3, 0.4, 0.5]
            - reg_lambda: [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        Your instructions: 
            - Above is the models specs and hyperparameters for 5 models, dont use it all, just use the names given in model dict in the context

            - You will recommend the hyperparameters for each model in the context

            - You will fill actions dict with one key for each model, with the value as a dict of the hyperparameters for that model

            - you will also provide reasoning for your actions in the reasoning key

        """,
        "schema": HPOptions,
        "model": small_model,
        "output_name": "hp0",
        "data_demands": ['n_rows', 'n_cols', 'categorical_stats','data_description', 'models'],
    },
    {
        "prompt": """

            Context: 

            {context}
            ---
            You are part of a team conducting a machine learning pipeline for a user, your task is to look at the context provided and return a list of prompts for the user
            Your instructions: 
            - you will generate a prompt for the user for every key in the context, (except reasoning).

            Your prompt will include each hyper parameter value, as well as the given recommendation and the reasoning behind it.

            - create only one prompt for each key.

            You will use markdown for good stylization and professional language in your answers.
            """,
        "schema": Prompts,
        "model": small_model,
        "output_name": "hp1",
        "context_demands": {'struct': 'hp0'}
    }],
    "interrupt_name": ["hp"],
    }
