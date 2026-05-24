from config.model import HPOptions, Prompts

hp_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending hyperparameters for a model to a user

        hyperparameters list: 
        Logistic Regression:
            - C: [0.001, 0.01, 0.1, 1, 10, 100]
            - solver: ["saga"]                       
            - l1_ratio: [0.1, 0.3, 0.5, 0.7, 0.9]    
            - max_iter: [500, 1000, 2000]

        Random Forest Classifier:
            - n_estimators: [100, 300, 500, 1000]
            - max_depth: [5, 10, 20, 30]        
            - min_samples_split: [2, 5, 10]
            - min_samples_leaf: [1, 2, 4, 8]
            - max_features: ["sqrt", "log2", 0.5]      
            - bootstrap: [True, False]
            - criterion: ["gini", "entropy"]

        XGBoost Classifier:
            - n_estimators: [100, 300, 500, 1000]
            - max_depth: [3, 4, 5, 6, 7, 8]          
            - subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
            - colsample_bytree: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            - learning_rate: [0.01, 0.05, 0.1, 0.2, 0.3]
            - gamma: [0, 0.1, 0.3, 0.5, 1.0]          
            - reg_alpha: [0, 0.01, 0.1, 0.5, 1.0]    
            - reg_lambda: [0.5, 1.0, 1.5, 2.0, 5.0]    

        SVM:
            - C: [0.01, 0.1, 1, 10, 100]
            - kernel: ["linear", "poly", "rbf", "sigmoid"]
            - degree: [2, 3, 4, 5]                     
            - gamma: ["scale", "auto", 0.001, 0.01, 0.1]
            - coef0: [0.0, 0.1, 0.5, 1.0]             
            - max_iter: [-1, 1000, 5000]              

        LightGBM Classifier:
            - n_estimators: [100, 200, 300, 500, 750, 1000]
            - max_depth: [-1, 3, 5, 7, 10, 15]
            - num_leaves: [15, 31, 63, 127, 255]
            - min_child_samples: [1, 5, 10, 20, 50, 100, 200]
            - subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
            - colsample_bytree: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            - learning_rate: [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
            - reg_alpha: [0, 0.01, 0.1, 0.5, 1.0, 5.0]
            - reg_lambda: [0, 0.01, 0.1, 0.5, 1.0, 5.0]
        Your instructions: 
            - Above is the models specs and hyperparameters for 5 models, dont use it all, just use the names given in model dict in the context

            - You will recommend the hyperparameters for each model in the context

            - You will fill actions dict with one key for each model, with the value as a dict of the hyperparameters for that model
            - only provide one value for each hyperparameter

            - you will also provide reasoning for your actions in the reasoning key


        """,
        "schema": HPOptions,
        "output_name": "hp0",
        "data_demands": ['n_rows', 'n_cols', 'categorical_stats','data_description', 'models'],
    },
    {
        "prompt": """

        Context:
        {context}
        ---
        You are part of a machine learning pipeline. Given the context above, call the Prompts tool.

        Instructions:
        - Generate exactly one prompt string per model in the context.
        - Each prompt must ask the user to confirm the recommended hyperparameters for that model.
        - Each prompt must include: the model name, each hyperparameter and its value, and a brief reason.
        - Use professional language. Format each prompt in markdown.
        - Do NOT add headings, reasoning sections, or any text outside the tool call.
        - Return ONLY the tool call with the list of prompt strings.
        """,
        "schema": Prompts,
        "output_name": "hp1",
        "context_demands": {'struct': 'hp0'}
    }],
    "interrupt_name": ["hp"],
    }
