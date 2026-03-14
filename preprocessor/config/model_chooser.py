from preprocessor.model import Options, Prompts,small_model, FullPayload


models_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending ML models to a user based on the dataset
        Your instructions: 
            - Examine the given data and recommend 1-3 models suitable for training this dataset
            - You may only choose from the following: ["logistic_regression", "random_forest", "xgboost", "svm","lightgbm"]

            - You will fill actions attribute with only two keys: "True" and "False", True will contain your recommendations and False will contain the remaining options, copy the names exactly as given above

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) asking them to choose the models, while mentioning and explaining your recommendation

            - Use markdown for good stylization and professional language

        """,
        "schema": FullPayload,
        "model": small_model,
        "output_name": "mod0",
        "data_demands": ['Class Distributions','Sample Data','n_rows', 'n_cols (without target)', 'numeric_features', 'categorical_features', 'all_stats','imbalance_ratio','data_description', 'metrics'],
    }],
    "interrupt_name": ["models"],
    }
