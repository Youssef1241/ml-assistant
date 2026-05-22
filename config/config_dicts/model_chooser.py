from config.model import Options, Prompts, FullPayload


models_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending ML models to a user based on the dataset
        Your instructions: 
            - Examine the given data and recommend 1-3 models suitable for training this dataset
            - You may only choose from the following: ["logistic_regression", "random_forest", "xgboost", "svm","lightgbm"]

            - You will fill actions attribute with only two keys: "True" and "False", True will contain your recommendations and False will contain the remaining options, copy the names exactly as given above

            - you will also provide reasoning for your actions in the reasoning key

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) asking them to choose the models, while mentioning and explaining your recommendation
            - in the beginning, create a ### heading named Models, then put your prompt

            - Use markdown for good stylization and professional language

        """,
        "schema": FullPayload,
        "output_name": "mod0",
        "data_demands": ['Class Distributions','Sample Data','n_rows', 'n_cols', 'numeric_features', 'categorical_features', 'all_stats','imbalance_ratio','data_description', 'metrics'],
    }],
    "interrupt_name": ["models"],
    }
