from config.model import FullPayload

metrics_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending data handling action to a user
        Your instructions: 
            - Examine the given data and recommend 1-3 metrics suitable for evaluating this dataset
            - You may only choose from the following: ["f1", "accuracy", "roc_auc", "precision","recall"]

            - You will fill actions attribute with only two keys: "True" and "False", True will contain your recommendations and False will contain the remaining options, copy the names exactly as given above
            
            - you will also provide reasoning for your actions in the reasoning key

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) asking them to choose metrics for the model, while mentioning and explaining your recommendation
            - in the beginning, create a ### heading named Metrics, then put your prompt

            - Use markdown for good stylization and professional language

        """,
        "schema": FullPayload,
        "output_name": "met0",
        "data_demands": ['Class Distributions','Sample Data','n_rows', 'n_cols', 'numeric_features', 'categorical_features', 'all_stats','imbalance_ratio','data_description'],
    }],
    "interrupt_name": ["metrics"],
    }
