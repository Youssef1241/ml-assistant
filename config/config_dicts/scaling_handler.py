from config.model import FullPayload


scaling_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending scaling actions to a user based on the dataset
        Your instructions: 
            - Examine the given data and recommend a scaling technique for the dataset for each model

            - You must choose between: ["StandardScaler", "RobustScaler"] or no scaling at all

            - You will fill actions attribute with three keys: one for each scaling method and one for no scaling "no_scaling", the value of the key will be a list of the models you recommend to scale this with, models have been given in the context above. only fill each model in one method, do not fill the same model twice

            - Copy the names exactly in the dict

            - you will also provide reasoning for your actions in the reasoning key

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) asking them to choose a scaling method for each model, while mentioning and explaining your recommendation

            - in the beginning, create a ### heading named Scaling, then put your prompt, also dont mention you are an expert data scientist

            - Use markdown for good stylization and professional language


        """,
        "schema": FullPayload,
        "output_name": "sca0",
        "data_demands": ['n_rows', 'n_cols',  'numeric_stats', 'models'],
    }],
    "interrupt_name": ["scaling"],
    }
