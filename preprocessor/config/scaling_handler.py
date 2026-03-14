from preprocessor.model import small_model, FullPayload


skew_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending scaling actions to a user based on the dataset
        Your instructions: 
            - Examine the given data and recommend a scaling technique for the dataset

            - You will fill actions attribute with two keys: True and False, True will contain your recommendation (a list of one string of the method), and False will contain the other option

            - You must choose between: ["StandardScaler", "RobustScaler"] or no scaling at all

            - Copy the name exactly in the dict, if you recommend no scaling just provide and empty list

            - you will also provide reasoning for your actions in the reasoning key

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) asking them to choose the columns they would like to process for skewing, while mentioning and explaining your recommendation

            - Use markdown for good stylization and professional language

        """,
        "schema": FullPayload,
        "model": small_model,
        "output_name": "sca0",
        "data_demands": ['n_rows', 'n_cols (without target)',  'numeric_stats', 'models'],
    }],
    "interrupt_name": ["scaling"],
    }
