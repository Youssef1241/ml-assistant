from config.model import FullPayload


skew_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending actions to a user based on the dataset
        Your instructions: 
            - Examine the given data and recommend whether or not the columns are skewed and need to be fixed

            - You will fill actions attribute with only one key: "skewed", which will contain a list of your recommendations (a list of strings of column names that are skewed)
            - If no columns need to be fixed for skew, create same "skewed" key with an empty list []

            - you will also provide reasoning for your actions in the reasoning key

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) asking them to choose the columns they would like to process for skewing, while mentioning and explaining your recommendation

            - Use markdown for good stylization and professional language
            only yeo-johnson transformation will be used, dont mention other kinds

        """,
        "schema": FullPayload,
        "output_name": "ske0",
        "data_demands": ['n_rows', 'n_cols',  'numeric_stats', 'metrics', 'models'],
    }],
    "interrupt_name": ["skew"],
    }
