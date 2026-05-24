from config.model import Options, Prompts

encoding_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending categorical encoding actions to a user
        Your instructions: 
            - Examine the given data and recommend the suitable categorical encoding method for each categorical column in this dataset
            - For each categorical column, recommend one of the following methods: ["one_hot", "ordinal_encoding", "binary_encoding",
            "frequency_encoding", "target_encoding"]

            - You will fill actions attribute with 5 keys, one for each encoding method, copy the names exactly as given above. For each key add a list of the column names that you recommend this method for. do not add one column name more than once in more than one key.
            - ALL CATEGORICAL COLUMNS MUST BE RECOMMENDED FOR ENCODING, IF A CATEGORICAL COLUMN IS NOT IN YOUR ANSWER, YOU WILL RECEIVE A PENALTY

            - you will also provide reasoning for your actions in the reasoning key
            - do not include the target column in your answer

        """,
        "schema": Options,
        "output_name": "enc0",
        "data_demands": ['n_rows', 'n_cols', 'categorical_stats','data_description', 'models', "target_column"],
    },
    {
        "prompt": """

            Context: 

            {context}
            ---
            You are part of a team conducting a machine learning pipeline for a user, your task is to look at the context provided and return a list of prompts for the user
            Your instructions: 
            - you will generate a prompt for the user for every key in the context, (except reasoning).

            Your prompt will include the column names, the possible actions for the user (one_hot, ordinal_encoding, binary_encoding, frequency_encoding, target_encoding), as well as the given recommendation and the reasoning behind it.

            in the given context the key is the recommendation, there are five possible recommendations: [one_hot, ordinal_encoding, binary_encoding, frequency_encoding, target_encoding], you will return one prompt for each key. if a key doesnt have a recommendation just put "None" as the value.

            make one prompt for each encoding method, NOT for each column

            - in the beginning of the prompt create a ### heading with the encoding method name (not the name exactly like One Hot Encoding instead of one_hot), then mention the columns for that action, and recommendations, mention your reasoning as well but no need to label it Reasoning, also dont add --- in the beginning, and dont add any other # headings in the rest of the prompt  

            You will use markdown for good stylization and professional language in your answers.
            """,
        "schema": Prompts,
        "output_name": "enc1",
        "context_demands": {'struct': 'enc0'}
    }],
    "interrupt_name": ["encoding"],
    }
