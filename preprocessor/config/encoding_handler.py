from preprocessor.model import Options, small_model, Prompts

encoding_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending categorical encoding actions to a user
        Your instructions: 
            - Examine the given data and recommend the suitable categorical encoding method for each categorical column in this dataset
            - For each categorical column, recommend one of the following methods: ["one-hot", "ordinal-encoding", "binary-encoding",
            "frequency-encoding", "target-encoding"]

            - You will fill actions attribute with 5 keys, one for each encoding method, copy the names exactly as given above. For each key add a list of the column names that you recommend this method for.

            - you will also provide reasoning for your actions in the reasoning key

        """,
        "schema": Options,
        "model": small_model,
        "output_name": "enc0",
        "data_demands": ['n_rows', 'n_cols (without target)', 'categorical_stats','data_description', 'models'],
        "context_demands": {'struct': 'mod0'}
    },
    {
        "prompt": """

            Context: 

            {context}
            ---
            You are part of a team conducting a machine learning pipeline for a user, your task is to look at the context provided and return a list of prompts for the user
            Your instructions: 
            - you will generate a prompt for the user for every key in the context, (except reasoning).

            Your prompt will include the column names, the possible actions for the user (one-hot, ordinal-encoding, binary-encoding, frequency-encoding, target-encoding), as well as the given recommendation and the reasoning behind it.

            in the given context the key is the recommendation, there are five possible recommendations: [one-hot, ordinal-encoding, binary-encoding, frequency-encoding, target-encoding], you will return one prompt for each key, that is a list of three prompts

            You will use markdown for good stylization and professional language in your answers.
            """,
        "schema": Prompts,
        "model": small_model,
        "output_name": "enc1",
        "context_demands": {'struct': 'enc0'}
    }],
    "interrupt_name": ["encoding"],
    }
