from config.model import Options, Prompts
null_handler_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data preprocessor and handler tasked with recommending and performing data handling tasks on behalf of a user. Your instructions: 
            - Examine the columns with null data from the given context.
            - For each column with nulls recommend an action either to
                - drop_column: drop the entire column from the dataset
                - drop_rows: drop only the null rows in the dataset
                - fill_with_average: fill the null values with average
            
            - Only create 3 keys in actions attributes. fill the values for the keys (drop_column, drop_rows, fill_with_average) with the appropriate column names in a list
            - if there are no columns for an action just provide and empty list [] 
            - you will also provide reasoning for your actions in the reasoning key
            - Make sure to choose only numeric columns for replacing with average
        

            Steps: 
            1- Examine all column data
            2- provide reasoning behind your action
            3- group the columns by their recommendations
        """,
        "schema": Options,
        "output_name": "null0",
        "data_demands": ['n_rows', 'n_cols', 'numeric_features', 'categorical_features', 'all_stats', 'null_percentages'],
    },
    {
        "prompt": 
            """
            Context: 

            {context}
            ---

            You are part of a team conducting a machine learning pipeline for a user, your task is to look at the context provided and return a list of prompts for the user

            you will generate a prompt for the user for every key in the context, (except reasoning).
            Your prompt will include the column names, the possible actions for the user (drop column, drop null rows, replace with average), as well as the given recommendation and the reasoning behind it.

            in the given context the key is the recommendation, there are three possible recommendations: drop the entire column, drop only the null rows, replace the null with average value, you will return one prompt for each key, that is a list of three prompts
            in the beginning of the prompt create a ### heading named Null Handling, then mention the actions, and recommendations, mention your reasoning as well but no need to label it Reasoning, also dont add --- in the beginning, and dont add any other # headings in the rest of the prompt

            You will use markdown for good stylization and professional language in your answers.
            make the list in the same order as the given dict, first the drop column and its elements in the same order, then the drop null rows, then the replace with average.
            """,
        "schema": Prompts,
        "output_name": "null1",
        "context_demands": {"struct": "null0"},
    }],
    "interrupt_name": ["null_columns"],
    }


    