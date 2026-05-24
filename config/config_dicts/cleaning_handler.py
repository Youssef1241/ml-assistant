from config.model import FullPayload
clean_config = {
    "struct": [
        {
            "prompt": 
            """
            You are an expert data cleaner and handler tasked with recommending actions to a user based on the dataset
            Your instructions:
                - Recommend to the user whether or not any columns need to be dropped, because they are redundant or not useful
                - If any outliers need to be dropped in any columns
                - If any columns need to be cast properly due to formatting issues
                - If any sentinel values need to be replaced in any columns

            You will provide your answer in a dict named actions, with the following keys:
                - drop_columns: value: a list of columns names that need to be dropped
                - drop_outliers: value: a list of columns names that need to be dropped
                - cast_columns: value: a list of column names that need to be cast
                - replace_sentinels: value: a list of column names that need sentinel values to be replaced
            you will also provide reasoning for your actions in the reasoning key
            you will also create a prompt for the user (in the prompts attribute (just one string item in the list)) providing your recommendations, and explaining why you chose them, and ask the user whether they want to proceed
            """,
            "schema": FullPayload,
            "output_name": "cle0",
            "data_demands": ["issues", "numeric_stats", "categorical_stats"]
        }
    ],
    "interrupt_name": ["cleaning"],
}