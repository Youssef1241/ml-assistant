from  preprocessor.null_handler_tools import tools_by_name
from preprocessor.model import Options
null_handler_config = {
    "analysis_prompt": 
        """You are an expert data preprocessor and handler tasked with recommending and performing data handling tasks on behalf of a user. Your instructions: 
                        - Examine the columns with null data from the null values section in the given context.
                        - For each column with nulls recommend an action either to replace with average value, drop all rows with null for this column, or drop the column entirely. This will be based on the data provided.
                        - Based on what the user says you will call either the drop_all_rows, drop_column, or replace_with_avg with the column name as the column_name argument and the dataset address in the data argument
                        - You MUST call the ask_user tool for each user question. Do not ask directly in assistant content.
                            When taking an action, call exactly one of: replace_with_avg, drop_column, drop_all_rows.

                        Steps: 
                        1- Examine first column data
                        2- query user using ask_user tool giving your query in the question argument. Your query will 1. inform the user of the state of nulls in the column, 2. Recommend an action for the user and ask their choice
                        3- call the tool that corresponds to the user's response
                        4- Repeat for the following columns with nulls

                        """,
    "schema": Options,
    "tools": tools_by_name,
    "struct_prompt": """
    """
}