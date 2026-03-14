from preprocessor.model import Options, Prompts, small_model
from preprocessor.null_handler_tools import *

# def return_null_values(state) -> str:
#     df = state['df_info']
#     output_string = ""
#     output_string += "Context: \n\n"
#     for key in ['n_rows', 'n_cols (without target)', 'numeric_features', 'categorical_features', 'all_stats']:
#         if key in df:
#             output_string += f"{key}: {df[key]}\n"
#     output_string+="\n\nNull Values Percentages\n"
#     output_string+=str(df["null_percentages"])
#     return output_string

def update_nulls(state):
    try:
        import pickle
        with open("pickles/update.pkl", "wb") as f:
            pickle.dump(state, f) 
        # impute_columns = state["user_choice"]["column_names"]["fill_with_average"]
        # drop_columns = state["user_choice"]["column_names"]["drop_column"]
        # drop_rows = state["user_choice"]["column_names"]["drop_rows"]
        # df = pd.DataFrame(state['original_df'])
        # impute_response, df = replace_with_avg(impute_columns, df) if len(impute_columns) > 0 else []
        # col_drop_response, df = drop_column(drop_columns, df) if len(drop_columns) > 0 else []
        # row_drop_response, df = drop_all_rows(drop_rows, df) if len(drop_rows) > 0 else []
        
    except Exception as e:
        print(f"Error at {e}")   
    # return {'update': [impute_response, col_drop_response, row_drop_response],
    #         'original_df': df.to_dict(),
    #         "subgraph": 0 if reset_subgraph else state['subgraph'] + 1,
    #         }
    return {'update': [],
            }

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
        "model": small_model,
        "output_name": "null0",
        "data_demands": ['n_rows', 'n_cols (without target)', 'numeric_features', 'categorical_features', 'all_stats', 'null_percentages'],
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

            You will use markdown for good stylization and professional language in your answers.
            """,
        "schema": Prompts,
        "model": small_model,
        "output_name": "null1",
        "context_demands": {"struct": "null0"},
    }],
    "interrupt_name": ["null_columns"],
    "update": update_nulls
    }


    