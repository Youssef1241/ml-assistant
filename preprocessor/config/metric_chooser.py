from preprocessor.model import small_model, FullPayload



# def get_data_context(state) -> str:
#     output_string = ""
    
#     df = state['df_info']
#     import pickle
#     with open("pickles/metrics.pkl", "wb") as f:
#         pickle.dump(state, f)  
    
#     per_class = {
#     cls: {
#         "count": cnt,
#         "pct": round(cnt / sum(df['class_dist'].values()) * 100,2)
#     }
#     for cls, cnt in df['class_dist'].items()
#     }
    
#     output_string += "Classes Data: \n" + str(per_class)
#     output_string += "\nSample Data: \n\n"
#     output_string += str(df['head'])
#     output_string += "\n\n"
#     output_string += "\nData description: \n\n"
#     for key in ['n_rows', 'n_cols (without target)', 'numeric_features', 'categorical_features', 'all_stats']:
#         if key in df:
#             output_string += f"{key}: {df[key]}\n"
#     output_string += "\n\n"
#     output_string += "Imbalance Ratio: " + str(df["imbalance_ratio"])
    
#     if state.get('data_description', None):
#         output_string+= "\nDataset Description" + str(state['data_description']) + '\n\n'
    
#     return output_string

metrics_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending data handling action to a user
        Your instructions: 
            - Examine the given data and recommend 1-3 metrics suitable for evaluating this dataset
            - You may only choose from the following: ["f1", "accuracy", "roc-auc", "precision","recall"]

            - You will fill actions attribute with only two keys: "True" and "False", True will contain your recommendations and False will contain the remaining options, copy the names exactly as given above
            
            - you will also provide reasoning for your actions in the reasoning key

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) asking them to choose metrics for the model, while mentioning and explaining your recommendation

            - Use markdown for good stylization and professional language

        """,
        "schema": FullPayload,
        "model": small_model,
        "output_name": "met0",
        "data_demands": ['Class Distributions','Sample Data','n_rows', 'n_cols (without target)', 'numeric_features', 'categorical_features', 'all_stats','imbalance_ratio','data_description'],
    }],
    "interrupt_name": ["metrics"],
    }


    