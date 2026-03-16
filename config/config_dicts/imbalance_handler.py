import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import entropy
# from collections import Counter
# from imblearn.combine import SMOTETomek
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from config.model import Options, Prompts,small_model, FullPayload

load_dotenv()

def calculate_imbalance_data( total, counts):
    n_classes = len(counts)
    ir_threshold = 6
    entropy_threshold = 0.95
    class_counts = np.array(list(counts.values))
    ir = float(round(class_counts.max() / class_counts.min(), 4))
    probs = class_counts / total
    norm_entropy = float(round(entropy(probs) / np.log(n_classes), 4))
    is_imbalanced = bool(ir > ir_threshold or norm_entropy < entropy_threshold)
    return (ir, norm_entropy, is_imbalanced)
  
# def imbalance_data(state) -> str:
#     output_string = ""
#     data_path = os.environ.get("DATA_PATH")
#     df = pd.read_csv(data_path)
#     assumed_target = df['Survived']
#     ir, norm_entropy, counts, total, _ = calculate_imbalance_data(assumed_target)

#     per_class = {
#     cls: {
#         "count": cnt,
#         "pct": cnt / total * 100
#     }
#     for cls, cnt in counts.items()
#     }
#     info_dict = {
#         "imbalance_ratio": ir,
#         "normalized_entropy": norm_entropy,
#         "per_class": per_class
#     }
#     output_string += "Classes Data: \n" + str(info_dict)
#     output_string += "\nSample Data: \n\n"
#     output_string += df.head().to_string()
#     output_string += "\nData description: \n\n"
#     output_string += df.describe().to_string()
#     import io
#     buffer = io.StringIO()
#     df.info(buf=buffer)
#     info_str = buffer.getvalue()
#     output_string += "Dataset info: \n\n" + info_str

#     output_string+="\n\nNull Values\n\n"
#     output_string+=df.isnull().sum().to_string()

#     return output_string

def update_imbalance(state):
    # methods_list = state["user_choice"]["imbalance_methods"]
    # methods_dict = get_methods()
    # for item in methods_list:
    pass
        
    

# def get_methods():
#     return{
#         "random_oversampling": RandomOverSampler(random_state=42),
#         "random_undersampling": RandomUnderSampler(random_state=42),
#         "SMOTE":SMOTE(random_state=42) ,
#         "ADASYN":ADASYN(random_state=42) ,
#         "SMOTETomek":SMOTETomek(random_state=42) ,
#     }

imbalance_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data preprocessor and handler tasked with recommending data handling action to a user
        Your instructions: 
            - Examine the given data and recommend 1-3 methods for handling the class imbalance
            - You may only choose from the following: ["random_undersampling", "random_oversampling", "SMOTE", "ADASYN","SMOTETomek"]

            - You will fill actions attribute with only two keys: "True" and "False", True will contain your recommendations and False will contain the remaining options, copy the names exactly as given above

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) explaining that there was an imbalance detected and that these are the recommended ways to solve it

            - Use markdown for good stylization and professional language

        """,
        "schema": FullPayload,
        "model": small_model,
        "output_name": "imb0",
        "data_demands": ['imbalance_ratio','Normalized Entropy', 'Class Distributions', 'n_rows', 'n_cols (without target)', 'numeric_features', 'categorical_features', 'data_description','metrics', 'models'],
    }],
    "interrupt_name": ["imbalance_methods"],
    }


    