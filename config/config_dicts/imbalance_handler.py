import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import entropy
from config.model import FullPayload

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

imbalance_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data preprocessor and handler tasked with recommending data handling action to a user
        Your instructions: 
            - Examine the given data and recommend 1-3 methods for handling the class imbalance
            - You may only choose from the following: ["random_undersampling", "random_oversampling", "SMOTE", "ADASYN","SMOTETomek","SMOTENC]

            - You will fill actions attribute with only two keys: "True" and "False", True will contain your recommendations and False will contain the remaining options, copy the names exactly as given above

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) explaining that there was an imbalance detected and that these are the recommended ways to solve it

            - in the beginning, create a ### heading named Imbalance Handling, then put your prompt

            - Use markdown for good stylization and professional language

        """,
        "schema": FullPayload,
        "output_name": "imb0",
        "data_demands": ['imbalance_ratio','Normalized Entropy', 'Class Distributions', 'n_rows', 'n_cols', 'numeric_features', 'categorical_features', 'data_description','metrics', 'models'],
    }],
    "interrupt_name": ["imbalance_methods"],
    }


    