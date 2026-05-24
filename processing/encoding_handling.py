import os
import pickle
import logging
import pandas as pd
from imblearn.combine import SMOTETomek
from state import load_df, load_df_fromdf
from logging_utils import get_logger, log_event
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from category_encoders import BinaryEncoder, CountEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTENC
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, RobustScaler


logger = get_logger(__name__)


encoding_config = {
        "one_hot":{"encoding-class": OneHotEncoder,"params": {"sparse_output": False}},
        "ordinal_encoding":{"encoding-class": OrdinalEncoder,"params": {}},
        "binary_encoding":{"encoding-class": BinaryEncoder,"params": {}},
        "frequency_encoding":{"encoding-class": CountEncoder,"params": {}},
        "target_encoding":{"encoding-class": TargetEncoder,"params": {}},
        }
scaling_config ={
    "StandardScaler": { "scaling-class": StandardScaler, "params": {}},
    "RobustScaler": { "scaling-class": RobustScaler, "params": {}},
    "no_scaling": { "scaling-class": None, "params": {}},
}

imbalance_config = {
        "random_oversampling":RandomOverSampler,
        "random_undersampling":RandomUnderSampler,
        "SMOTE":SMOTE,
        "ADASYN":ADASYN,
        "SMOTETomek":SMOTETomek,
        "SMOTENC":SMOTENC
    }

def safe_filename(col):
    import re
    return re.sub(r'[^\w\-]', '_', col)

def handle_encoding(choices_tuple, Xtrain, Xtest, ytrain, ytest, inference_pipeline, full_data = None):
    choices_dict = choices_tuple[0]
    pickle.dump(inference_pipeline, open(f"pickles/handle_encoding.pkl", "wb"))
    log_event(logger, logging.INFO, "handle_encoding start", methods=choices_dict)
    inference_pipeline['encoders'] = {}
    for method in choices_dict:
        if encoding_config[method]['params'] == {}:
            encoder = encoding_config[method]['encoding-class']()
        else:
            # if method == "target-encoding":
            #     encoding_config[method]['params']['y'] = Xtrain[choices_tuple[1]]
            encoder = encoding_config[method]['encoding-class'](**encoding_config[method]['params'])
        inference_pipeline['encoders'][method] = {}
        for col in choices_dict[method]:
            if method == "binary_encoding" or method == "one_hot":
                encoded = encoder.fit_transform(Xtrain[[col]])
                encoded_test = encoder.transform(Xtest[[col]])
                encoded_df = pd.DataFrame(encoded,columns=encoder.get_feature_names_out([col]),index=Xtrain.index,)
                encoded_test = pd.DataFrame(encoded_test,columns=encoder.get_feature_names_out([col]),index=Xtest.index,)
                Xtrain = Xtrain.drop(col, axis=1)
                Xtest = Xtest.drop(col, axis=1)
                Xtrain = pd.concat([Xtrain, encoded_df], axis=1)
                Xtest = pd.concat([Xtest, encoded_test], axis=1)
            elif method == "target_encoding":
                encoded = encoder.fit_transform(Xtrain[[col]], ytrain)
                encoded_test = encoder.transform(Xtest[[col]])
                Xtrain[[col]] = encoded
                Xtest[[col]] = encoded_test
            else:
                Xtrain[col] = encoder.fit_transform(Xtrain[[col]])
                Xtest[col] = encoder.transform(Xtest[[col]])
            if full_data:
                path = f"pipeline/encoding/{method}/{safe_filename(col)}.pkl"
                os.makedirs(f"pipeline/encoding/{method}", exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(encoder, f)
                inference_pipeline['encoders'][method][col] = path
                log_event(logger, logging.INFO, "Saved encoder artifact", method=method, path=path)
    return Xtrain, Xtest, inference_pipeline

def handle_skew(choices_tuple, Xtrain, Xtest, ytrain, ytest, inference_pipeline, full_data=None):
    cols = choices_tuple[0]
    log_event(logger, logging.INFO, "handle_skew start", column_count=len(cols))
    inference_pipeline['skew_transformers'] = {}
    for col in cols:
        pt = PowerTransformer(method="yeo-johnson")
        Xtrain[col] = pt.fit_transform(Xtrain[[col]])
        Xtest[col] = pt.transform(Xtest[[col]])
        if full_data:
            path = f"pipeline/skew/{safe_filename(col)}.pkl"
            os.makedirs(f"pipeline/skew", exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(pt, f)
            inference_pipeline['skew_transformers'][col] = path
            log_event(logger, logging.INFO, "Saved skew transformer", path=path)
    return Xtrain, Xtest, inference_pipeline

def handle_scaling(Xtrain, Xtest, ytrain, ytest, method, df_info, inference_pipeline, slug):
    if method == "no_scaling":
        log_event(logger, logging.INFO, "Scaling skipped")
        return Xtrain, Xtest, ytrain, ytest, inference_pipeline
    scaler = scaling_config[method]["scaling-class"]()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    path = f"pipeline/scaling/{slug}.pkl"
    os.makedirs(f"pipeline/scaling", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    if 'scaler' in inference_pipeline:
        inference_pipeline['scaler'][slug] = path
    
    log_event(logger, logging.INFO, "Saved scaler artifact", method=method, path=path)

    return Xtrain, Xtest, ytrain, ytest, inference_pipeline

def handle_imbalance(Xtrain, Xtest, ytrain, ytest, method, df_info, inference_pipeline, slug):
    target = df_info['target']
    log_event(logger, logging.INFO, "handle_imbalance start", method=method, target=target)
    if method == "SMOTENC":
        categorical_features = df_info['categorical_features']["data"]
        imb = imbalance_config[method](categorical_features=categorical_features, random_state=42)
    else:
        imb = imbalance_config[method](random_state=42)

    Xresampled, yresampled = imb.fit_resample(Xtrain, ytrain) 
    return Xresampled, Xtest, yresampled, ytest, inference_pipeline

combinations_config = {
    "encoding": {
        "method": handle_encoding,
    },
    "skew": {
        "method": handle_skew,
    },
    "scaling": {
        "method": handle_scaling,
    },
    "imbalance_method": {
        "method": handle_imbalance,
    }
}



def skew_handling(state):
    """ Handles encoding then skew """
    log_event(logger, logging.INFO, "skew_handling start")
    user_choice = state['user_choice']
    sampling = user_choice.get('sampling', {}).get('sampling', False)
    df = pd.read_csv(state['df_info']['filepath'], )

    df = df.sample(frac = 0.2) if sampling == True else df.sample(frac=0.1)
    log_event(logger, logging.INFO, "Loaded preprocessing dataframe", rows=df.shape[0], cols=df.shape[1])
    col_wise = ["encoding", "skew"]
    target_col = state["df_info"]["target"]
    Xtrain, Xtest, ytrain, ytest = train_test_split(df.drop(target_col, axis=1), df[target_col], test_size=0.2, random_state=42) 
    for col_op in col_wise:
        choices_dict = user_choice[col_op][col_op]
        choices_tuple = (choices_dict, target_col)
        Xtrain, Xtest, _ = combinations_config[col_op]["method"](choices_tuple, Xtrain, Xtest, ytrain, ytest, {})
    
    state["df_info"].update(load_df_fromdf(Xtrain, state['df_info']['target']))
    data_split = (Xtrain, Xtest, ytrain, ytest)
    pickle.dump(data_split, open(f"/tmp/data_split.pkl", "wb"))
    return {
        "data_split": "/tmp/data_split.pkl",
        "df_info": state['df_info'],
    }