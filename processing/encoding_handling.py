from re import X
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from category_encoders import BinaryEncoder, CountEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, RobustScaler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTENC

encoding_config = {
        "one-hot":{"encoding-class": OneHotEncoder,"params": {"sparse_output": False}},
        "ordinal-encoding":{"encoding-class": OrdinalEncoder,"params": {}},
        "binary-encoding":{"encoding-class": BinaryEncoder,"params": {}},
        "frequency-encoding":{"encoding-class": CountEncoder,"params": {}},
        "target-encoding":{"encoding-class": TargetEncoder,"params": {"y": None}},
        }
scaling_config ={
    "StandardScaler": { "scaling-class": StandardScaler, "params": {}},
    "RobustScaler": { "scaling-class": RobustScaler, "params": {}},
    "no-scaling": { "scaling-class": None, "params": {}},
}

imbalance_config = {
        "random_oversampling":RandomOverSampler,
        "random_undersampling":RandomUnderSampler,
        "SMOTE":SMOTE,
        "ADASYN":ADASYN,
        "SMOTETomek":SMOTETomek,
        "SMOTENC":SMOTENC
    }

def handle_encoding(choices_tuple, Xtrain, Xtest):
    choices_dict = choices_tuple[0]
    for method in choices_dict:
        if encoding_config[method]['params'] == {}:
            encoder = encoding_config[method]['encoding-class']()
        else:
            if method == "target-encoding":
                encoding_config[method]['params']['y'] = Xtrain[choices_tuple[1]]
            encoder = encoding_config[method]['encoding-class'](**encoding_config[method]['params'])
        for col in choices_dict[method]:
            if method == "binary-encoding" or method == "one-hot":
                encoded = encoder.fit_transform(Xtrain[[col]])
                encoded_test = encoder.transform(Xtest[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
                encoded_test = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out([col]))
                Xtrain = Xtrain.drop(col, axis=1)
                Xtest = Xtest.drop(col, axis=1)
                Xtrain = pd.concat([Xtrain, encoded_df], axis=1)
                Xtest = pd.concat([Xtest, encoded_test], axis=1)
            else:
                Xtrain[col] = encoder.fit_transform(Xtrain[[col]])
                Xtest[col] = encoder.transform(Xtest[[col]])
    return Xtrain, Xtest

def handle_skew(choices_tuple, Xtrain, Xtest):
    choices_dict = choices_tuple[0]
    cols = choices_dict["skewed"]
    for col in cols:
        pt = PowerTransformer(method="yeo-johnson")
        Xtrain[col] = pt.fit_transform(Xtrain[[col]])
        Xtest[col] = pt.transform(Xtest[[col]])
    return Xtrain, Xtest

def handle_scaling(Xtrain, Xtest, ytrain, ytest, method, df_info):
    if method == "no-scaling":
        return Xtrain, Xtest, ytrain, ytest
    scaler = scaling_config[method]["scaling-class"]()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest, ytrain, ytest

def handle_imbalance(Xtrain, Xtest, ytrain, ytest, method, df_info):
    target = df_info['target']
    if method == "SMOTENC":
        categorical_features = df_info['categorical_features']
        imb = imbalance_config[method]["scaling-class"](categorical_features=categorical_features, random_state=42)
    else:
        imb = imbalance_config[method]["scaling-class"](random_state=42)

    Xresampled, yresampled = imb.fit_resample(Xtrain, ytrain) 
    return Xresampled, Xtest, yresampled, ytest