from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

class InferencePipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, imputer,imputer_cols, encoders, skew_transformers, scaler, model):
        self.imputer = imputer
        self.imputer_cols = imputer_cols
        self.encoders = encoders
        self.skew_transformers = skew_transformers
        self.scaler = scaler
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X = X.copy()
        X = X[self.imputer_cols]
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)
        
        choices_dict = {}
        for method, col_path_pair in self.encoders.items():
            temp_list = []
            for col, path in col_path_pair.items():
                temp_list.append(col)
            choices_dict[method] = temp_list
        for method, cols in choices_dict.items():
            for col in cols:
                encoder = self.encoders[method][col]
                if method in ('one_hot', 'binary_encoding'):
                    encoded = encoder.transform(X[[col]])
                    encoded_df = pd.DataFrame(encoded,columns=encoder.get_feature_names_out([col]),index=X.index,)
                    X = X.drop(col, axis=1)
                    X = pd.concat([X, encoded_df], axis=1)
                else:
                    X[col] = encoder.transform(X[[col]])
        for col, transformer in self.skew_transformers.items():
            X[col] = transformer.transform(X[[col]])
        
        X = self.scaler.transform(X)
        
        return self.model.predict(X)

    def predict_proba(self, X):
        X = X.copy()
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)

        choices_dict = {}
        for method, col_path_pair in self.encoders.items():
            temp_list = []
            for col, path in col_path_pair.items():
                temp_list.append(col)
            choices_dict[method] = temp_list

        for method, cols in choices_dict.items():
            for col in cols:
                encoder = self.encoders[method][col]
                if method in ('one_hot', 'binary_encoding'):
                    encoded = encoder.transform(X[[col]])
                    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
                    X = X.drop(col, axis=1)
                    X = pd.concat([X, encoded_df], axis=1)
                else:
                    X[col] = encoder.transform(X[[col]])
        for col, transformer in self.skew_transformers.items():
            X[col] = transformer.transform(X[[col]])
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)