from re import X
from typing import Any
import pandas as pd
import lightgbm as lgb
from sklearn.svm import SVC
from xgboost import XGBClassifier
from joblib import Parallel, delayed
from null_handling import update_nulls
from processing.config import combinations_config
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

models_config = {
"logistic_regression": LogisticRegression,
"random_forest": RandomForestClassifier,
"xgboost": XGBClassifier,
"svm": SVC,
"lightgbm": lgb.LGBMClassifier,
}

metrics_config = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "roc_auc": roc_auc_score,
    "confusion_matrix": confusion_matrix,
}


def train_and_test(state: dict):
    user_choice = state['user_choice']
    df = pd.read_csv(state['df_info']['filepath'], )
    full_data = user_choice.get('final').get('final').get('full_data')
    metrics = user_choice['metrics']['metrics']['True']
    metrics = { item: metrics_config[item] for item in metrics}
    if full_data == None or full_data == False:
        df = df.sample(frac = 0.2) if state['user_choice']['sampling']['sampling'] == True else df.sample(frac=0.1)
        model_combinations = user_choice['final']['final']['models'].split('-')
        spec_names = ['scaling','model','imbalance_method']
        model_combinations =  [dict(zip(spec_names,item)) for item in model_combinations]
    else:
        model_combinations = list(user_choice['filter']['filter'].values())

    dfs_list = preprocess(state, user_choice, model_combinations, df)
    trained = Parallel(n_jobs=-1)(delayed(train)(item['model'],item['data'], metrics) for item in dfs_list)
    if full_data:
        create_visualizations(state, df, trained)
    state['update'].update({'model_metrics': trained})
    return{
        'update': state['update']
    }

        
def train(model: Any, item: tuple, slug: str, metrics: dict):
    Xtrain, Xtest, ytrain, ytest = item
    model = model.fit(Xtrain, ytrain)
    ypredict = model.predict(Xtest)
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric]={metrics[metric](ytest, ypredict)}
    metrics_dict['confusion_matrix'] = confusion_matrix(ytest, ypredict)
    return {slug: metrics_dict}

def preprocess(state: dict, user_choice: dict, model_combinations: list, df: pd.DataFrame):
    """Preprocess the data and return the preprocessed data"""
    logs, df = update_nulls(state, df)
    dfs_list = []
    col_wise = ["encoding", "skew"]
    target_col = state["df_info"]["target"]
    Xtrain, Xtest, ytrain, ytest = train_test_split(df, df[target_col], test_size=0.2, random_state=42)
    for col_op in col_wise:
        choices_dict = user_choice[col_op][col_op]
        choices_tuple = (choices_dict, target_col)
        Xtrain, Xtest = combinations_config[col_op](choices_tuple, Xtrain, Xtest)

    for combination in model_combinations:
        model_name = combination["model"]
        model_dicts = {};
        model_class = models_config[model_name](**model_dicts[model_name])
        Xtrain_temp, ytrain_temp, Xtest_temp, ytest_temp = Xtrain.copy(), ytrain.copy(), Xtest.copy(), ytest.copy()
        for item in combination:
            slug = '-'.join(item.values())
            if item != 'model':
                Xtrain_temp, Xtest_temp, ytrain_temp, ytest_temp = combinations_config[item](Xtrain_temp, Xtest_temp, ytrain_temp, ytest_temp, combination[item], state['df_info'])
        dfs_list.append({"model": model_class, "data": (Xtrain_temp, Xtest_temp, ytrain_temp, ytest_temp),"slug": slug})

def create_visualizations(state, df, trained):
    pass