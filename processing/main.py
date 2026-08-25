import os
import pickle
import joblib
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn import pipeline
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from processing.null_handling import update_nulls
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from processing.encoding_handling import safe_filename
from processing.inference_pipeline import InferencePipeline
from processing.encoding_handling import combinations_config
from logging_utils import get_logger, log_event, log_values_with_types
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

logger = get_logger(__name__)

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
    log_event(logger, logging.INFO, "train_and_test start")
    inference_pipeline = state['pipeline']
    user_choice = state['user_choice']
    if user_choice.get("final", None) != None and user_choice.get("retrain",None) == None:
        full_data = True
    else:
        try:
            if  user_choice['final']['final']["retrain"] == "full-retrain":
                full_data = True
        except:
            full_data = False
    metrics = user_choice['metrics']['metrics']['True']
    metrics = {item: metrics_config[item] for item in metrics}
    if full_data:
        df = pd.read_csv(state['df_info']['filepath'], )
        model_combinations = [item.split('-') for item in state['user_choice']['final']['final']["True"]]
        spec_names = ['scaling','model','imbalance_method']
        model_combinations =  [dict(zip(spec_names,item)) for item in model_combinations]
        dfs_list, inference_pipeline = preprocess(state, user_choice, model_combinations, inference_pipeline, full_data, df)
    else:
        model_combinations = user_choice['filter']['filter']
        dfs_list, inference_pipeline = preprocess(state, user_choice, model_combinations, inference_pipeline, full_data)
        inference_pipeline = {}
    trained = Parallel(n_jobs=-1)(delayed(train)(item['model'],item['data'], item["slug"], metrics) for item in dfs_list)
    trained_metrics = [item[0] for item in trained]
    if full_data:
        os.makedirs(f"/tmp/pipeline/models", exist_ok=True)
        inference_pipeline['model'] = {}
        for item in trained:
            model = item[1]
            path = f"/tmp/pipeline/models/{safe_filename(item[0]['slug'])}.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            inference_pipeline['model'][item[0]['slug']] = path
        create_visualizations(state, df, trained_metrics)
    state['update'].update({'model_metrics': trained_metrics})
    return{
        'update': state['update'],
        "pipeline": inference_pipeline
    }
        
def preprocess(state: dict, user_choice: dict, model_combinations: list, inference_pipeline: dict, full_data: bool,  df: pd.DataFrame=None):
    """Preprocess the data and return the preprocessed data"""
    log_values_with_types(logger,logging.INFO,"preprocess start",model_combinations=model_combinations,)
    dfs_list = []
    if full_data:
        logs, df = update_nulls(state, df)
        col_wise = ["encoding", "skew"]
        target_col = state["df_info"]["target"]
        Xtrain, Xtest, ytrain, ytest = train_test_split(df.drop(target_col, axis=1), df[target_col], test_size=0.2, random_state=42)
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(Xtrain)
        imputer_cols = Xtrain.columns.tolist()
        os.makedirs("/tmp/pipeline", exist_ok=True)
        pickle.dump(imputer_cols, open('/tmp/pipeline/imputer_cols.pkl', 'wb'))
        path = "/tmp/pipeline/imputer.pkl"
        os.makedirs("/tmp/pipeline", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(imputer, f)
        inference_pipeline['imputer'] = path
        inference_pipeline['scaler'] = {}



        for col_op in col_wise:
            choices_dict = user_choice[col_op][col_op]
            choices_tuple = (choices_dict, target_col)
            Xtrain, Xtest, inference_pipeline = combinations_config[col_op]["method"](choices_tuple, Xtrain, Xtest, ytrain, ytest, inference_pipeline, full_data)
    else:
        Xtrain, Xtest, ytrain, ytest = pickle.load(open(state["data_split"], "rb"))
    for combination in model_combinations:
        model_name = combination["model"]
        model_dicts = user_choice["hp"]["hp"]
        for param in model_dicts[model_name]:
            value = model_dicts[model_name][param]
            if value == "True" or value == "False":
                model_dicts[model_name][param] = value == "True"

        model_class = models_config[model_name](**model_dicts[model_name])
        Xtrain_temp, ytrain_temp, Xtest_temp, ytest_temp = Xtrain.copy(), ytrain.copy(), Xtest.copy(), ytest.copy()
        slug = '-'.join(combination.values())
        for item in combination:
            if item != 'model':
                Xtrain_temp, Xtest_temp, ytrain_temp, ytest_temp, inference_pipeline = combinations_config[item]["method"](Xtrain_temp, Xtest_temp, ytrain_temp, ytest_temp, combination[item], state['df_info'], inference_pipeline, slug)
        dfs_list.append({"model": model_class, "data": (Xtrain_temp, Xtest_temp, ytrain_temp, ytest_temp),"slug": slug})
    return dfs_list, inference_pipeline

def train(model: Any, item: tuple, slug: str, metrics: dict):
    Xtrain, Xtest, ytrain, ytest = item
    model = model.fit(Xtrain, ytrain)
    log_event(logger, logging.INFO, "Model trained", slug=slug, train_rows=len(Xtrain), test_rows=len(Xtest))
    ypredict = model.predict(Xtest)
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric]= metrics[metric](ytest, ypredict)
    metrics_dict['confusion_matrix'] = confusion_matrix(ytest, ypredict).tolist()
    metrics_dict["slug"] = slug
    metrics_dict["ytest"] = (list(ytest), ypredict.tolist())
    return metrics_dict, model

def create_visualizations(state, df, trained):
    os.makedirs("frontend/static", exist_ok=True)
    log_event(logger, logging.INFO, "create_visualizations start", trained_models=len(trained))
    rows = []
    for item in trained:
        model = item["slug"]
        for metric, value in item.items():
            if metric == "slug" or metric == "ytest" or metric == "confusion_matrix":
                continue
            else:
                rows.append({"model": model, "metric": metric, "value": value})
    df_metrics = pd.DataFrame(rows)
    g = sns.catplot(
        data=df_metrics,x="model", y="value", hue="metric",kind="bar",
        height=6, aspect=1.5, palette="deep")
    sns.despine(left=True)
    g.savefig("frontend/static/model_metrics_barplot.png")
    plt.close("all")
    log_event(logger, logging.INFO, "Saved chart", path="frontend/static/model_metrics_barplot.png")

    slugs_list = [item["slug"] for item in trained]
    fig, ax = plt.subplots() 
    for model in slugs_list:
        trained_model = next(item for item in trained if item["slug"] == model)        
        fpr, tpr, _ = roc_curve(trained_model["ytest"][0], trained_model["ytest"][1])
        ax.plot(fpr, tpr, label=model)
    ax.legend()
    fig.savefig(f"frontend/static/roc_curve.png")
    plt.close("all")
    log_event(logger, logging.INFO, "Saved chart", path="frontend/static/roc_curve.png")
    
    metrics = list(trained[0].keys())[:-3]
    N = len(metrics)
    if N >= 3:
        values = [list(item.values())[:-3] for item in trained]
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (model, vals) in enumerate(zip(slugs_list, values)):
            vals += vals[:1]
            ax.plot(angles, vals, color=colors[i], linewidth=2, label=model)
            ax.fill(angles, vals, color=colors[i], alpha=0.1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Model Comparison', size=15, pad=20)
        plt.tight_layout()
        plt.savefig(f"frontend/static/spider_comparison.png")
        log_event(logger, logging.INFO, "Saved chart", path="frontend/static/spider_comparison.png")
        plt.close("all")

def create_pipeline(state: dict):
    inference_pipeline = state['pipeline']
    chosen_model = state['user_choice']['chooseone']['chooseone']
    classes_dict = {}
    for item, value in inference_pipeline.items():
        if item == 'imputer':
            classes_dict[item] = pickle.load(open(value, "rb"))
        elif item == 'encoders':
            classes_dict[item] = {}
            for method, cols in value.items():
                classes_dict[item][method] = {}
                for col, path in cols.items():
                    classes_dict[item][method][col] = pickle.load(open(path, "rb"))
        elif item == 'skew_transformers':
            classes_dict[item] = {}
            for col, path in value.items():
                classes_dict[item][col] = pickle.load(open(path, "rb"))
        elif item == 'scaler':
            path = value[chosen_model]
            classes_dict[item] = pickle.load(open(path, "rb"))
        elif item == 'model':
            path = value[chosen_model]
            classes_dict[item] = pickle.load(open(path, "rb"))
    classes_dict['imputer_cols'] = pickle.load(open('/tmp/pipeline/imputer_cols.pkl', 'rb'))
    pipe = InferencePipeline(**classes_dict)
    joblib.dump(pipe, '/tmp/pipeline.pkl')
    return state