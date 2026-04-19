from config.model import small_model, FullPayload
output_name = "fil0"
def create_models_dict(state: dict):
    """Create a dictionary of possible models and their preprocessing methods"""
    from itertools import product
    user_choice = state["user_choice"]
    struct = state["struct"]
    models_list = list()
    options_list = ["models", "scaling", "imbalance_methods"]
    names_list = ["model", "scaling-method", "imbalance-method"]
    scaling_models = user_choice["scaling"]["scaling"]
    imbalance_methods = user_choice['imbalance_methods']["imbalance_methods"]["True"]
    for method in scaling_models:
        for model in scaling_models[method]:
            for imb_method in imbalance_methods:
                models_list.append([method, model,imb_method])
    spec_names = ['scaling','model','imbalance_method']
    models_dict =  [dict(zip(spec_names,item)) for item in models_list ]
    struct[output_name] = models_dict
    return {"struct": struct}

filter_config = {
    "struct": 
    [{
        "prompt": """
        You are an expert data scientist and handler tasked with recommending actions to a user based on the dataset
        Your instructions: 
            - Examine the given model and preprocessing mixes, and choose only 6 for training

            - You must choose the ones that will most likely improve the model's performance according to the metrics given above

            - You will provide in the actions dict the keys as model1, model2, and so on, the values will be dicts given exactly the same as the context

            - you will also provide reasoning for your actions in the reasoning key

            - You will also create a prompt for the user (in the prompts attribute (just one string item in the list)) showing them your recommendations, and explaining why you chose them and ask the user whether they would like to proceed or choose other model mixes instead.

            - Use markdown for good stylization and professional language

        """,
        "schema": FullPayload,
        "model": small_model,
        "output_name": "fil1",
        "data_demands": ['n_rows', 'n_cols',  'Class Distributions', 'numeric_features','categorical_features','imbalance_ratio', 'metrics'],
        "context_demands": {'struct': 'fil0'}
    }],
    "interrupt_name": ["filter"],
    "update": {"method": create_models_dict, "output_name": output_name}
    }
