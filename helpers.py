def use_persistent(model, path, messages):
    import os
    import pickle
    pkls_folder = "pickles"
    os.makedirs(pkls_folder, exist_ok=True)
    pkls_path = os.path.join(pkls_folder, path)
    if os.path.exists(pkls_path):
        with open(pkls_path, "rb") as f:
            result = pickle.load(f)
    else:
        result = model.invoke(messages)
        with open(pkls_path, "wb") as f:
            pickle.dump(result, f)
    return result
