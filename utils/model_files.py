import os

def create_model_state_folder(model_path: str):
    try:
        os.mkdir(model_path.split("/")[0])
    except:
        pass