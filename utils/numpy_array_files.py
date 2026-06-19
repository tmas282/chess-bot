import numpy as np, os

def preprocessed_state_exists(path: str, total: int):
    for i in np.arange(total):
        file_path = f"{path}_{i}.npz"
        if not os.path.exists(file_path):
            print(f"Missing file detected: {file_path}")
            return False
                
    return True

def get_saved_arrays(path: str, index: int):
    try:
        with np.load(f'{path}_{index}.npz', mmap_mode="r") as data_comp:
            X_train = data_comp['X_train']
            y_train = data_comp['y_train']
            X_test = data_comp['X_test']
            y_test = data_comp['y_test']
        return X_train, y_train, X_test, y_test
    except Exception as e:
        exception_name = type(e).__name__
        print(f"Exception caught: {exception_name}")
        print(e)
        return None
    
def save_arrays(path: str, index: int, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    try:
        os.mkdir(f"{path.split("/")[0]}")
    except:
        pass
    finally:
        np.savez_compressed(f"{path}_{index}.npz", X_train, y_train, X_test, y_test, allow_pickle=True)