import numpy as np, os

def preprocessed_state_exists(dataset_train_path: str, dataset_test_path: str, total: int):
    for i in np.arange(total):
        expected_files = [
            f"{dataset_train_path}_X_{i}.npy",
            f"{dataset_test_path}_X_{i}.npy",
            f"{dataset_train_path}_y_{i}.npy",
            f"{dataset_test_path}_y_{i}.npy"
        ]
        
        for file_path in expected_files:
            if not os.path.exists(file_path):
                print(f"Missing file detected: {file_path}")
                return False
                
    return True

def get_saved_array(path: str, index: int):
    try:
        X = np.load(f"{path}_X_{index}.npy", allow_pickle=True)
        y = np.load(f"{path}_y_{index}.npy", allow_pickle=True)
        return X, y
    except Exception as e:
        exception_name = type(e).__name__
        print(f"Exception caught: {exception_name}")
        print(e)
        return None
    
def save_sub_array(dataset_train_path: str, dataset_test_path: str, index: int, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    try:
        os.mkdir(f"{dataset_train_path.split("/")[0]}")
    except:
        pass
    finally:
        np.save(f"{dataset_train_path}_X_{index}.npy", X_train, allow_pickle=True)
        np.save(f"{dataset_test_path}_X_{index}.npy", X_test, allow_pickle=True)
        np.save(f"{dataset_train_path}_y_{index}.npy", y_train, allow_pickle=True)
        np.save(f"{dataset_test_path}_y_{index}.npy", y_test, allow_pickle=True)