import numpy as np, os

def get_saved_arrays(dataset_train_path: str, dataset_test_path: str):
    try:
        X_train = np.load(f"{dataset_train_path}_X.npy")
        X_test = np.load(f"{dataset_test_path}_X.npy")
        y_train = np.load(f"{dataset_train_path}_y.npy")
        y_test = np.load(f"{dataset_test_path}_y.npy")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(e)
        return None
    
def save_arrays(dataset_train_path: str, dataset_test_path: str, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    try:
        os.mkdir(f"{dataset_train_path.split("/")[0]}")
    except:
        pass
    finally:
        np.save(f"{dataset_train_path}_X.npy", X_train)
        np.save(f"{dataset_test_path}_X.npy", X_test)
        np.save(f"{dataset_train_path}_y.npy", y_train)
        np.save(f"{dataset_test_path}_y.npy", y_test)