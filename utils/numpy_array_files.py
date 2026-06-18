import numpy as np, os

def get_saved_arrays(dataset_train_path: str, dataset_test_path: str, total: int):
    try:
        X_train_list = []
        X_test_list = []
        y_train_list = []
        y_test_list = []
        
        for i in range(total):
            X_train_list.append(np.load(f"{dataset_train_path}_X_{i}.npy", allow_pickle=True))
            X_test_list.append(np.load(f"{dataset_test_path}_X_{i}.npy", allow_pickle=True))
            y_train_list.append(np.load(f"{dataset_train_path}_y_{i}.npy", allow_pickle=True))
            y_test_list.append(np.load(f"{dataset_test_path}_y_{i}.npy", allow_pickle=True))
            
        print("All chunks loaded. Concatenating into final arrays...")
    
        X_train = np.concatenate(X_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
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