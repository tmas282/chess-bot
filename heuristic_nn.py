import kagglehub
import numpy as np
import polars
import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.optim.lr_scheduler as lr_schedulers
from torch.utils.tensorboard.writer import SummaryWriter #tensorboard --logdir=runs
from sklearn.model_selection import train_test_split

from utils.fen_to_array import fen_to_array
from utils.log_loss_csv import write_to_file
from utils.model_files import create_model_state_folder
from utils.numpy_array_files import get_saved_array, preprocessed_state_exists, save_sub_array

class ChessHeuristicDataset(Dataset):
    def __init__(self, features, targets):
        self.X = features
        self.y = targets
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        x = torch.tensor(fen_to_array(self.X[idx]))
        y = torch.tensor(self.y[idx])
        return x, y
    
class ChessHeuristicEvaluator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=14,out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=(14+32),out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=(14+64),out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
        )
        self.heuristic = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128 * 8**2 + 14 * 8**2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
    def forward(self, x):
        res_conv1 = self.conv1(x)
        res_conv2 = self.conv2(torch.concatenate((res_conv1, x), dim=1 ))
        res_conv3 = self.conv3(torch.concatenate((res_conv2, x), dim=1 ))
        logits = self.heuristic(torch.concatenate((res_conv3, x), dim=1 )) #dim=1 : channels
        return logits

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessHeuristicEvaluator().to(DEVICE)
EPOCHS = 50
CHUNKS = 10
INITIAL_LEARNING_RATE = 0.001
BATCH_SIZE = 256
OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=0.001)
SCHEDULER = lr_schedulers.StepLR(OPTIMIZER, step_size=10, gamma=0.5)
LOSS_FN = torch.nn.MSELoss()
MODEL_PATH = "model_states/chess_heuristic_evaluator"
DATASET_TRAIN_PATH = "preprocessed_data/chess_heuristic_evaluator_train_dataset"
DATASET_TEST_PATH = "preprocessed_data/chess_heuristic_evaluator_test_dataset"

def pre_process_df(df: polars.DataFrame):
    print("Normalizing Final Evaluation")
    df = df.with_columns(
        Evaluation=polars.when(polars.col("mate").is_null())
            .then(polars.col("cp").cast(polars.Float32).map_batches(lambda x: np.tanh(x/200)))
            .otherwise(polars.col("mate").cast(polars.Float32).sign())

    )
    y = df.select(polars.col("Evaluation")).to_numpy()
    X = df.select(polars.col("fen")).to_numpy().reshape(-1, 1)

    X_train, X_test, y_train, y_test = normalize_split_data(X, y)

    return X_train, X_test, y_train, y_test

def normalize_split_data(X: np.ndarray, y: np.ndarray):
    print("Splitting testing data and training data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def pre_process_data():
    if preprocessed_state_exists(DATASET_TRAIN_PATH, DATASET_TEST_PATH, 10) == False:
        print("Getting dataset")
        abs_path = kagglehub.dataset_download("mateuszgrzybpl/lichess-chess-positions-ml-ready-and-deduplicated")
        for i in range(CHUNKS):
            path = os.path.join(abs_path, f"train-{i:05d}.parquet")
            print(f"Reading dataset {i}")
            df = polars.read_parquet(path)
            print("Preprocessing dataset")
            arrs = pre_process_df(df=df)
            save_sub_array(DATASET_TRAIN_PATH, DATASET_TEST_PATH, i, arrs[0], arrs[1], arrs[2], arrs[3])
            del arrs
    else:
        print("Preprocessing skipped, using saved normalization")

def create_dataset(X, y):
    print("Creating Dataset")
    ds = ChessHeuristicDataset(X, y)
    return ds

def create_dataloader(dataset, shuffle=False):
    print("Creating Dataloader")
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=True, num_workers=4)
    return dl

def train_one_epoch(train_dl: "DataLoader"):
    running_loss = 0.
    last_loss = 0.
    avg_loss = 0.0
    n_avg = 0
    for i, data in enumerate(train_dl):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        OPTIMIZER.zero_grad()
        outputs = model(inputs)
        loss = LOSS_FN(outputs, labels)
        loss.backward()
        OPTIMIZER.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            avg_loss = avg_loss + last_loss
            n_avg = n_avg + 1
            print(f"Lr {OPTIMIZER.param_groups[0]['lr']}  batch {i + 1} loss: {last_loss}")
            running_loss = 0.

    return avg_loss / n_avg
 
def train():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 1_000_000.
    print("Starting Training")
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}:')
        avg_tloss = 0.0
        avg_vloss = 0.0
        for j in range(CHUNKS):
            print(f'CHUNK {j + 1}:')
            X, y = get_saved_array(DATASET_TRAIN_PATH, j)
            ds = create_dataset(X, y)
            train_dl = create_dataloader(ds, shuffle=True)
            model.train(True)
            avg_tloss += train_one_epoch(train_dl)
            running_vloss = 0.0
            X, y = get_saved_array(DATASET_TEST_PATH, j)
            ds = create_dataset(X, y)
            test_dl = create_dataloader(ds, shuffle=False)
            model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(test_dl):
                    vinputs, vlabels = vdata
                    vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)
                    voutputs = model(vinputs)
                    vloss = LOSS_FN(voutputs, vlabels)
                    running_vloss += vloss
            avg_vloss = running_vloss / (i + 1)
            print(f'Avg validation {avg_vloss}')
        print(f'LOSS: avg train {avg_tloss / CHUNKS} && avg validation {avg_vloss / CHUNKS}')
        SCHEDULER.step()
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{MODEL_PATH}_{timestamp}_{epoch}'
            torch.save(model.state_dict(), model_path)
        write_to_file(epoch, avg_tloss, avg_vloss)
    print("Ended Training")

def start_training():
    print(f"DEVICE={DEVICE}\nINITIAL_LEARNING_RATE={INITIAL_LEARNING_RATE}\nEPOCHS={EPOCHS}\nBATCH_SIZE={BATCH_SIZE}")
    create_model_state_folder(MODEL_PATH)
    pre_process_data()
    train()

if __name__ == "__main__":
    start_training()

def use_model():
    create_model_state_folder()
    model = ChessHeuristicEvaluator()
    model.load_state_dict(torch.load(MODEL_PATH))