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

from utils.fen_to_array import fens_to_arrays
from utils.model_files import create_model_state_folder
from utils.numpy_array_files import get_saved_arrays, save_arrays

class ChessHeuristicDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.uint8)
        self.y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ChessHeuristicEvaluator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heuristic = nn.Sequential(
            nn.Conv2d(in_channels=12,out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(128 * 8**2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        x = x.type(torch.float32)
        logits = self.heuristic(x)
        return logits

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessHeuristicEvaluator().to(DEVICE)
EPOCHS = 200
INITIAL_LEARNING_RATE = 0.001
BATCH_SIZE = 256
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
SCHEDULER = lr_schedulers.ReduceLROnPlateau(OPTIMIZER, mode='min', factor=0.1, patience=1)
LOSS_FN = torch.nn.L1Loss()
MODEL_PATH = "model_states/chess_heuristic_evaluator"
DATASET_TRAIN_PATH = "preprocessed_data/chess_heuristic_evaluator_train_dataset"
DATASET_TEST_PATH = "preprocessed_data/chess_heuristic_evaluator_test_dataset"

def pre_process_df(df: polars.DataFrame):
    print("Normalizing Final Evaluation")
    df = df.filter(~polars.col("Evaluation").str.contains("#"))
    df = df.filter((polars.col("Evaluation").cast(polars.Float32) <= 1000) & (polars.col("Evaluation").cast(polars.Int32) >= -1000))
    df = df.with_columns(
        Evaluation=polars.col("Evaluation").cast(polars.Float32).map_batches(lambda x: np.tanh(x/200))
    )
    y = df.select(polars.col("Evaluation")).to_numpy()
    print("Normalizing FEN")
    X = fens_to_arrays(df.select(polars.col("FEN")).to_numpy())

    X_train, X_test, y_train, y_test = normalize_split_data(X, y)

    return X_train, X_test, y_train, y_test

def normalize_split_data(X: np.ndarray, y: np.ndarray):
    print("Splitting testing data and training data")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def get_datasets():
    arrs = get_saved_arrays(DATASET_TRAIN_PATH, DATASET_TEST_PATH)
    if arrs is None:
        print("Getting dataset")
        abs_path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
        path = os.path.join(abs_path, "chessData.csv")
        print("Reading dataset")
        df = polars.read_csv(path)
        print("Preprocessing dataset")
        arrs = pre_process_df(df=df)
        save_arrays(DATASET_TRAIN_PATH, DATASET_TEST_PATH, arrs[0], arrs[1], arrs[2], arrs[3])
    else:
        print("Preprocessing skipped, using saved normalization")
    print("Creating Datasets and Dataloaders")
    train_dataset = ChessHeuristicDataset(arrs[0], arrs[2])
    test_dataset = ChessHeuristicDataset(arrs[1], arrs[3])
    
    return train_dataset, test_dataset

def create_dataloaders(train_dataset: 'ChessHeuristicDataset', test_dataset: 'ChessHeuristicDataset'):
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    return train_dl, test_dl

def train_one_epoch(epoch_index, tb_writer, train_dl: "DataLoader"):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dl):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Zero your gradients for every batch!
        OPTIMIZER.zero_grad()

        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = LOSS_FN(outputs, labels)
        loss.backward()

        # Adjust learning weights
        OPTIMIZER.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print(f"Lr {OPTIMIZER.param_groups[0]['lr']}  batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(train_dl) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
 
def train(train_dl: "DataLoader", test_dl: "DataLoader"):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/ChessHeuristicEvaluator_train_{timestamp}')

    best_vloss = 1_000_000.

    print("Starting Training")
    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer, train_dl)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(test_dl):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)
                voutputs = model(vinputs)
                vloss = LOSS_FN(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')
        
        SCHEDULER.step(avg_vloss)
        
        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'{MODEL_PATH}_{timestamp}_{epoch}'
            torch.save(model.state_dict(), model_path)
    print("Ended Training")

def start_training():
    print(f"DEVICE={DEVICE}\nINITIAL_LEARNING_RATE={INITIAL_LEARNING_RATE}\nEPOCHS={EPOCHS}\nBATCH_SIZE={BATCH_SIZE}")
    create_model_state_folder(MODEL_PATH)
    train_ds, test_ds = get_datasets()
    train_dl, test_dl = create_dataloaders(train_ds, test_ds)
    train(train_dl, test_dl)

if __name__ == "__main__":
    start_training()

def use_model():
    create_model_state_folder()
    model = ChessHeuristicEvaluator()
    model.load_state_dict(torch.load(MODEL_PATH))