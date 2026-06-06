import time

import kagglehub
import numpy as np
import pandas
import torch
import os
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class ChessHeuristicDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features, dtype=torch.float).to(DEVICE)
        self.y = torch.tensor(targets, dtype=torch.float).view(-1, 1).to(DEVICE)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class ChessHeuristicEvaluator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flat = nn.Flatten()
        self.heuristic = nn.Sequential(
            nn.Linear(68 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.flat(x)
        logits = self.heuristic(x)
        return logits

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessHeuristicEvaluator().to(DEVICE)
EPOCHS = 200
LEARNING_RATE = 0.002
SCALER = MinMaxScaler()
OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
LOSS_FN = torch.nn.MSELoss()
PATH = "chess_heuristic_evaluator"

def pre_process_df(df: pandas.DataFrame):
    df["Evaluation"] = df["Evaluation"].apply(parse_evaluation)
    X = convert_fens_to_arrays(df["FEN"])
    y = df["Evaluation"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = SCALER.fit_transform(X_train)
    X_test = SCALER.transform(X_test)

    train_dataset = ChessHeuristicDataset(X_train, y_train)
    test_dataset = ChessHeuristicDataset(X_test, y_test)
    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dl, test_dl

def parse_evaluation(eval: str) -> np.ndarray:
    if(eval.startswith("#+")):
        return np.array(10_000, dtype=np.int32)
    if(eval.startswith("#-")):
        return np.array(-10_000, dtype=np.int32)
    eval_v = np.array(eval, dtype=np.int32)
    if(eval_v < -10_000):
        return np.array(-10_000, dtype=np.int32)
    if(eval_v > 10_000):
        return np.array(10_000, dtype=np.int32)
    return eval_v

def convert_fens_to_arrays(s: pandas.Series) -> np.ndarray:
    vec = np.vectorize(fen_to_integer_array, signature="()->(n)")
    res = vec(s.to_numpy())
    return res

def fen_to_integer_array(board_fen: str) -> np.ndarray:
    fen_parts = board_fen.split(" ")
    arr = []
    for i in str(fen_parts[0]):
        try:
            arr.extend([0 for _ in range(int(i))])
            continue
        except:
            pass
        if( i == 'r'):
            arr.append(-5)
        elif( i == 'n'):
            arr.append(-3)
        elif( i == 'b'):
            arr.append(-4)
        elif( i == 'q'):
            arr.append(-9)
        elif( i == 'k'):
            arr.append(-10)
        elif( i == 'p'):
            arr.append(-1)
        elif( i == 'R'):
            arr.append(5)
        elif( i == 'N'):
            arr.append(3)
        elif( i == 'B'):
            arr.append(4)
        elif( i == 'Q'):
            arr.append(9)
        elif( i == 'K'):
            arr.append(10)
        elif( i == 'P'):
            arr.append(1)
        elif( i == '.'):
            arr.append(0)
    
    can_castle = [1 if "K" in fen_parts[2] else 0, 1 if "Q" in fen_parts[2] else 0,
                  1 if "k" in fen_parts[2] else 0, 1 if "q" in fen_parts[2] else 0]

    arr.extend(can_castle)
    return np.array(arr, dtype=np.int8)

def train_one_epoch(epoch_index, tb_writer, train_dl: "DataLoader"):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_dl):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        OPTIMIZER.zero_grad()

        # Make predictions for this batch
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
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(train_dl) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
 
def train(train_dl: "DataLoader", test_dl: "DataLoader"):
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/ChessHeuristicEvaluator_train_{timestamp}')

    best_vloss = 1_000_000.

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
                voutputs = model(vinputs)
                vloss = LOSS_FN(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'chess_heuristic_evaluator_{timestamp}_{epoch}'
            torch.save(model.state_dict(), model_path)

def start_training():
    abs_path = kagglehub.dataset_download("ronakbadhe/chess-evaluations")
    path = os.path.join(abs_path, "chessData.csv")
    df = pandas.read_csv(path)
    train_dl, test_dl = pre_process_df(df=df)
    train(train_dl, test_dl)

if __name__ == "__main__":
    print(DEVICE)
    start_training()

def use_model():
    model = ChessHeuristicEvaluator()
    model.load_state_dict(torch.load(PATH))