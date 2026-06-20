import csv, os
import polars as pl
import matplotlib.pyplot as plt

TRAIN_FILENAME = 'log_tloss.csv'
TEST_FIIENAME = 'log_vloss.csv'
def write_train(epoch: int, chunk: int, batch: int, tloss: float):
    write_header = not os.path.exists(TRAIN_FILENAME) or os.path.getsize(TRAIN_FILENAME) == 0
    with open(TRAIN_FILENAME, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'chunk', 'batch', 'tloss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({'epoch': epoch, 'chunk': chunk, 'batch': batch, 'tloss': tloss})
def write_validation(epoch: int, chunk: int, vloss: float):
    write_header = not os.path.exists(TEST_FIIENAME) or os.path.getsize(TEST_FIIENAME) == 0
    with open(TEST_FIIENAME, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'chunk', 'vloss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({'epoch': epoch, 'chunk': chunk, 'vloss': vloss})

def plot():
    try:
        df_train = pl.read_csv(TRAIN_FILENAME)
        df_val = pl.read_csv(TEST_FIIENAME)
    except Exception as e:
        print(f"Error: {e}")
        return

    plt.figure(figsize=(10, 6))

    if 'tloss' in df_train.columns and df_train.height > 0:
        plt.plot(range(df_train.height), df_train['tloss'] / 1000, label='Treino', color='royalblue', alpha=0.6)

    if 'vloss' in df_val.columns and df_val.height > 0:
        if df_train.height > 0:
            scale = df_train.height / df_val.height
            val_x = [i * scale for i in range(df_val.height)]
            plt.plot(val_x, df_val['vloss'], label='Validação', color='crimson', marker='o', linewidth=2)

    plt.title('Curva de Aprendizagem - Loss de Treino e Validação')
    plt.xlabel('Passos')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    plot()