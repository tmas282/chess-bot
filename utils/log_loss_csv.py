import csv, os

def write_train(epoch: int, chunk: int, batch: int, tloss: float):
    filename = 'log_tloss.csv'
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'chunk', 'batch', 'tloss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({'epoch': epoch, 'chunk': chunk, 'batch': batch, 'tloss': tloss})
def write_validation(epoch: int, chunk: int, vloss: float):
    filename = 'log_vloss.csv'
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'chunk', 'vloss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({'epoch': epoch, 'chunk': chunk, 'vloss': vloss})