import csv

def write_train(epoch: int, chunk: int, batch: int, tloss: float):
    with open('log_tloss.csv', 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'chunk', 'batch', 'tloss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'epoch': epoch, 'chunk': chunk, 'batch': batch, 'tloss': tloss})
def write_validation(epoch: int, chunk: int, vloss: float):
    with open('log_vloss.csv', 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'chunk', 'vloss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'epoch': epoch, 'chunk': chunk, 'vloss': vloss})