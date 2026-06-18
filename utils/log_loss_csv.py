import csv

def write_to_file(epoch: int, tloss: float, vloss: int):
    with open('log_loss.csv', 'a', newline='') as csvfile:
        fieldnames = ['epoch','tloss', 'vloss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'epoch': epoch, 'tloss':tloss, 'vloss': vloss})