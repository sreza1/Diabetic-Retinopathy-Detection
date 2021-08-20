import torch
import pandas as pd
import numpy as np
import config
from tqdm import tqdm

def make_prediction(model, loader, output_csv="submission.csv"):
    preds = []
    filenames = []
    model.eval()


    for x, y, files in tqdm(loader):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            #cross entropy loss
            pred = model(x).argmax(1)
            preds.append(pred.cpu().numpy())
            filenames += files

        df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
        df.to_csv(output_csv, index=False)
        model.train()
        print("Predictions done")

#cpu since cuda not possible
def check_accuracy(loader, model, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0

    for x, y, filename in tqdm(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            scores = model(x)

        
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.shape[0]

        #add to lists
        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100: .2f}")
    model.train()
    return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(
        all_labels, axis=0, dtype=np.int64
    )
    
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer, lr):
    print("=> Loading checkpint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    #if we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging 
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr