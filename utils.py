import numpy as np
import pandas as pd
import csv
import torch

def read_data_csv(filename):
    reviews = []
    sentiments = []

    with open ('/content/drive/My Drive/' + filename) as csv_data_file:
        csv_reader = csv.reader(csv_data_file)
        for row in csv_reader:
            reviews.append(row[0])
            sentiments.append(row[1])

    df = pd.DataFrame({'REVIEW': np.asarray(reviews), 'SENTIMENT': np.asarray(sentiments, dtype=int)})
    return df

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(model, training_loader, loss_function, optimizer, epoch, device, report_steps=50):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    print(f"There are {len(training_loader)} batches in the training set")
    for i,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if i%report_steps==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss after step {i}: {loss_step}")
            print(f"Training Accuracy after step {i}: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss after Epoch {epoch}: {epoch_loss}")
    print(f"Training Accuracy after Epoch {epoch}: {epoch_accu}")

    return

def valid(model, testing_loader, loss_function, epoch, device, report_steps=10):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    tr_loss = 0; nb_tr_steps = 0; nb_tr_examples = 0
    print(f"There are {len(testing_loader)} batches in the test set")
    with torch.no_grad():
        for i, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if i%report_steps==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Validation Loss after {i} steps: {loss_step}")
                print(f"Validation Accuracy after {i} steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss after Epoch {epoch}: {epoch_loss}")
    print(f"Validation Accuracy after Epoch {epoch}: {epoch_accu}")
    
    return epoch_accu

# Part of the code in this file is modified from the tutorial here 
# https://github.com/abhimishra91/transformers-tutorials/blob/master/transformers_multiclass_classification.ipynb