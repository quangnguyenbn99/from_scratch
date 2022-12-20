from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from layers import MLP
import os
import wandb

os.environ['WANDB_API_KEY'] = ''
wandb.login()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# make dataset
def make_datset():
    X,y = make_circles(n_samples = 10000, noise = 0.05, random_state = 36)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = 0.3, random_state = 123)
    return X_tr, X_te, y_tr, y_te
#Create data loader

class Data(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.len = self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return self.len

# nerual network
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(NeuralNetwork,self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
         nn.Dropout(dropout), nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Linear(hidden_dim, output_dim),
         nn.Dropout(dropout), nn.Sigmoid())
        self.init_fn = nn.init.xavier_uniform_
        self.in_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = True
        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.layer_1[0].weight, 1 / self.in_dim)
            init_fn(self.layer_2[0].weight, 1 / self.hidden_dim)
        if self.bias:
            self.layer_1[0].bias.data.zero_()
            self.layer_2[0].bias.data.zero_()

    def forward(self, features):
        x = self.layer_1(features)
        x = self.layer_2(x)
        return x  

class MLP_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_norm=False, dropout=0.0,
     mid_activation: str = 'SiLU', last_activation = 'Sigmoid', device ='cpu'):
        super(MLP_net,self).__init__()
        self.layer = MLP(
            in_dim=input_dim,
            hidden_size=hidden_dim,
            out_dim=output_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            layers=2,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation=last_activation,
            device = device
        )
    def forward(self,features):
        return self.layer(features)


# step = np.linspace(0, 1, 10)
def plot_loss_values(loss_values):
    fig, ax = plt.subplots(figsize=(8,5))
    plt.plot(np.array(loss_values))
    plt.title("Step-wise Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()



def main(mode,X_tr, X_te, y_tr, y_te, model = '1', epochs = 500):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if mode == "train":

        input_dim = 2
        hidden_dim = 10
        output_dim = 1
        if model == '1':
            network = NeuralNetwork(input_dim, hidden_dim, output_dim)
        else:
            network = MLP_net(input_dim, hidden_dim, output_dim)
        # move to gpu
        network = network.to(device)
        # optimizer

        learning_rate = 0.001
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate)


        loss_values = []
        min_loss = np.inf
        # train the model
        for epoch in tqdm(range(epochs)):
            for batch, (features, labels) in enumerate(tqdm(train_loader)):
                # zero the paremeter gradients
                optimizer.zero_grad()
                # forward
                pred = network(features.cuda())
                loss = loss_fn(pred, labels.unsqueeze(-1).cuda())
                loss_values.append(loss.item())
                # backward
                loss.backward()
                # optimize
                optimizer.step()
            wandb.log({'epoch': epoch, 
                'loss': loss})
            if loss.item() <= min_loss:
                min_loss = min_loss
                torch.save(network.state_dict(),'best_model.pt')
    if mode == 'test':
        input_dim = 2
        hidden_dim = 10
        output_dim = 1
        if model == '1':
            network = NeuralNetwork(input_dim, hidden_dim, output_dim)
        else:
            network = MLP_net(input_dim, hidden_dim, output_dim)

        network.load_state_dict(torch.load('best_model.pt'))
        network.to(device)
        network.eval()

        y_pred, y_test = [] ,[]
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (features, labels) in enumerate(tqdm( test_loader)):
                outputs = network(features.cuda())
                predicted = np.where(outputs.cpu().numpy() < 0.5, 0, 1)
                predicted = list(itertools.chain(*predicted))
                y_pred.append(predicted)
                y_test.append(labels)
                total += labels.size(0)
                correct += (predicted == labels.numpy()).sum().item()
        wandb.log({'acc': 100 * correct // total})
        print(f'Accuracy of the network on the 3300 test instances: {100 * correct // total}%')
        y_pred = list(itertools.chain(*y_pred))
        y_test = list(itertools.chain(*y_test))
        print(classification_report(y_test, y_pred))
        cf_matrix = confusion_matrix(y_test, y_pred)

        plt.subplots(figsize=(8, 5))

        sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")
        plt.show()
if __name__ == '__main__':
    project_name = 'test_project2'
    run = wandb.init(project=project_name)
    mode = 'train'
    mode1 = 'test'
    X_tr, X_te, y_tr, y_te = make_datset()
    batchsize = 128
    epochs = 300
    # make dataloader
    train_data = Data(X_tr, y_tr)
    train_loader = DataLoader(train_data, batch_size = batchsize, shuffle =True)

    test_data = Data(X_te, y_te)
    test_loader = DataLoader(test_data, batch_size = batchsize, shuffle = False)

    main(mode,X_tr, X_te, y_tr, y_te, model='2',epochs = epochs)
    main(mode1,X_tr, X_te, y_tr, y_te, model='2')
    run.finish()

    run = wandb.init(project=project_name)
    main(mode,X_tr, X_te, y_tr, y_te, model='1',epochs = epochs)
    main(mode1,X_tr, X_te, y_tr, y_te, model='1')
    run.finish()