import torch
import numpy as np
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.nn.conv import MessagePassing
from typing import Optional
from torch import Tensor
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from rdkit import Chem
import numpy as np
import ogb
from ogb.utils import smiles2graph
import pandas as pd
import itertools
import wandb
import os
from sklearn.metrics import classification_report
os.environ['WANDB_API_KEY'] = '24c624b508c19653be95bc4b5db977b43193ac64'
wandb.login()
# from ogb.graphproppred import PygGraphPropPredDataset

# edge index: the tensor defining the source and target nodes of  all deges
# edge_index = torch.tensor([[0, 1, 1, 2],[1, 0, 2, 1]], dtype=torch.long)
# # x: node features
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
# data = Data(x=x, edge_index = edge_index)
# print(data)

# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# data = Data(x=x, edge_index=edge_index.t().contiguous())
# print(data)

# What we can know from data
# data.x: node features
# data.edge_index: edge index (show which node links to which node)
# data.edge_attr: edge attribute also knows as edge features

# the dataset feature should be saved as Tensor
class CustomDataset(Dataset):
    def __init__(self, smiles, comp_property):

        self.smiles = smiles
        self.property = comp_property
        self.len = len(smiles)
        self.data = []
        for idx, smile in enumerate(tqdm(self.smiles)):
            graph = smiles2graph(smile)
            datapoint = Data (x = torch.from_numpy(graph['node_feat']),     
                edge_index = torch.from_numpy(graph['edge_index']),
                edge_attr = torch.from_numpy(graph['edge_feat']),
                num_nodes = graph['num_nodes'],
                y = self.property[idx])
                #     import datace()
            self.data.append(datapoint)
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.len

class GCNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: str = False , **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        # embed the nodes matrix
        x = self.lin(x)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # propagate, which internally calls message(), aggregate() and update().

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        return out

class GCN(torch.nn.Module):
    #GCN net (the cobination of multiple Gcn layers)
    def __init__(self, input_dim, hidden, agrregation,out =1):
        super(GCN,self).__init__()
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, agrregation)
        self.out = nn.Linear(agrregation, out)

    def forward(self, data):
        x, edge_index = data.x.type(dtype=torch.FloatTensor).cuda(), data.edge_index.cuda()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x, edge_index)
        # readout
        x = global_mean_pool(x, data.batch.cuda())
        return F.sigmoid(self.out(x))

def main():
    
    batchsize = 20
    epochs = 100
    df = pd.read_csv(r'./HIV.csv')
    train_length = round(len(df)*0.8)
    smiles = list(df['smiles'])[:train_length ]
    comp_property = list(df['HIV_active'])[:train_length]
    smiles_test = list(df['smiles'])[train_length:]
    comp_property_test = list(df['HIV_active'])[train_length:]

    # make dataloader
    train_data = CustomDataset(smiles, comp_property)
    train_loader = DataLoader(train_data, batch_size = batchsize, shuffle =True)

    test_data = CustomDataset(smiles_test, comp_property)
    test_loader = DataLoader(test_data, batch_size = batchsize, shuffle =False)

    # set up input dimension of the network
    input_dim = train_data[0].x.size(1)
    hidden_dim = 16
    agrregation_dim = 9
    output_dim = 1
    
    # Network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(input_dim, hidden_dim, agrregation_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = nn.BCELoss()

    min_loss = np.inf
    # training
    model.train()
    
    for epoch in tqdm(range(epochs)):
        for x, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y.unsqueeze(-1).type(dtype=torch.FloatTensor).cuda())
            loss.backward()
            optimizer.step()
        wandb.log({'epoch': epoch, 'loss': loss})
        if loss.item() <= min_loss:
            min_loss = min_loss
            torch.save(model.state_dict(),'best_model.pt')
    # testing
    model = GCN(input_dim, hidden_dim, agrregation_dim, output_dim)
    model.load_state_dict(torch.load('best_model.pt'))
    model.to(device)
    model.eval()
    y_pred, y_test = [] ,[]
    correct, total = 0, 0
    
    with torch.no_grad():
        for x, batch in enumerate(tqdm(test_loader)):
            pred = model(batch)
            predicted = np.where(pred.cpu().numpy() < 0.5, 0, 1)
            predicted = list(itertools.chain(*predicted))
            y_pred.append(predicted)
            y_test.append(batch.y.numpy())
            total += batch.y.size(0)
            correct += (predicted == batch.y.numpy()).sum().item()
    
    acc = 100 * correct // total
    print(f'Accuracy: {acc:.4f}')
    y_pred = list(itertools.chain(*y_pred))
    y_test = list(itertools.chain(*y_test))
    print(classification_report(y_test, y_pred))
    wandb.log({'acc': acc})
if __name__ == '__main__':
    project_name = 'test_gnn_from_scratch'
    run = wandb.init(project=project_name)
    main()
    run.finish()