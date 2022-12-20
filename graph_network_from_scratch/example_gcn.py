import rdkit
from torch_geometric.datasets import MoleculeNet
 
data = MoleculeNet(root=".", name="ESOL")
print(data)

print("Dataset type: ", type(data))
print("Dataset features: ", data.num_features)
print("Dataset target: ", data.num_classes)
print("Dataset length: ", data.len)
print("Dataset sample: ", data[0])
print("Sample  nodes: ", data[0].num_nodes)
print("Sample  edges: ", data[0].num_edges)


import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
embedding_size = 64

class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers ( for Message Passing )
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer ( for scalar output ... REGRESSION )
        self.out = Linear(embedding_size*2, 1)

    def forward(self, x, edge_index, batch_index):
        hidden = F.tanh(self.initial_conv(x, edge_index))
        hidden = F.tanh(self.conv1(hidden, edge_index))
        hidden = F.tanh(self.conv2(hidden, edge_index))
        hidden = F.tanh(self.conv3(hidden, edge_index))
          
        # Global Pooling (stack different aggregations)
        ### (reason) multiple nodes in one graph....
        ## how to make 1 representation for graph??
        ### use POOLING! 
        ### ( gmp : global MAX pooling, gap : global AVERAGE pooling )
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)
        import pdb
        pdb.set_trace()
        out = self.out(hidden)
        import pdb
        pdb.set_trace()
        return out, hidden

model = GCN()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data_size = len(data)
NUM_GRAPHS_PER_BATCH = 64
loader = DataLoader(data[:int(data_size * 0.8)], 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):], 
                         batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def train(data):
    for batch in loader:
      batch.to(device)  
      optimizer.zero_grad() 
      #---------------------------------------------------------------#
      # data : (1) node features & (2) connection info
      import pdb
      pdb.set_trace()
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      #---------------------------------------------------------------#
      loss = torch.sqrt(loss_fn(pred, batch.y))       
      loss.backward()  
      optimizer.step()   
    return loss, embedding

print("Starting training...")
losses = []
for epoch in range(2000):
    loss, h = train(data)
    losses.append(loss)
    if epoch % 100 == 0:
      print(f"Epoch {epoch} | Train Loss {loss}")