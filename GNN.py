import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch.optim as optim
import os
import numpy as np
import pandas as pd
import data_loader as dl
import math
from analysis import DataPlotter
from time import strftime
from sklearn.model_selection import train_test_split

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
batch_size = 128
lr = 0.0001 #0.0002
# Beta1 hyperparam for Adam optimizers
# beta1 = 0.8
num_epochs = 2000
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

plt.rc('font', size=15)     
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)
plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['axes.grid'] = True
# Initialize graph 
#   plt.style.use('fivethirtyeight')

in_target_dir = os.curdir
date = strftime("%m:%d")
target_dir = os.path.join(os.curdir, f'GNN_results/{date}/{strftime("%H:%M")}')
os.makedirs(target_dir, exist_ok=True)

x_path = os.path.join(in_target_dir, "InputN.csv")
y_path = os.path.join(in_target_dir, "new_VoltN.csv")


data = dl.csv_data(x_path, y_path, device)
(x_train, y_train), (x_test, y_test) = data.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
tempN = x_train.iloc[:,:137]
speedN = x_train.iloc[:,137:274]
cloudN = x_train.iloc[:,274:411]
Input=np.dstack([tempN,speedN,cloudN])
# print(Input.shape)
path = os.path.join(target_dir, "Input.npy")
np.save(path, Input)

train_loader, test_loader = data.GNNDataLoader()
print(len(train_loader), len(test_loader)) # 55,14
print("x training set", np.max(x_train), np.min(x_train))
print("y training set", np.max(y_train), np.min(y_train))


class EarlyStopping:
  def __init__(self, patience=10, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_loss = float('inf')  # Initialize best loss as infinity

  def early_stop(self, val_loss):
    if val_loss < self.best_loss:
      self.best_loss = val_loss
      self.counter = 0
      return False
    elif val_loss > (self.best_loss + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        print(f"Early stopping triggered after {self.counter} epochs without improvement.")
        return True
    return False  # Continue training

class GraphConvolution(nn.Module):
    """GCN layer"""
    def __init__(self, in_feats, out_feats, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_feats
        self.out_features = out_feats
        
        # Parameter
        # Create a 2D tensor with dimensions `in_features` by `out_features`
        self.weight = nn.Parameter(torch.FloatTensor(in_feats, out_feats))
        # initializing the bias term
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)
        # Initialize bias if it exists
        if self.bias is not None:
            nn.init.zeros_(self.bias)

# Implementation of the GCN layer forward pass
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj.unsqueeze(0).expand(input.shape[0],6717,6717), support)
    
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
class GraphConvolution_branch(nn.Module):
    """GCN layer"""

    def __init__(self, in_feats, out_features, bias=True):
        super(GraphConvolution_branch, self).__init__()
        self.in_features = in_feats
        self.out_features = out_features
        
        # Parameter
        self.weight = nn.Parameter(torch.FloatTensor(in_feats, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj.unsqueeze(0).expand(input.shape[0],9140,6717), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GNN(nn.Module):
    def __init__(self, ngpu, num_features, num_hidden1, num_class1, dropout, training = True):
        super(GNN, self).__init__()
        self.ngpu = ngpu
        
        self.gc1 = GraphConvolution(num_features, num_hidden1[0])
        self.gc2 = GraphConvolution(num_hidden1[0], num_hidden1[1])
        self.gc3 = GraphConvolution(num_hidden1[1], num_hidden1[2])
        self.gc4 = GraphConvolution(num_hidden1[2], num_hidden1[3])
        self.gc5 = GraphConvolution(num_hidden1[3], num_class1)
        # self.gc6 = GraphConvolution_branch(num_hidden1[3], 1)
        # map from (all) edge features to n selected edges
#         self.lin_output = nn.Linear(1,1)
        self.dropout = dropout
        self.training = training

# net(inputs, adj_tensor, adj_w_tensor, Inc_tensor.T)
# adj_tensor.shape = torch.Size([6717, 6717])
# adj_w_tensor.shape (137, 6717) 
# Inc_tensor.T.shape = torch.Size([9140, 6717])

    def forward(self, x, adj, adj_w, incid):
#         x == inputs ?
#         x = torch.transpose(inputs,1,2) # was this
        x = torch.transpose(x,1,2) # ?
        # inputs: torch.Size([128, 137, 3])
        # transpose: torch.Size([128, 3, 137])
        x = torch.matmul(x, adj_w)
        # adj_w.shape (137, 6717) 
        # x.shape = (128, 3, 6717)
        x = torch.transpose(x,1,2)
        # x.shape = (128, 6717, 3)
        #x = x.unsqueeze(2)
        
#     def forward(self, input, adj):
#         support = torch.matmul(input, self.weight)
#         output = torch.bmm(adj.unsqueeze(0).expand(inputs.shape[0],6717,6717), support)
        x = self.gc1(x, adj)
    # self.weight = tensor (in_feat, out_feat) = (3, 10)
    # support = matmul(x, self.weight) = (128, 6717, 3) * (3, 10) = (128, 6717, 10)
    # output = torch.bmm(adj.unsqueeze(0).expand(inputs.shape[0],6717,6717), support)
    # adj.torch.Size([6717, 6717]).unsqueeze(0) = torch.Size([1, 6717, 6717])
    # x.shape = (128, 6717, 10)
        x = F.dropout(x, self.dropout, training = self.training)
        # randomly zeros some of the elements in input tensor with prbability p (default = 0.5)
        x = self.gc2(x, adj)
    # x.shape = (128, 6717, 10) adj.torch.Size([6717, 6717]
    # matmul(x, self.weight) = (128, 6717, 10) * (10, 10) = (128, 6717, 10)
    # output = torch.bmm(adj.unsqueeze(0).expand(inputs.shape[0],6717,6717), support)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        # matmul(x, self.weight) = (128, 6717, 10) * (10, 10) = (128, 6717, 10)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj) 
        # matmul(x, self.weight) = (128, 6717, 10) * (10, 5) = (128, 6717, 5)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc5(x, adj)
        bus = torch.relu(x) # Applies the rectified linear unit function element-wise
        
        return torch.squeeze(bus, 2)
# Computes the expit (also known as the logistic sigmoid function) of the elements of input.
# line = (128, 9140, 1)
# return torch.squeeze(bus, 2), torch.squeeze(line, 2)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    # checks whether the class name (classname) contains the substring 'Conv'
    if classname.find('Conv') != -1:
        # initializes the weights of convolutional layers with a normal distribution (mean=0.0, std=0.02)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # the weights of batch normalization layers with a normal distribution (mean=1.0, std=0.02)
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # setting the biases of batch normalization layers to zero
        nn.init.constant_(m.bias.data, 0)

# in current folder
path = os.path.join(in_target_dir, "WS_lintransform_act_full.npy")
adj_w=np.load(path)
path = os.path.join(in_target_dir, "adj_full.npy")
adj=np.load(path)
path = os.path.join(in_target_dir, "Incidence.npy")
Incidence=np.load(path)

adj_tensor = torch.FloatTensor(adj).to(device)
adj_w_tensor = torch.FloatTensor(adj_w).to(device)
Inc_tensor = torch.FloatTensor(Incidence).to(device)

features = 3
hidden_features = [10,10,10,5]
net = GNN(ngpu, features, hidden_features, 1, dropout=0.02).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    # This allows the model to be trained on multiple GPUs in parallel
    net = nn.DataParallel(net, list(range(ngpu)))
print(net)

# Initialize weights
net.apply(weights_init)
# # Print the model
# print(net)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr)#, betas=(beta1, 0.999))                                                     

# Training Loop
loss_graph = [] # training loss.
valid_loss_graph = [] # validation loss
n = len(train_loader)
n_valid = len(test_loader)
early_stopping = EarlyStopping(patience=100, min_delta=0.001)


print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    training_loss = 0.0
    for data in train_loader:
        net.zero_grad()
        # sets the gradients of all its parameters (including parameters of submodules) to zero
        inputs, targets_V = data
        # print(inputs.shape) # 128, 411
        outputs_V = net(inputs, adj_tensor, adj_w_tensor, Inc_tensor.T)
#         print(outputs_L.shape, targets_L.shape)
        err_V = criterion(outputs_V, targets_V)
        err = err_V
        err.backward()
        optimizer.step()
        training_loss += err.item()
        
    loss_graph.append(training_loss / n)
    
    #Validation
    validation_loss = 0.0
    #For each batch in the dataloader
    for data in test_loader:
        inputs, targets_V = data
        outputs_V = net(inputs, adj_tensor, adj_w_tensor, Inc_tensor.T)
        err_valid_V = criterion(outputs_V, targets_V)
        err_valid = err_valid_V
        validation_loss += err_valid.item()
    valid_loss_graph.append(validation_loss / n_valid)
    
    if epoch % 1 == 0:
        print('[epoch: %d] loss: %.4f \tValidation loss: %.4f'%(epoch, training_loss/n, validation_loss/n_valid))
    if early_stopping.early_stop(validation_loss):
        print(f"Training stopped early at epoch {epoch}")
        break  

ax = plt.subplot()
plt.loglog(np.abs(loss_graph))
plt.loglog(np.abs(valid_loss_graph))
ax.legend(['Traning','Validation'],fancybox=False,facecolor='white',edgecolor='black')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()
path = os.path.join(target_dir, f"training_loss.png")
plt.savefig(path,format='png',bbox_inches='tight')
path = os.path.join(target_dir, f"training_loss.pt")
torch.save(net, path)

y_pred_np = []
y_act_np = []

with torch.no_grad():
    for data in test_loader:
        inputs, targets_V = data
        outputs_V = net(inputs, adj_tensor, adj_w_tensor, Inc_tensor.T)
        y_pred_np.append(outputs_V.cpu().numpy())
        y_act_np.append(targets_V.cpu().numpy())
    y_pred_all = np.concatenate(y_pred_np, axis=0)
    y_act_all = np.concatenate(y_act_np, axis=0)

#     print('y_pred shape', y_pred_all.shape, 'y_act_all', y_act_all.shape)
    path = os.path.join(target_dir, f'gen_GNN_Volt.npy')
    np.save(path, y_pred_all)
    path = os.path.join(target_dir, f'act_GNN_Volt.npy')
    np.save(path, y_act_all)

def run_analysis():
  path = os.path.join(target_dir, f"gen_GNN_Volt.npy")
  y_pred = np.load(path)
  path = os.path.join(target_dir, f"act_GNN_Volt.npy")
  y_test = np.load(path)
  print('y_pred shape', y_pred.shape, 'y_test shape', y_test.shape)
  mse_sample = np.mean((y_pred - y_test) ** 2, axis=1)
  mse_bus = np.mean((y_pred - y_test) ** 2, axis=0)
  print('mse_sample shape', mse_sample.shape, 'mse_bus shape', mse_bus.shape)
  mse_sample_np = mse_sample.reshape(-1,1)
  mse_bus_np = mse_bus.reshape(-1,1)


  dp = DataPlotter(mse_sample_np, target_dir)
  dp.scatter_sample_err('sample_err_plot.png')
  dp.hist('sample_err_hist.png')
  dp_bus = DataPlotter(mse_bus_np, target_dir)
  dp_bus.scatter_sample_err('bus_err_plot.png')
  dp_bus.hist('bus_err_hist.png')
  
  file_path = os.path.join(target_dir, f"result.txt")
  with open(file_path, "w") as f:
      f.write('Model: GNN\n')
      f.write(f'Epochs: {num_epochs}\n')
      f.write(f'shape of x training set, y training set: {x_train.shape}, {y_train.shape}\n')
      f.write(f'shape of x testing set, y testing set: {x_test.shape}, {y_test.shape}\n')
      f.write(f'shape of prediction set: {y_pred_all.shape}\n')
      f.write(f'MSE for samples (Min/Max/Mean): {np.min(mse_sample)}, {np.max(mse_sample)}, {np.mean(mse_sample)}\n')
      f.write(f'MSE for buses (Min/Max/Mean): {np.min(mse_bus)}, {np.max(mse_bus)}, {np.mean(mse_bus)}\n')

analysis = True
if analysis:
    run_analysis()