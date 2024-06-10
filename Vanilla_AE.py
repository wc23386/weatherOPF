import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import data_loader as dl
from time import strftime
from sklearn.preprocessing import MinMaxScaler
from analysis import DataPlotter

epochs = 3000
learning_rate = 0.001  
batch_size = 128
date = strftime("%m:%d")
in_target_dir = os.curdir
target_dir = os.path.join(os.curdir, f'AE_results/{date}/{strftime("%H:%M")}')
os.makedirs(target_dir, exist_ok=True)
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()
print(f"Device: {device}")

# Initialize graph 
#   plt.style.use('fivethirtyeight')
plt.rc('font', size=15)     
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=10) 
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=10)
plt.rcParams["figure.figsize"] = (8,6)
plt.rcParams['axes.grid'] = True

x_path = os.path.join(in_target_dir, "InputN.csv")
y_path = os.path.join(in_target_dir, "new_VoltN.csv")

data = dl.csv_data(x_path, y_path, device)
(x_train, y_train), (x_test, y_test) = data.load_data()
train_loader, test_loader = data.get_loaders()
print("max", np.max(x_test), np.max(y_test))
print("min", np.min(x_test), np.min(y_test))


class EarlyStopping:
  def __init__(self, patience=10, min_delta=0):
    """
    Args:
      patience (int, optional): Number of epochs to wait for improvement before stopping. Defaults to 10.
      min_delta (float, optional): Minimum change in validation loss to consider as improvement. Defaults to 0.
    """
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
  

# define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self,in_feat, num_hidden1, num_hidden2, num_pred, dropout=0.2, training = True):
        super(Autoencoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.training = training
 
        # encoder
        self.enc1 = nn.Linear(in_features=in_feat, out_features=num_hidden1[0])
        self.enc2 = nn.Linear(in_features=num_hidden1[0], out_features=num_hidden1[1])
        self.enc3 = nn.Linear(in_features=num_hidden1[1], out_features=num_hidden1[2])
        self.enc4 = nn.Linear(in_features=num_hidden1[2], out_features=num_hidden1[3])
 
        # decoder 
        self.dec1 = nn.Linear(in_features=num_hidden1[3], out_features=num_hidden1[2])
        self.dec2 = nn.Linear(in_features=num_hidden1[2], out_features=num_hidden1[1])
        self.dec3 = nn.Linear(in_features=num_hidden1[1], out_features=num_hidden1[0])
        self.dec4 = nn.Linear(in_features=num_hidden1[0], out_features=in_feat)

        # y prediction
        self.pred1 = nn.Linear(in_features=num_hidden1[-1], out_features=num_hidden2[0])
        self.pred2 = nn.Linear(in_features=num_hidden2[0], out_features=num_hidden2[1])
        self.pred3 = nn.Linear(in_features=num_hidden2[1], out_features=num_hidden2[2])
        self.pred4 = nn.Linear(in_features=num_hidden2[2], out_features=num_hidden2[3])
        self.pred5 = nn.Linear(in_features=num_hidden2[3], out_features=num_pred)
 
    def forward(self, x):
        x = x.to(torch.float32)
        x = self.dropout(F.relu(self.enc1(x)))  # Apply dropout after ReLU
        x = self.dropout(F.relu(self.enc2(x)))  # Apply dropout after ReLU
        x = self.dropout(F.relu(self.enc3(x)))
        z = F.relu(self.enc4(x))

        x = self.dropout(F.relu(self.dec1(z)))
        x = self.dropout(F.relu(self.dec2(x)))
        x = self.dropout(F.relu(self.dec3(x)))
        x = self.dropout(F.relu(self.dec4(x)))

        y = self.dropout(F.relu(self.pred1(z)))
        y = self.dropout(F.relu(self.pred2(y)))
        y = self.dropout(F.relu(self.pred3(y)))
        y = self.dropout(F.relu(self.pred4(y)))
        y = F.relu(self.pred5(y))
        
#         x = x.to(torch.float32)
#         # x = x.reshape(128, -1)  
#         x = F.relu(self.enc1(x))
#         x = F.relu(self.enc2(x))
#         x = F.relu(self.enc3(x))
#         z = F.relu(self.enc4(x))

#         # decoding
#         x = F.relu(self.dec1(z))
#         x = F.relu(self.dec2(x))
#         x = F.relu(self.dec3(x))
#         x = F.relu(self.dec4(x))

#         y = F.relu(self.pred1(z))
#         y = F.relu(self.pred2(y))
#         y = F.relu(self.pred3(y))
#         y = F.relu(self.pred4(y))
#         y = F.relu(self.pred5(y))

        return x, y, z

    
class loss():
    def combined_loss(self, x_hat, x, y_pred, y_act):
        lambda1 = 411/(6717+411)
        lambda2 = 6717/(6717+411)
        loss1 = nn.MSELoss()(x_hat, x)
        loss2 = nn.MSELoss()(y_pred, y_act)
        return lambda1 * loss1 + lambda2 * loss2
    
in_feat = 411
num_hidden1 = [256, 128, 64, 8]
num_hidden2 = [64, 512, 1024, 3072]
num_pred = 6717
model = Autoencoder(in_feat, num_hidden1, num_hidden2, num_pred)

# Loss Function 
loss_function = loss().combined_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
net = model.to(device)
model_children = list(model.children())
early_stopping = EarlyStopping(patience=300, min_delta=0.001)


def run_train():
# Training loop variables
  loss_graph = []
  valid_loss_graph = []
  z_train = []

  # Get data loader lengths
  n = len(train_loader)
  n_valid = len(test_loader)
  print("Starting Training Loop...")
  start = time.time()
  for epoch in range(epochs):
    net.train()
    training_loss = 0.0

    for data in train_loader:
      x, y = data
      # print('before', x.device, y.device)
      x = x.to(device)
      y = y.to(device)
      net.zero_grad()
      if x.size(0) == batch_size:
        x = x.reshape(-1, 411) # x = 128,411
        # print('after', x.device, y.device)
        
        x_hat, y_hat, z = net(x)
        z_train.append(z.detach().cpu().numpy())
#         optimizer.zero_grad() # not sure of this function
        err = loss_function(x_hat, x, y_hat, y)
        err.backward()
        optimizer.step()
        training_loss += err.item()
    loss_graph.append(training_loss / n)
#         print(loss_graph, len(loss_graph))

    net.eval()  # Set model to evaluation mode (disable dropout etc.)
    validation_loss = 0.0
    for data in test_loader:
      x, y = data
#       print(x.device, y.device)
      x = x.to(device)
      y = y.to(device)
      x = x.reshape(-1, 411) # x = 128,411
#       print(x.device, y.device)
      x_hat, y_hat, z = net(x)
      err = loss_function(x_hat, x, y_hat, y)
      validation_loss += err.item()
      # Append average validation loss for this epoch
    valid_loss_graph.append(validation_loss / n_valid)
#       print(valid_loss_graph, len(valid_loss_graph))
    if epoch % 1 == 0:
      print('[epoch %d] loss: %.4f \tValidation loss: %.4f' %
            (epoch, training_loss / n, validation_loss / n_valid))
    path = os.path.join(target_dir, 'VanillaAE_model.pth')
    torch.save(model.state_dict(), path)
    # Early stopping
    if early_stopping.early_stop(validation_loss):
      print(f"Training stopped early at epoch {epoch}")
      break  
        
  plt.loglog(np.abs(loss_graph), label='Training')  # Add label for training loss
  plt.loglog(np.abs(valid_loss_graph), label='Validation')  # Add label for validation loss
  plt.legend()  # This will automatically include labels from each plt.loglog call
  plt.xlabel('Epoch')
  plt.ylabel('Training Loss')
  path = os.path.join(target_dir, f"training_loss.png")
  plt.savefig(path, bbox_inches='tight')
  plt.clf()
    
  end = time.time()
  print(f"Time taken: {(end - start)/60:.3f} min")

  

def run_prediction():
    y_pred = []
    y_act = []
    
    in_feat = 411
    num_hidden1 = [256, 128, 64, 8]
    num_hidden2 = [64, 512, 1024, 3072]
    num_pred = 6717
    # Load the model 
    model = Autoencoder(in_feat, num_hidden1, num_hidden2, num_pred)
    path = os.path.join(target_dir, 'VanillaAE_model.pth')
    net = torch.load(path)
    model.load_state_dict(net)
    
    #set model to evaluate mode
    model.eval()
    model.to(device)
    
    with torch.no_grad():
      for data in test_loader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        
        x_hat, y_hat, z = model(x)
        y_pred.append(y_hat.cpu().numpy())
        y_act.append(y.cpu().numpy())
      
      y_pred_all = np.concatenate(y_pred, axis=0)
      y_act_all = np.concatenate(y_act, axis=0)
      path = os.path.join(target_dir, 'y_pred.npy')
      np.save(path, y_pred_all)
      path = os.path.join(target_dir, 'y_act.npy')
      np.save(path, y_act_all)
#       print('y_pred shape', y_pred_all.shape, 'y_act shape', y_act_all.shape)

    scaler = np.genfromtxt('scalar.txt')
#     print(scaler.shape)

    y_pred_inv = y_pred_all*(scaler[1]-scaler[0]) + scaler[0]
    y_act_inv = y_act_all*(scaler[1]-scaler[0]) + scaler[0]

    mse_sample = np.mean((y_pred_inv - y_act_inv)**2, axis=1).reshape(-1,1)
    mse_bus = np.mean((y_pred_inv - y_act_inv)**2, axis=0).reshape(-1,1)
#     print('mse_sample shape', mse_sample.shape, 'mse_bus shape', mse_bus.shape)

    ydp = DataPlotter(mse_sample, target_dir)
    ydp.scatter_sample_err('sample_err_plot.png')
    ydp.hist('sample_err_hist.png')
    ydp_bus = DataPlotter(mse_bus, target_dir)
    ydp_bus.scatter_sample_err('bus_err_plot.png')
    ydp_bus.hist('bus_err_hist.png')

    file_path = os.path.join(target_dir, 'result.txt')
    with open(file_path, "w") as f:
        f.write(f'Model: Vanilla AE \n')
        f.write(f'epochs: {epochs}\n')
        f.write(f'shape of x training set, y training set: {x_train.shape}, {y_train.shape}\n')
        f.write(f'shape of x testing set, y testing set: {x_test.shape}, {y_test.shape}\n')
        f.write(f'layer construction: x {num_hidden1}, y {num_hidden2} \n')
        f.write(f'shape of prediction set: {y_pred_all.shape}, {y_act_all.shape}\n')
        f.write(f'ground true stats (Min/Max/Mean): {np.min(y_act_inv)}, {np.max(y_act_inv)}, {np.mean(y_act_inv)}\n')
        f.write(f'prediction stats (Min/Max/Mean): {np.min(y_pred_inv)}, {np.max(y_pred_inv)}, {np.mean(y_pred_inv)}\n')
        f.write(f'MSE for samples (Min/Max/Mean): {np.min(mse_sample)}, {np.max(mse_sample)}, {np.mean(mse_sample)}\n')
        f.write(f'MSE for buses (Min/Max/Mean): {np.min(mse_bus)}, {np.max(mse_bus)}, {np.mean(mse_bus)}\n')

train = True
prediction = True
if train:
  run_train()
if prediction:
  run_prediction()