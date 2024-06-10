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

# Constructing the argument parser
# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
# # ap.add_argument("-l", "--reg_param", type=float, default=0.1, help="regularization parameter")
# ap.add_argument("-sc", "--add_sparse", type=str, default='yes', help="whether to add sparsity constraint or not")
# ap.add_argument("-d", "--date", type=str, default='0506', help="date")
# args = vars(ap.parse_args())

epochs = 3000
# reg_param = args['reg_param']
add_sparse = 'yes'
learning_rate = 0.001  
batch_size = 128
latent_dim = 32
exponent_list = np.arange(-1.0, 0.0)
date = strftime("%m:%d")
in_target_dir = os.curdir
target_dir = os.path.join(os.curdir, f'Sparse_results/{date}/{strftime("%H:%M")}')
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
# print(y_train.iloc[:, 2807], y_test.iloc[:, 2807])
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(y_train.head, y_test)
train_loader, test_loader = data.get_loaders()
# print("x", np.max(x_train), np.min(x_train))
# print("y", np.max(y_train), np.min(y_train))
# print("x", np.max(x_test), np.min(x_test))
# print("y", np.max(y_test), np.min(y_test))


class EarlyStopping:
  def __init__(self, patience=100, min_delta=0):
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
class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
 
        # encoder
        self.enc = nn.Linear(in_features=411, out_features=latent_dim)
 
        # decoder 
        self.dec = nn.Linear(in_features=latent_dim, out_features=411)

        # y prediction
        self.pred1 = nn.Linear(in_features=latent_dim, out_features=6717)
 
    def forward(self, x):
        x = x.to(torch.float32)
        # x = x.reshape(128, -1)  
        # encoding
        z = F.relu(self.enc(x))
        # decoding
        y_hat = F.relu(self.pred1(z))
        x_hat = F.relu(self.dec(z))
        return x_hat, y_hat, z
    
class loss():
    def combined_loss(self, x_hat, x, y_pred, y_act):
        lambda1 = 411/(6717+411)
        lambda2 = 6717/(6717+411)
        loss1 = nn.MSELoss()(x_hat, x)
        loss2 = nn.MSELoss()(y_pred, y_act)
        return lambda1 * loss1 + lambda2 * loss2

model = SparseAutoencoder().to(device)
# Loss Function 
loss_function = loss().combined_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
net = model.to(device)
# Get the layers as a list
model_children = list(model.children())
# print(model_children)
early_stopping = EarlyStopping(patience=100, min_delta=0.001)  # Create EarlyStopping instance


# Define the sparse loss function
def sparse_loss(input):
    loss = 0
    # print(input.shape , input.dtype, len(model_children))
    values = input.to(torch.float32)
    for i in range(len(model_children) - 1):
        # print(model_children[i], values.shape, values.dtype)
        values = F.relu(model_children[i](values))
        loss += torch.mean(torch.abs(values))
    return loss

def count_zero(data_array, exponent_for_episilon):
   count = 0
   value = data_array.flatten()
   for i in value:
      if np.abs(i) <= 10**exponent_for_episilon:
         count += 1
      return count 
   
def run_train():
  exponent_values = []
  validation_mse_list = []
  # Get data loader lengths
  n = len(train_loader)
  n_valid = len(test_loader)
  print("Starting Training Loop...")
  start = time.time()
  for reg_param in exponent_list:
      print(reg_param)
      dir = os.path.join(target_dir, f"reg_param_{reg_param}/")
      os.makedirs(dir, exist_ok=True)  # Create directory if it doesn't exist
    # Training loop variables
      loss_graph = []
      valid_loss_graph = []
      z_train = []

      for epoch in range(epochs):
          net.train()
          training_loss = 0.0
          counter=0
          for data in train_loader:
              counter += 1
              x, y = data
              x=x.to(device)
              y=y.to(device)
              # print(x.shape, y.shape)
              if x.size(0) == batch_size:
                x = x.reshape(-1, 411) # x = 128,411
                optimizer.zero_grad()
                x_hat, y_hat, z = net(x)
                z_train.append(z.detach().cpu().numpy())
                mse_loss = loss_function(x_hat, x, y_hat, y)
                if add_sparse == 'yes':
                    l1_loss = sparse_loss(x)
                    err = mse_loss + (10**reg_param) * l1_loss
                else:
                    err = mse_loss   
                err.backward()
                optimizer.step()
                training_loss += err.item()
          epoch_loss = training_loss/counter
          loss_graph.append(epoch_loss)

          # Validation loop
          net.eval()  # Set model to evaluation mode (disable dropout etc.)
          validation_loss = 0.0
          counter = 0
          z_test=[]
        
          with torch.no_grad():
        
              for data in test_loader:
                counter += 1
                x, y = data
                x = x.to(device)
                y = y.to(device)
                x = x.reshape(-1, 411) # x = 128,411
                x_hat, y_hat, z = net(x)
                z_test.append(z.detach().cpu().numpy()) 
                err = loss_function(x_hat, x, y_hat, y)
                validation_loss += err.item()
          # Append average validation loss for this epoch
          epoch_loss = validation_loss / counter
          valid_loss_graph.append(epoch_loss)
          z_test_np = np.concatenate(z_test, axis=0)
          path = os.path.join(dir, f"z_test.npy")
          np.save(path, z_test_np)
          path = os.path.join(dir, f"sparse_model_latent({latent_dim}).pth")
          torch.save(model.state_dict(), path)
          if epoch % 1 == 0:
              print('[epoch %d] loss: %.4f \tValidation loss: %.4f' %
                (epoch, training_loss / n, validation_loss / n_valid)) 
                
          plt.loglog(np.abs(loss_graph[:-1])) # TODO: can improve later
          plt.loglog(np.abs(valid_loss_graph[:-1]))
          plt.legend(['Training','Validation'])
          plt.xlabel('Epoch')
          plt.ylabel('Training Loss')
          path = os.path.join(dir, f"training_loss.png")
          plt.savefig(path, bbox_inches='tight')
          plt.clf()
          plt.close()
      end = time.time()
      print(f"Time taken: {(end - start)/60:.3f} min")

      if early_stopping.early_stop(validation_loss):
          print(f"Training stopped early at epoch {epoch}")
          break
      validation_mse = valid_loss_graph[-1]
      validation_mse_list.append(validation_mse)
      print(validation_mse_list)
      exponent_values.append(reg_param) # Store current lambda
      print('reguarizer:', exponent_values)
    
  # plot the reg_param vs mse
  plt.plot(exponent_values, np.log10(validation_mse_list))
  plt.xlabel('log(lambda)')
  plt.ylabel("Validation MSE")
  plt.title("Validation MSE vs Regularization Strength")
  plt.grid(True)
  path = os.path.join(target_dir, f"mse_reg.png")
  plt.savefig(path)
  plt.clf()
  plt.close()

def run_prediction():
    exponent_value = []
    pred_err_sample = []
    pred_err_bus = []
    for reg_param in exponent_list:
        print('reg_param: ', reg_param)
        dir = os.path.join(target_dir, f"reg_param_{reg_param}/")
        os.makedirs(dir, exist_ok=True)  # Create directory if it doesn't exist
        model = SparseAutoencoder()
        path = os.path.join(dir, f"sparse_model_latent({latent_dim}).pth")
        net = torch.load(path)
        model.load_state_dict(net)
        model.eval()
        model.to(device)
        y_pred = []
        y_act = []
        with torch.no_grad():
            
            for data in test_loader:
             
                x, y = data
                x=x.to(device)
                y=y.to(device)
                x_hat, y_hat, z = model(x)  # Perform predictions
                # x_pred_np = x_hat.cpu().numpy()
                y_pred.append(y_hat.cpu().numpy())
#                 print(len(y_pred), y_pred[0].shape)
                y_act.append(y.cpu().numpy())
#                 print(y_act, len(y_act))

              # Concatenate all arrays in y_pred along axis 0
            y_pred_all = np.concatenate(y_pred, axis=0)
            y_act_all = np.concatenate(y_act, axis=0)
#             print('y_pred_all shape', y_pred_all.shape, 'y_act_all shape', y_act_all.shape)
            
            # Save the arrays as numpy files
            path = os.path.join(dir, f"y_pred.npy")
            np.save(path, y_pred_all)
            path = os.path.join(dir, f"y_act.npy")
            np.save(path, y_act_all)
        
            # Convert to real value
            scaler = np.genfromtxt('scalar.txt')
#             print('scaler shape', scaler.shape)

            y_pred_inv = y_pred_all*(scaler[1]-scaler[0]) + scaler[0]
            y_act_inv = y_act_all*(scaler[1]-scaler[0]) + scaler[0]
#             print('y_pred_inv Max/Min', np.max(y_pred_inv), np.min(y_pred_inv))
#             print('y_act_inv Max/Min', np.max(y_act_inv), np.min(y_act_inv))
                
            mse_sample = np.mean((y_pred_inv - y_act_inv)**2, axis=1).reshape(-1,1)
#             print('sample err shape', mse_sample.shape)
            mse_bus = np.mean((y_pred_inv - y_act_inv)**2, axis=0).reshape(-1,1)
#             print('bus err shape', mse_bus.shape)

            ydp = DataPlotter(mse_sample, dir)
            ydp.scatter_sample_err('sample_err_plot')
            ydp.hist('sample_err_frequency')
            ydp_bus = DataPlotter(mse_bus, dir)
            ydp_bus.scatter_sample_err('bus_err_plot')
            ydp_bus.hist('bus_err_frequency')

        exponent_value.append(reg_param)
#         print('exponent_value: ', exponent_value)
        pred_err_sample.append(np.mean(mse_sample))
        pred_err_bus.append(np.mean(mse_bus))

        print('Plotting Heatmap...')
        # print(net.state_dict()) # tensors 
        encoder_weights= np.abs(net['enc.weight'].data.cpu().numpy()) # 64 x 411
        pred_weights= net['pred1.weight']  # should be 64 x 6717
        # print('encoder shape', encoder_weights.shape)
        non_zero_number = count_zero(encoder_weights, -5)
        # non_zero = np.count_nonzero(encoder_weights)/np.size(encoder_weights)
        zero_percentage = 1 - non_zero_number
        print(f'percentage to zero: {zero_percentage}%')
        # plot the heatmap
        plt.matshow(encoder_weights, cmap='gray')
        plt.title('Encoder Weights Heatmap')
        plt.xlabel('Input Feature Index')
        plt.ylabel('Latent Dimension Index')
        plt.colorbar(label='weight value')
        path = os.path.join(dir, f'Encoder_weights_all.png')
        plt.savefig(path)
        plt.close()

        features = ['temperature', 'wind speed', 'cloud coverage and solar radiation']
        # seperate the features
        for i in range(3):
          enc_weight = encoder_weights[:,i*137:(i+1)*137]
          plt.matshow(enc_weight, cmap = 'gray')
          plt.title(features[i])
          plt.xlabel('Input Feature Index')
          plt.ylabel('Latent Dimension Index')
          plt.colorbar(label='weight value')
          path = os.path.join(dir, f'Encoder_weights_{i+1}.png')
          plt.savefig(path)
          plt.close()

        file_path = os.path.join(dir, 'result.txt')
        with open(file_path, 'w') as f:
          f.write(f'reg_param {reg_param}\n')
          f.write(f'MSE for samples (Min/Max/Mean): {np.min(mse_sample)}, {np.max(mse_sample)}, {np.mean(mse_sample)}\n')
          f.write(f'MSE for buses (Min/Max/Mean): {np.min(mse_bus)}, {np.max(mse_bus)}, {np.mean(mse_bus)}\n')
          f.write(f'weight stats (Min/Max/Mean): {np.min(encoder_weights)}, {np.max(encoder_weights)}, {np.mean(encoder_weights)}\n')
          f.write(f'proportion of weights zero: {zero_percentage} ')
        

    file_path = os.path.join(target_dir, f"result_all.txt")             
    with open(file_path, "w") as f:
        f.write('Model: Sparse AE/ L1 loss')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Latent Variables: {latent_dim}\n')
        f.write(f'shape of x training set, y training set: {x_train.shape}, {y_train.shape}\n')
        f.write(f'shape of x testing set, y testing set: {x_test.shape}, {y_test.shape}\n')
        f.write(f'ground true stats (Min/Max/Mean): {np.min(y_act_inv)}, {np.max(y_act_inv)}, {np.mean(y_act_inv)}\n')
        f.write(f'prediction stats (Min/Max/Mean): {np.min(y_pred_inv)}, {np.max(y_pred_inv)}, {np.mean(y_pred_inv)}\n')
        f.write(f'shape of prediction set (sample, bus): {mse_sample.shape}, {mse_bus.shape}\n')
        f.write(f'exponent_value: {exponent_value}\n')
        f.write(f'pred_err_sample: {pred_err_sample}\n')
        f.write(f'pred_err_bus: {pred_err_bus}\n')

                
    # plot the reg_param vs mse
    plt.plot(exponent_value, np.log10(pred_err_sample))
    plt.xlabel('log(lambda)')
    plt.ylabel("Prediction Error MSE")
    plt.title("Prediction Error MSE vs Regularization Strength")
    plt.grid(True)
    path = os.path.join(target_dir, f"pred_mse_reg.png")
    plt.savefig(path)
    plt.clf()
    plt.close()

train = True
prediction = True
if train:
  run_train()
if prediction:
  run_prediction()