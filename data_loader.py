import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Example dataset class
class csv_data(Data.Dataset):
    '''arg: csv files must have a header'''
    def __init__(self, x_csv_file, y_csv_file, device, shuffle=True, batchsize=128):
        self.data_x = pd.read_csv(x_csv_file)
        self.data_y = pd.read_csv(y_csv_file)

        self.train_x_data, self.test_x_data = train_test_split(self.data_x, test_size=0.2, random_state=42, shuffle=shuffle)
        self.train_y_data, self.test_y_data = train_test_split(self.data_y, test_size=0.2, random_state=42, shuffle=shuffle)
        self.device = device
        self.batchsize = batchsize
        
    def GNNDataLoader(self):
        tempN = self.data_x.iloc[:,:137]
        speedN = self.data_x.iloc[:,137:274]
        cloudN = self.data_x.iloc[:,274:411]
        Input=np.dstack([tempN,speedN,cloudN])
        train_x_data, test_x_data = train_test_split(Input, test_size=0.2, random_state=42, shuffle=True)
        train_x_tensor = torch.FloatTensor(train_x_data).to(self.device)
        test_x_tensor = torch.FloatTensor(test_x_data).to(self.device)
        train_y_tensor = torch.FloatTensor(self.train_y_data.values).to(self.device)
        test_y_tensor = torch.FloatTensor(self.test_y_data.values).to(self.device)
        self.train_data = Data.TensorDataset(train_x_tensor, train_y_tensor)
        self.test_data = Data.TensorDataset(test_x_tensor, test_y_tensor)
        self.train_loader = Data.DataLoader(dataset=self.train_data, batch_size=128)
        self.test_loader = Data.DataLoader(dataset=self.test_data, batch_size=128)
        return self.train_loader, self.test_loader


    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        sample_x = self.data_x.iloc[index]
        sample_y = self.data_y.iloc[index]
        return sample_x, sample_y
    
    def get_loaders(self):
        # Convert train and test data to tensors
        train_x_data = torch.FloatTensor(self.train_x_data.values)
        train_y_data = torch.FloatTensor(self.train_y_data.values)
        test_x_data = torch.FloatTensor(self.test_x_data.values)
        test_y_data = torch.FloatTensor(self.test_y_data.values)

        self.train_data = Data.TensorDataset(train_x_data, train_y_data)
        self.test_data = Data.TensorDataset(test_x_data, test_y_data)
        self.train_loader = Data.DataLoader(dataset=self.train_data, batch_size=self.batchsize)
        self.test_loader = Data.DataLoader(dataset=self.test_data, batch_size=self.batchsize)
        return self.train_loader, self.test_loader
    
    def load_data(self):
        return (self.train_x_data, self.train_y_data), (self.test_x_data, self.test_y_data)

class npy_data(Data.Dataset):  # Use 'object' for Python 2/3 compatibility
    def __init__(self, filepath):
        try:
            self.data = np.load(filepath)
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")

    def load_np(self):
        try:            
            return self.data
        except Exception as e:  # Catch any exceptions during loading
            print(f"Error loading data: {e}")
            return None  # Or handle the error differently
