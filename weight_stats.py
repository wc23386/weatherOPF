import numpy as np
import pandas as pd
import torch
# from Sparse_l1_rangeoflambda_newest import SparseAutoencoder
import os
from sklearn.cluster import KMeans
# import matplotlib as plt
import matplotlib.pyplot as plt
from time import strftime

target_dir = os.curdir
exponent_list = np.arange(-1.0, -0.0)
latent_dim = 64
date = '05:21'
time = '15:47'
target_dir = os.path.join(os.curdir, f'Sparse_results/{date}/{time}')
out_date = strftime('%m:%d')
out_time = strftime('%H:%M')
out_target_dir = os.path.join(os.curdir, f'Sparse_results/{out_date}/{out_time}')


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()
print(f"Device: {device}")


for reg_param in exponent_list:
        print('reg_param: ', reg_param)
        dir = os.path.join(target_dir, f"reg_param_{reg_param}/")
        out_dir = os.path.join(out_target_dir, f"reg_param_{reg_param}/")
        os.makedirs(dir, exist_ok=True)  # Create directory if it doesn't exist
        os.makedirs(out_dir, exist_ok=True)  # Create directory if it doesn't exist

        # model = SparseAutoencoder()
        path = os.path.join(dir, f"sparse_model_latent({latent_dim}).pth")
        net = torch.load(path)
        encoder_weights= np.abs(net['enc.weight'].data.cpu().numpy()) # 64 x 411
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
        weight_labels = kmeans.fit_predict(encoder_weights.flatten().reshape(-1,1))
        print(weight_labels.shape) #(26304,)
        # print(weight_labels)
        index = np.where(weight_labels==1)
        print('# of labels=1:', len(index))
        value = encoder_weights.flatten()[index]
        print('stats:', np.min(value), np.max(value), np.size(value))
        binary = np.where(encoder_weights >= np.min(value), 1, 0)
        path = os.path.join(out_dir, 'binary.npy')
        np.save(path, binary)
        # print(np.sum(binary).sum()) #153 out of 26304
        # print(binary.shape) #(64,411)
        coordinates = np.argwhere(binary==0) # can change between 0 and 1
        # print('coordinates:', coordinates)
        print('# of 0:', len(coordinates))
        path = os.path.join(out_dir, 'binary_0.npy')
        np.save(path, coordinates)
        coordinates_1 = np.argwhere(binary==1) # can change between 0 and 1
        print('# of 1:', len(coordinates_1))
        path = os.path.join(out_dir, 'binary_1.npy')
        np.save(path, coordinates_1)

        # non_zero_number = count_zero(encoder_weights, -5)
        # non_zero = np.count_nonzero(encoder_weights)/np.size(encoder_weights)
        # zero_percentage = 1 - non_zero_number
        # print(f'percentage to zero: {zero_percentage}%')
        # plot the heatmap
        plt.matshow(encoder_weights, cmap='gray')
        plt.title('Encoder Weights Heatmap')
        plt.xlabel('Input Feature Index')
        plt.ylabel('Latent Dimension Index')
        plt.colorbar(label='weight value')
        path = os.path.join(out_dir, f'Encoder_weights_all.png')
        # plt.savefig(path)
        plt.close()

        features = ['temperature', 'wind speed', 'cloud coverage and solar radiation']
        # seperate the features
        for i in range(3):
          enc_weight = binary[:,i*137:(i+1)*137]
          plt.matshow(enc_weight, cmap = 'gray')
          plt.title(features[i])
          plt.xlabel('Input Feature Index')
          plt.ylabel('Latent Dimension Index')
          plt.colorbar(label='weight value')
          path = os.path.join(out_dir, f'Encoder_weights_{i+1}.png')
        #   plt.savefig(path)
          plt.close()

        # in average
        avg_list = []
        for i in np.arange(encoder_weights.shape[0]):
            avg = np.mean(encoder_weights[i,:])
            # print(f'latent index {i}', avg)
            avg_list.append(avg)
        # print('length', len(avg_list))

        avg_arr = np.array(avg_list).reshape(-1,1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
        labels = kmeans.fit_predict(avg_arr)
        index_1 = np.where(labels == 1)
        # print('index where label is 1', index_1)


def analysis():

  out_date = strftime('%m:%d')
  out_time = strftime('%H:%M')
  out_target_dir = os.path.join(os.curdir, f'Sparse_results/{out_date}/{out_time}')
  os.makedirs(out_target_dir, exist_ok=True)

  zero_coordinates = np.load('binary_0.npy')
  one_coordinates = np.load('binary_1.npy')
  print('0:', zero_coordinates.shape, '1:', one_coordinates.shape)
  # print('one:', one_coordinates)

  temp_coords = [coord for coord in one_coordinates if coord[1] < 137]
  # print(temp_coords, len(temp_coords))
  # extract unique values
  temp_x = list(set([coord[0] for coord in temp_coords]))
  # print(temp_x)
  data = {}
  for x in temp_x:
    # Filter coordinates with current x value and y < 137
    x_coords = [coord[1] for coord in temp_coords if coord[0] == x]
    data[x] = x_coords
  temp_df = pd.DataFrame.from_dict(data, orient='index')
  temp_df = temp_df.transpose()
  print(temp_df)
  # temp_df.to_csv('temp_active.csv', index=False)

  wind_coords = [coord for coord in one_coordinates if coord[1]>=137 and coord[1]<=137*2]
  # get the unique x
  wind_x = list(set([coord[0] for coord in wind_coords]))
  wind_x.sort()
  # print(wind_x)
  data = {}
  # use unique x as the filter
  for x in wind_x:
      x_coords = [coord[1]%137 for coord in wind_coords if coord[0] == x]
      data[x] = x_coords
  wind_df = pd.DataFrame.from_dict(data, orient='index')
  wind_df = wind_df.transpose()
  # print(wind_df)
  # wind_df.to_csv('wind_active.csv', index=False)

  cloud_coords = [coord for coord in one_coordinates if coord[1]>=137*2 and coord[1]<=137*3]
  # get the unique x
  cloud_x = list(set([coord[0] for coord in cloud_coords]))
  cloud_x.sort()
  # print(cloud_x)
  data = {}
  # use unique x as the filter
  for x in cloud_x:
      x_coords = [coord[1]%137 for coord in cloud_coords if coord[0] == x]
      data[x] = x_coords
  cloud_df = pd.DataFrame.from_dict(data, orient='index')
  cloud_df = cloud_df.transpose()
  # print(cloud_df)
  # cloud_df.to_csv('cloud_active.csv', index=False)
