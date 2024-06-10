import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from time import strftime

class DataPlotter:
    def __init__(self, data, target_dir) -> None:
        self.data = data
        self.target_dir = target_dir
        self.num_samples, self.num_features = data.shape
        assert self.num_samples!=0, self.num_features!=0

    def scatter_sample_err(self, filename):
        plt.figure()
        for i in range(self.num_samples):
            plt.plot(np.ones(self.num_features)*i, self.data[i,:], 'o', markersize=0.8)
        plt.xlabel('features')
        plt.ylabel('error value')
        plt.title(filename)
        plt.grid()

        filepath = os.path.join(self.target_dir, filename)
        plt.savefig(filepath)
        plt.clf
        
    def hist(self, filename):
        """Creates a histogram for each feature in the data and saves the figure.
        Args:
        filename (str): Name of the file to save the figure.
        """

      # Create figure and axes
        fig, axes = plt.subplots(nrows=2, ncols=1)  # Create 2 rows and 1 column for subplots
        fig.suptitle(filename)
        for i in range(self.num_features):
            axes[0].grid(True)
            axes[0].hist(self.data[:, i])
            axes[0].set_ylabel('Frequency')

        for i in range(self.num_features):
            bins, edges = np.histogram(self.data[:,i], 100)
            left, right = edges[:-1], edges[1:]
            x_axis = np.array([left, right]).T.flatten()
            y_axis = np.array([bins, bins]).T.flatten()
            axes[1].grid(True)
            axes[1].plot(x_axis, y_axis)
            axes[1].set_xlabel('error value')
            axes[1].set_ylabel('frequency')

        fig.tight_layout()
        path = os.path.join(self.target_dir, filename)
        fig.savefig(path)
        fig.clf()