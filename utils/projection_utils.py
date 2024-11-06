import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

import numpy as np

def head(output_layer, label):
	label = np.hstack((label, label))

	# y_hat = clf.predict(output_layer)
	y_hat = output_layer
	# y_hat = (y_hat)/y_hat-(np. mean(y_hat-label))
	y_hat = y_hat - label

	transform = nn.Sequential(
	# nn.Linear(1,1),  # Linear transformation
	# nn.Dropout(p=0.1),
	# nn.BatchNorm1d(1),
	# nn.LeakyReLU(negative_slope=1e-02),
	)
	y_hat = torch.from_numpy(y_hat).float().reshape(-1,1)
	return transform(y_hat)


def projection(output_layer, input_layer):
	input_layer =  torch.from_numpy(input_layer).reshape(-1,1).float()
	# input_layer =  torch.from_numpy(input_layer).view(int(input_layer.shape[0]/2),2).mean(1).reshape(-1,1)
	label_size, batch_size = 1, output_layer.shape[1]
	transform = nn.Sequential(
	nn.Linear(label_size, batch_size),  # Linear transformation
	# nn.BatchNorm1d(batch_size),
	# nn.ReLU()   # Batch Normalization
	)
	return transform(input_layer)

def modelCoventional(output_layer, label):
	clf.fit(output_layer, label)
	print(mean_squared_error(label,clf.predict(output_layer)))
	return clf
