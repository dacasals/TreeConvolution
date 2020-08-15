# < begin copyright > 
# Copyright Ryan Marcus 2019
# 
# This file is part of TreeConvolution.
# 
# TreeConvolution is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# TreeConvolution is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with TreeConvolution.  If not, see <http://www.gnu.org/licenses/>.
# 
# < end copyright > 
 
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.sgd import SGD
from torch.utils.data.dataset import Dataset

from util import prepare_trees, split_ds, prepare_log_std_target
import tcnn
import  pandas as pd
# First tree:
#               (0, 1)
#       (1, 2)        (-3, 0)
#   (0, 1) (-1, 0)  (2, 3) (1, 2)

tree1 = (
    (0, 1),
    ((1, 2), ((0, 1),), ((-1, 0),)),
    ((-3, 0), ((2, 3),), ((1, 2),))
)

# Second tree:
#               (16, 3)
#       (0, 1)         (2, 9)
#   (5, 3)  (2, 6)

tree2 = (
    (16, 3),
    ((0, 1), ((5, 3),), ((2, 6),)),
    ((2, 9),)
)

tree3 = (
    (16, 3),
    ((0, 1), ((5, 3),), ((2, 6),)),
    ((2, 9),)
)

trees = [tree1, tree2, tree3]
treesVal = [tree1, tree2, tree3]

# function to extract the left child of a node
def left_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[1]

# function to extract the right child of node
def right_child(x):
    assert isinstance(x, tuple)
    if len(x) == 1:
        # leaf.
        return None
    return x[2]

# function to transform a node into a (feature) vector,
# should be a numpy array.
def transformer(x):
    return np.array(x[0])






class WatDivDataset(Dataset):
    # def __init__(self, range0_1, range_1_5, range_5_10, range_10_20, range_20_last):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, x, y):
        """
        Initialization
        x: nparray
        y: nparray
        x an y  will be converted to FloatTensor
        """
        self.x = x
        self.y = torch.from_numpy(y).to(torch.float32)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.x[0].shape[0]

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample

        # Load data and get label
        xitem = tuple([self.x[0][index], self.x[1][index]])
        yitem = self.y[index]

        return xitem, yitem





urifile = "./dataset_prepared_with_trees1000.csv"
#Read csv
all_data = pd.read_csv(urifile, sep="ᶶ")

#split data
train_dataset, val_dataset, test_dataset = split_ds(all_data, 0.3, 0.2)
y_train = train_dataset[['time']]
y_val = val_dataset[['time']]
y_test = test_dataset[['time']]

x_train = train_dataset['exec_tree_single'].apply(lambda  x: eval(x))
x_val = val_dataset['exec_tree_single'].apply(lambda  x: eval(x))
x_test = test_dataset['exec_tree_single'].apply(lambda  x: eval(x))
params = {'batch_size': 128, 'shuffle': True, 'num_workers': 2}


# prepared_tredddes_train = prepare_trees(trees, transformer, left_child, right_child)
# training_set = WatDivDataset(prepared_tredddes_train, np.array([1., 2., 3.]))
# training_generator = torch.utils.data.DataLoader(training_set, **params)
#
# prepared_tredddes_val = prepare_trees(treesVal, transformer, left_child, right_child)
# validation_set = WatDivDataset(prepared_tredddes_val, np.array([1., 2., 3.]))
# validation_generator = torch.utils.data.DataLoader(validation_set, **params)
#
# for train_step, (trees, target) in enumerate(training_generator):
#     print(trees, target)

#scale target in log scale and StandardScaller
scalery, y_train_log_std, y_val_log_std, y_test_log_std = prepare_log_std_target(y_train, y_val, y_test)

prepared_trees_train = prepare_trees(x_train.values, transformer, left_child, right_child)
prepared_trees_val = prepare_trees(x_val.values, transformer, left_child, right_child)
prepared_trees_test = prepare_trees(x_test.values, transformer, left_child, right_child)
# Parameters


# Generators
training_set = WatDivDataset(prepared_trees_train, y_train_log_std.reshape(-1,1))
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = WatDivDataset(prepared_trees_val, y_val_log_std.reshape(-1,1))
validation_generator = torch.utils.data.DataLoader(validation_set, **params)



# CUDA for PyTorch
import torch.utils.data as Data


max_epochs = 5

# A tree convolution neural network mapping our input trees with
# 2 channels to trees with 16 channels, then 8 channels, then 4 channels.
# Between each mapping, we apply layer norm and then a ReLU activation.
# Finally, we apply "dynamic pooling", which returns a flattened vector.
net = nn.Sequential(
    tcnn.BinaryTreeConv(57, 16),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(16, 8),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.BinaryTreeConv(8, 4),
    tcnn.TreeLayerNorm(),
    tcnn.TreeActivation(nn.ReLU()),
    tcnn.DynamicPooling(),
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

MODELS = []
# output: torch.Size([2, 4])
# def train_net(train_data, validation_data):
#
#     print(net(prepared_trees).shape)
#     learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
#     loss_func = MSELoss()
#     model = net(prepared_trees)
#     optimizer = SGD(model.parameters(), lr=learning_rates[0])
#
#     test_error = torch.zeros(len(learning_rates))
#     validation_error = torch.zeros(len(learning_rates))
#     for i, learning_rate in enumerate(learning_rates):
#
#         y_train_hat = model(train_data.x)
#         loss = loss_func(y_train_hat, train_data.y)
#         test_error[i] = loss.item()
#
#         MODELS.append(model)
#
#         y_val_hat = model(validation_data.x)
#         loss = loss_func(y_val_hat, validation_data.y)
#         validation_error[i] = loss.item()


def train_net(net, train_data, validation_data, epochs):
    print("Starting training..")

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1]
    loss_func = MSELoss()
    # model = net(train_data)
    optimizer = Adam(net.parameters(), lr=0.00015)

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    for e in range(0, epochs):
        print('=' * 20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('=' * 20)
        LOSS = []
        LOSS_Real = []
        LOSS_VAL = []
        LOSS_Real_VAL = []

        for train_step, (trees, target) in enumerate(train_data):
            net.zero_grad()

            y_train_hat = net(trees)
            loss = loss_func(y_train_hat, target)
            optimizer.zero_grad()
            LOSS.append(loss.item())
            inv_train_hat =  np.exp(scalery.inverse_transform(y_train_hat.detach().numpy()))
            inv_train =  np.exp(scalery.inverse_transform(target.numpy()))
            rmse = np.sqrt(mean_squared_error(inv_train, inv_train_hat))

            LOSS_Real.append(rmse)
            loss.backward()
            optimizer.step()

        with torch.set_grad_enabled(False):
            # for val_step, (trees, target) in enumerate(validation_data):
            y_val_hat = net(validation_data.dataset.x)
            val_loss = loss_func(y_val_hat, validation_data.dataset.y)
            LOSS_VAL.append(val_loss.item())

            inv_val_hat = np.exp(scalery.inverse_transform(y_val_hat.numpy()))
            inv_val = np.exp(scalery.inverse_transform(validation_data.dataset.y.numpy()))
            rmse = np.sqrt(mean_squared_error(inv_val, inv_val_hat))
            LOSS_Real_VAL.append(rmse)
            # print(f'Val loss: {val_loss:.4f}, Acc: {acc:.4f}')

        train_loss = np.average(LOSS)
        train_lossr = np.average(LOSS_Real)
        valid_loss = np.average(LOSS_VAL)
        valid_lossr = np.average(LOSS_Real_VAL)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print(f'Training loss: {train_loss:.4f} Val loss: {val_loss:.4f}')
        print(f'Training Real loss: {train_lossr:.4f} Val Real loss: {valid_lossr:.4f}')

    return net


net_trained = train_net(net, training_generator, validation_generator, epochs=max_epochs)

def getMetricstorch(dataset_name, net, x_test, y_test):
    net.eval()

    y_pred = np.exp(scalery.inverse_transform(net(x_test).detach().numpy()))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(len(y_test))
    print(f'Set {dataset_name}, Loss: {rmse:.4f}')

def printVals(dataset_name, net, x_test, y_test):
    net.eval()
    # y_pred = net(x_testTensor)
    # net(x_testTensor).cpu().data.numpy()
    loss_func = MSELoss()
    y_pred = np.exp(scalery.inverse_transform(net(x_test).detach().numpy()))
    #     rmse = np.sqrt(mean_squared_error(y_true_data, y_pred))
    for real,pred in list(zip(y_test, y_pred)):
        print(f'Set {dataset_name}, Real {real:.4f} :Prediction {pred:.4f}')


testing_set = WatDivDataset(prepared_trees_test, y_test_log_std.reshape(-1,1))
testing_generator = torch.utils.data.DataLoader(testing_set, **params)

getMetricstorch(net_trained, training_set.x, y_train.values)
getMetricstorch(net_trained, validation_set.x, y_val.values)
getMetricstorch(net_trained, testing_set.x, y_test.values)

printVals(net_trained, training_set.x, y_train.values)
printVals(net_trained, validation_set.x, y_val.values)
printVals(net_trained, testing_set.x, y_test.values)
