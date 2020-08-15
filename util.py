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
from sklearn.preprocessing import StandardScaler


class TreeConvolutionError(Exception):
    pass


def _is_leaf(x, left_child, right_child):
    has_left = left_child(x) is not None
    has_right = right_child(x) is not None

    if has_left != has_right:
        raise TreeConvolutionError(
            "All nodes must have both a left and a right child or no children"
        )

    return not has_left


def _flatten(root, transformer, left_child, right_child):
    """ turns a tree into a flattened vector, preorder """

    if not callable(transformer):
        raise TreeConvolutionError(
            "Transformer must be a function mapping a tree node to a vector"
        )

    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )

        
    accum = []

    def recurse(x):
        if _is_leaf(x, left_child, right_child):
            accum.append(transformer(x))
            return

        accum.append(transformer(x))
        recurse(left_child(x))
        recurse(right_child(x))

    recurse(root)

    try:
        accum = [np.zeros(accum[0].shape)] + accum
    except:
        raise TreeConvolutionError(
            "Output of transformer must have a .shape (e.g., numpy array)"
        )

    return np.array(accum)


def _preorder_indexes(root, left_child, right_child, idx=1):
    """ transforms a tree into a tree of preorder indexes """

    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a " +
            "tree node to its child, or None"
        )

    if _is_leaf(root, left_child, right_child):
        # leaf
        return idx

    def rightmost(tree):
        if isinstance(tree, tuple):
            return rightmost(tree[2])
        return tree

    left_subtree = _preorder_indexes(left_child(root), left_child, right_child,
                                     idx=idx + 1)

    max_index_in_left = rightmost(left_subtree)
    right_subtree = _preorder_indexes(right_child(root), left_child, right_child,
                                      idx=max_index_in_left + 1)

    return (idx, left_subtree, right_subtree)


def _tree_conv_indexes(root, left_child, right_child):
    """ 
    Create indexes that, when used as indexes into the output of `flatten`,
    create an array such that a stride-3 1D convolution is the same as a
    tree convolution.
    """

    if not callable(left_child) or not callable(right_child):
        raise TreeConvolutionError(
            "left_child and right_child must be a function mapping a "
            + "tree node to its child, or None"
        )

    index_tree = _preorder_indexes(root, left_child, right_child)

    def recurse(root):
        if isinstance(root, tuple):
            my_id = root[0]
            left_id = root[1][0] if isinstance(root[1], tuple) else root[1]
            right_id = root[2][0] if isinstance(root[2], tuple) else root[2]
            yield [my_id, left_id, right_id]

            yield from recurse(root[1])
            yield from recurse(root[2])
        else:
            yield [root, 0, 0]

    return np.array(list(recurse(index_tree))).flatten().reshape(-1, 1)


def _pad_and_combine(x):
    assert len(x) >= 1
    assert len(x[0].shape) == 2

    for itm in x:
        if itm.dtype == np.dtype("object"):
            raise TreeConvolutionError(
                "Transformer outputs could not be unified into an array. "
                + "Are they all the same size?"
            )

    second_dim = x[0].shape[1]
    for itm in x[1:]:
        assert itm.shape[1] == second_dim

    max_first_dim = max(arr.shape[0] for arr in x)

    vecs = []
    for arr in x:
        padded = np.zeros((max_first_dim, second_dim))
        padded[0:arr.shape[0]] = arr
        vecs.append(padded)

    return np.array(vecs)

import  pandas as pd
def prepare_trees(trees, transformer, left_child, right_child):
    flat_trees = [_flatten(x, transformer, left_child, right_child) for x in trees]
    flat_trees = _pad_and_combine(flat_trees)
    flat_trees = torch.Tensor(flat_trees)

    # flat trees is now batch x max tree nodes x channels
    flat_trees = flat_trees.transpose(1, 2)

    indexes = [_tree_conv_indexes(x, left_child, right_child) for x in trees]
    indexes = _pad_and_combine(indexes)
    indexes = torch.Tensor(indexes).long()

    return (flat_trees, indexes)

from sklearn.model_selection import train_test_split

def split_ds(all_data, val_rate, test_rate):
    """test_rate is a rate of the total, val_rate is a rate of the (total - test_rate)"""
    range = {}
    range['0_1'] = all_data[(all_data["time"] > 0) & (all_data["time"] <= 1)]
    range['1_5'] = all_data[(all_data["time"] > 1) & (all_data["time"] <= 5)]
    range['5_10'] = all_data[(all_data["time"] > 5) & (all_data["time"] <= 10)]
    range['10_20'] = all_data[(all_data["time"] > 10) & (all_data["time"] <= 20)]
    range['20_last'] = all_data[(all_data["time"] > 20)]
    train_data = []
    val_data = []
    test_data = []
    for rang in range.values():
        if rang.shape[0] >= 3:
            X_temp, X_test = train_test_split(
                rang, test_size=test_rate, random_state=42, shuffle=True)

            X_train, X_val = train_test_split(
                X_temp, test_size=val_rate, random_state=42, shuffle=True)

            train_data.append(X_train)
            val_data.append(X_val)
            test_data.append(X_test)
    train_data_list = pd.concat(train_data)
    val_data_list = pd.concat(val_data)
    test_data_list = pd.concat(test_data)
    print("Shapes : Train: {} Val: {}, Test: {}".format(train_data_list.shape, val_data_list.shape, test_data_list.shape))

    return train_data_list, val_data_list, test_data_list

def prepare_log_std_target(y_train, y_val, y_test):
    # Pass Target to log scale
    y_train_log = np.log(y_train.values.reshape(-1, 1))
    y_val_log = np.log(y_val.values.reshape(-1, 1))
    y_test_log = np.log(y_test.values.reshape(-1, 1))

    # Standarizar target
    scalery = StandardScaler()
    y_train_log_std = scalery.fit_transform(y_train_log)
    y_val_log_std = scalery.transform(y_val_log)
    y_test_log_std = scalery.transform(y_test_log)

    return  scalery, y_train_log_std, y_val_log_std, y_test_log_std
