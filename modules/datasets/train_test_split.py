# coding : utf-8
# Author : yuxiang Zeng

import numpy as np

def get_train_valid_test_dataset(tensor, args):
    p = np.random.permutation(len(tensor))
    tensor = tensor[p]

    X = tensor[:, :-1]
    Y = tensor[:, -1].reshape(-1, 1)
    # max_value = Y.max()
    max_value = 1
    Y /= max_value

    train_size = int(len(tensor) * args.density)
    if args.dataset == 'cpu':
        valid_size = int(100)
    elif args.dataset == 'gpu':
        valid_size = int(200)

    X_train = X[:train_size]
    Y_train = Y[:train_size]

    X_valid = X[train_size:train_size + valid_size]
    Y_valid = Y[train_size:train_size + valid_size]

    X_test = X[train_size + valid_size:]
    Y_test = Y[train_size + valid_size:]

    train_tensor = np.hstack((X_train, Y_train))
    valid_tensor = np.hstack((X_valid, Y_valid))
    test_tensor = np.hstack((X_test, Y_test))

    return train_tensor, valid_tensor, test_tensor, max_value