# coding : utf-8
# Author : yuxiang Zeng

import numpy as np

def get_train_valid_test_dataset(tensor, args):
    X = tensor[:, :-1]
    Y = tensor[:, -1].reshape(-1, 1)
    max_value = Y.max()
    Y /= max_value
    # print(np.array(X).shape)
    # print(np.array(Y).shape)
    # args.valid = 1
    trainsize = int(len(X) * args.density)
    validsize = int(len(X) * 0.05) if args.valid else int((len(X) - trainsize) * 1.0)

    Idx = np.arange(len(X))
    p = np.random.permutation(len(X))
    Idx = Idx[p]

    trainRowIndex = Idx[:trainsize]
    traintensorX = np.zeros_like(X)
    traintensorY = np.zeros_like(Y)
    traintensorX[trainRowIndex] = X[trainRowIndex]
    traintensorY[trainRowIndex] = Y[trainRowIndex]

    validStart = trainsize
    validRowIndex = Idx[validStart:validStart + validsize]
    validtensorX = np.zeros_like(X)
    validtensorY = np.zeros_like(Y)
    validtensorX[validRowIndex] = X[validRowIndex]
    validtensorY[validRowIndex] = Y[validRowIndex]

    testStart = validStart + validsize
    testRowIndex = Idx[testStart:]
    testtensorX = np.zeros_like(X)
    testtensorY = np.zeros_like(Y)
    testtensorX[testRowIndex] = X[testRowIndex]
    testtensorY[testRowIndex] = Y[testRowIndex]

    traintensor = np.hstack((traintensorX, traintensorY))
    validtensor = np.hstack((validtensorX, validtensorY))
    testtensor = np.hstack((testtensorX, testtensorY))

    return traintensor, validtensor, testtensor, max_value
