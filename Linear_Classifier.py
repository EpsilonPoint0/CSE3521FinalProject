# import scipy.io as sio
import numpy as np
from numpy import genfromtxt
import math
import argparse
import scipy.io as sio
import yfinance as yf
yf.pdr_override()





def data_loader(data_name):
    if data_name == "all":

        FB = genfromtxt('Train/FB2.csv', delimiter=',')
        AAPL = genfromtxt('Train/AAPL2.csv', delimiter=',')
        TSLA = genfromtxt('Train/TSLA1.csv', delimiter=',')

    print(TSLA)






def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def Logistic_Regression(X, Y, learningRate=0.01, maxIter=100):
    """
    Input:
        X: a (D+1)-by-N matrix (numpy array) of the input data; that is, we have concatenate "1" for you
        Y: a N-by-1 matrix (numpy array) of the label
    Output:
        w: the linear weight vector. Please represent it as a (D+1)-by-1 matrix (numpy array).
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
    """
    N = X.shape[1]
    D_plus_1 = X.shape[0]
    w = np.zeros((D_plus_1, 1))
    Y[Y == -1] = 0.0  # change label to be {0, 1}
    totalLoss = 0
    error = np.zeros((D_plus_1, 1))
    for t in range(maxIter):
        for i in range(N):
            y1 = LR(X, w)
            X_n = X[:, i].reshape((D_plus_1, 1))
            l = (-Y[i] * (1 - sigmoid(Y[i] * np.matmul(w.T, X_n)))) * X_n
            error = np.add(l, error)
        w = (w - ((error / N) * learningRate)).reshape((D_plus_1, 1))


    ####### TODO: implement logistic regression (You should be able to implement this < 10 lines)
    # remove this line when you implement your algorithm

    ####### TODO: implement logistic regression

    Y[Y == 0] = -1  # change label to be {-1, 1}
    return w


def LR(X, theta):
    z = np.matmul(theta.T, X)
    return 1 / (1 + np.exp(-z))


def loss(X, y, theta):
    y1 = LR(X, theta)
    print(X)
    print(np.matmul(theta.T, X))
    l = (-y * (1 - sigmoid(y * np.matmul(theta.T, X)))) * X
    return l


"""
def probability(w, x):
    return sigmoid(netInput(w, x))


def netInput(w, x):
    # print(repr(w))
    # print(repr(x))
    return np.dot(w, x)
"""


def Perceptron(X, Y, learningRate=0.01, maxIter=100):
    """
    Input:
        X: a (D+1)-by-N matrix (numpy array) of the input data; that is, we have concatenate "1" for you
        Y: a N-by-1 matrix (numpy array) of the label; labels are {-1, 1} and you have to turn them to {0, 1}
    Output:
        w: the linear weight vector. Please represent it as a (D+1)-by-1 matrix (numpy array).
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
        3. np.sign: for sign
    """
    N = X.shape[1]
    D_plus_1 = X.shape[0]
    w = np.zeros((D_plus_1, 1))
    np.random.seed(1)

    for t in range(maxIter):
        permutation = np.random.permutation(N)
        X = X[:, permutation]
        Y = Y[permutation, :]
        for n in range(N):

            y_n_hat = np.sign(np.dot(w.T, X))
            print(w.T, X)
            if y_n_hat != Y[n]:
                w[n] = w[n] + learningRate * (Y[n] * X[n])
            ####### TODO: implement perceptron (You should be able to implement this < 10 lines)
            # remove this line when you implement your algorithm

    ####### TODO: implement perceptron

    return w


def Accuracy(X, Y, w):
    Y_hat = np.sign(np.matmul(X.transpose(), w))
    correct = (Y_hat == Y)
    return float(sum(correct)) / len(correct)


def main(args):
    X_train, Y_train, X_test, Y_test = data_loader(args.data)
    print("number of training data instances: ", X_train.shape)
    print("number of test data instances: ", X_test.shape)
    print("number of training data labels: ", Y_train.shape)
    print("number of test data labels: ", Y_test.shape)

    if args.algorithm == "logistic":
        # #----------------Logistic Loss-----------------------------------
        w = Logistic_Regression(X_train, Y_train, maxIter=100, learningRate=0.1)
    # # ----------------Perceptron-----------------------------------
    else:
        w = Perceptron(X_train, Y_train, maxIter=100, learningRate=0.1)

    training_accuracy = Accuracy(X_train, Y_train, w)
    test_accuracy = Accuracy(X_test, Y_test, w)
    print("Accuracy: training set: ", training_accuracy)
    print("Accuracy: test set: ", test_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running linear classifiers")
    parser.add_argument('--data', default="all", type=str)
    args = parser.parse_args()
    main(args)
