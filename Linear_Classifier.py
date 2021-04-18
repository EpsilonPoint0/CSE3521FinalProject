import argparse
import numpy as np
import pandas as pd


def data_loader(data_path, split):
    FB = pd.read_csv(f'{data_path}/{split}/FB.csv', delimiter=',')
    AAPL = pd.read_csv(f'{data_path}/{split}/AAPL.csv', delimiter=',')
    TSLA = pd.read_csv(f'{data_path}/{split}/TSLA.csv', delimiter=',')

    data = pd.concat((FB, AAPL, TSLA))
    data = data.dropna()

    # construce label
    label_dict = {"I": 1, "D": 0}
    label = list(map(lambda x: label_dict.get(x, 0), data['I/D'].to_list()))
    
    # construct features
    feature = {}
    feature['label'] = label
    feature['S_10'] = data['Close'].rolling(window=10).mean()
    # feature['Corr'] = data['Close'].rolling(window=10).corr(data['S_10'])
    # feature['RSI'] = ta.RSI(np.array(data['Close']), timeperiod=10)
    feature['Open-Close'] = data['Open'] - data['Close'].shift(1)
    feature['Open-Open'] = data['Open'] - data['Open'].shift(1)
    feature = pd.DataFrame(feature)
    feature = feature.dropna()

    # return numpy array
    X = np.array(list(feature[key].to_list() for key in feature if key != 'label'))
    Y = np.array(feature['label'])

    # add bias term
    X = np.concatenate((X, np.ones((1, X.shape[-1]))), axis=0)
    return X, Y


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def LogisticForward(X, w):
    """X: (b, f) w: (f, f2)

    Args:
        X ([type]): [description]
        w ([type]): [description]

    Returns:
        [type]: [description]
    """
    z = np.matmul(X, w)
    return sigmoid(z)


def PerceptronForward(X, w):
    """X: (b, f) w: (f, f2)

    Args:
        X ([type]): [description]
        w ([type]): [description]

    Returns:
        [type]: [description]
    """
    z = np.matmul(X, w)
    return z


def Logistic_Regression(X, Y, learningRate=0.01, maxIter=5000):
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
    D_plus_1 = X.shape[0]
    w = np.random.uniform(size=(D_plus_1, 1))

    # TODO: implement logistic regression (You should be able to implement this < 10 lines)
    # remove this line when you implement your algorithm
    for t in range(maxIter):
        hidden = LogisticForward(X.T, w)
        loss = - np.log(hidden + 1e-10) * Y - np.log(1 - hidden) * (1 - Y)
        grad = (hidden - Y).mean() * w
        w = w - learningRate * grad
        print("Loss: %.3f" % loss.mean())
    # TODO: implement logistic regression

    return w


def Perceptron(X, Y, learningRate=0.01, maxIter=800):
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
    D_plus_1 = X.shape[0]
    w = np.random.uniform(size=(D_plus_1, 1))
    YY = Y.copy()
    YY[YY == 0] = -1

    # TODO: implement perceptron (You should be able to implement this < 10 lines)
    for t in range(maxIter):
        hidden = np.sign(PerceptronForward(X.T, w))

        # wrong cases
        wrong = hidden.squeeze() != YY
        if wrong.sum() == 0:
            return w
        for i in range(len(wrong)):
            if wrong[i]:
                w = w + learningRate * YY[i] * X[:, i: i + 1]

        print("Error: %d" % wrong.sum())

    # TODO: implement perceptron

    return w


def LogisticRegressionAccuracy(X, Y, w, threshold=0.5):
    Y_hat = np.sign(LogisticForward(X.T, w) - threshold)
    Y_hat[Y_hat == -1] = 0
    correct = (Y_hat.squeeze() == Y).sum()
    return float(correct / len(Y_hat))


def PerceptronAccuracy(X, Y, w):
    Y_hat = np.sign(PerceptronForward(X.T, w))
    Y_hat[Y_hat == -1] = 0
    correct = (Y_hat.squeeze() == Y).sum()
    return float(correct / len(Y_hat))


def main(args):
    X_train, Y_train = data_loader(args.data, 'Train')
    X_test, Y_test = data_loader(args.data, 'Test')

    print("number of training data instances: ", X_train.shape)
    print("number of test data instances: ", X_test.shape)
    print("number of training data labels: ", Y_train.shape)
    print("number of test data labels: ", Y_test.shape)

    if args.algorithm == "logistic":
        # #----------------Logistic Loss-----------------------------------
        w = Logistic_Regression(
            X_train, Y_train, maxIter=100, learningRate=0.1)
        training_accuracy = LogisticRegressionAccuracy(X_train, Y_train, w)
        test_accuracy = LogisticRegressionAccuracy(X_test, Y_test, w)
    # # ----------------Perceptron-----------------------------------
    else:
        w = Perceptron(X_train, Y_train, maxIter=100, learningRate=0.1)
        training_accuracy = PerceptronAccuracy(X_train, Y_train, w)
        test_accuracy = PerceptronAccuracy(X_test, Y_test, w)

    print("Accuracy: training set: ", training_accuracy)
    print("Accuracy: test set: ", test_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running linear classifiers")
    parser.add_argument('--data', default="Data", type=str)
    parser.add_argument('--algorithm', default="per", type=str)
    args = parser.parse_args()
    main(args)
