from sklearn import preprocessing
from  classifier.linear_classifier import *
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

def achieve_best(method):
    # Use the validation set to tune hyperparameters (regularization strength and
    # learning rate). You should experiment with different ranges for the learning
    # rates and regularization strengths; if you are careful you should be able to
    # get a classification accuracy of about 0.4 on the validation set.
    learning_rates = [1e-7, 5e-6]
    regularization_strengths = [2.5e4, 5e4]

    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
    # of data points that are correctly classified.
    results = {}
    best_val = -1  # The highest validation accuracy that we have seen so far.
    best_linear_method = None  # The LinearSVM object that achieved the highest validation rate.

    iters = 2000  # 100
    for lr in learning_rates:
        for rs in regularization_strengths:
            linear_method = method
            linear_method.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=iters)

            y_train_pred = linear_method.predict(X_train)
            acc_train = np.mean(y_train == y_train_pred)
            y_val_pred = linear_method.predict(X_val)
            acc_val = np.mean(y_val == y_val_pred)

            results[(lr, rs)] = (acc_train, acc_val)

            if best_val < acc_val:
                best_val = acc_val
                best_linear_method = linear_method

                # Print out results.


    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
            lr, reg, train_accuracy, val_accuracy))

    print('%s -- best validation accuracy achieved during cross-validation: %f' % (method.name(), best_val))

    return best_linear_method


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    tmp = np.loadtxt("iris.csv", dtype=np.str, delimiter=",")
    lst = list(range(tmp.shape[0]))
    shuffle(lst)

    percent = 0.7
    num_train_all = int(lst.__len__() * percent)
    num_test_all = lst.__len__() - int(lst.__len__() * percent)
    vali_percent = 0.2
    num_training = int(num_train_all * (1 - vali_percent))
    num_validation = int(num_train_all *  vali_percent)
    num_test = num_test_all
    num_dev = 50
    mask = range(num_training, num_training + num_validation)

    # build the dataset
    X_train = tmp[lst[: int(lst.__len__() * percent)],1:-1].astype(np.float)#加载数据部分
    X_train =  preprocessing.scale(X_train)
    y_train = tmp[lst[: int(lst.__len__() * percent)],-1].astype(np.int)#加载类别标签部分
    X_test = tmp[lst[int(lst.__len__() * percent):],1:-1].astype(np.float)
    X_test = preprocessing.scale(X_test)
    y_test = tmp[lst[int(lst.__len__() * percent):],-1].astype(np.int)


    # Our validation set will be num_validation points from the original
    # training set.
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    # We will also make a development set, which is a small subset of
    # the training set.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # We use the first num_test points of the original test set as our
    # test set.
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    # minus mean
    mean_value = np.mean(X_train, axis=0)
    X_train -= mean_value
    X_val -= mean_value
    X_test -= mean_value
    X_dev -= mean_value

    best_ML = achieve_best(LinearML())
    best_SVM = achieve_best(Softmax())
    loss_hist = Softmax().train(X_train, y_train, learning_rate=4e-5, reg=2e1,batch_size=30,
                          num_iters=1500, verbose=True)
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()
