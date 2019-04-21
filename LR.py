# -*- coding:utf-8 -*-
import numpy as np


class LR(object):
    """This class defines the standard logistic regression algorithm with logloss,
    which is optimized by gradient decent"""

    def __init__(self):
        self.w = None  # the weight vector, with the shape of (n_feature, 1)
        self.b = None  # the bias
        self.costs = []  # the cost after each iteration
        self.learn_rate = 0.001  # the learning rate used in gradient decent algorithm
        self.n_features = None  # the number of features
        self.batch_size = None  # if the batch size is given, the stochastic batch gradient decent is used

    def _sigmoid(self, x):
        """
        the sigmoid function y=1/(1+exp(-x))

        :param x: a scale value or numpy vector
        :return: the value calculated by sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def _propagate(self, x):
        """
        the forward propagate processing

        :param x: the input vector or matrix, of which each row is a sample.
                  The shape is (n_sample, n_feature) or (batch_size, n_feature) if batch_size is used
        :return: the predict value. (n_sample, 1) or (batch_size, 1)
        """
        return self._sigmoid(np.dot(x, self.w) + self.b)

    def _back_propagate(self, x, pred_y, true_y):
        """
        the back prpogate processing

        :param x: the input vector or matrix, of which each row is a sample.
                  The shape is (n_sample, n_feature) or (batch_size, n_feature) if batch_size is used
        :param pred_y: the predict value. (n_sample, 1) or (batch_size, 1)
        :param true_y: the true value. (n_sample, 1) or (batch_size, 1)
        """
        dw = np.dot(x.T, pred_y - true_y) / x.shape[0]
        db = np.average(pred_y - true_y)
        self.w = self.w - self.learn_rate * dw
        self.b = self.b - self.learn_rate * db

    def _train(self, x, y):
        """
        the forward propagate and back propagate processing

        :param x: the input vector or matrix, of which each row is a sample.
                  The shape is (n_sample, n_feature) or (batch_size, n_feature) if batch_size is used
        :param y: the true value. (n_sample, 1) or (batch_size, 1)
        :return: the logistic cost
        """
        pred_y = self._propagate(x)
        self._back_propagate(x, pred_y, y)
        return self._logloss(pred_y, y)

    def fit(self, x_train, y_train, iter_num, learn_rate, batch_size=None, print_log=False):
        """
        The training process

        :param x_train: a numpy array with the shape of (n_sample * n_feature), each row is a single training sample
        :param y_train: a vector containing the target value of each sample. (n_sample, 1)
        :param iter_num: the iteration number
        :param learn_rate: the learning rate used in gradient decent algorithm
        :param batch_size: the stochastic batch gradient decent is used, if the batch size is given
        :param print_log: weather print the log or not
        :return: a list containing the cost of each iteration
        """
        n_samples, self.n_features = x_train.shape
        assert y_train.shape == (n_samples, 1)

        self.w = np.zeros((self.n_features, 1))
        self.b = 0.
        self.learn_rate = learn_rate

        if batch_size:
            self.batch_size = batch_size
            temp_cost = []
            index = [i for i in range(0, n_samples, batch_size)]
            if n_samples % batch_size != 0:
                index.append(n_samples)

            for i in range(iter_num):
                temp_cost.clear()
                for idx in range(1, len(index)):
                    idx1 = index[idx - 1]
                    idx2 = index[idx]
                    temp_cost.append(self._train(x_train[idx1:idx2, :], y_train[idx1:idx2, :]))
                self.costs.append(np.average(temp_cost))
                if print_log:
                    if (i + 1) % 100 == 0:
                        print("Iteration", i + 1, ", the cost is:", self.costs[i])
        else:
            for i in range(iter_num):
                self.costs.append(np.average(self._train(x_train, y_train)))
                if print_log:
                    if (i + 1) % 100 == 0:
                        print("Iteration", i + 1, ", the cost is:", self.costs[i])

        return self.costs

    def predict(self, x_test):
        """
        predict the value

        :param x_test: the given samples
        :return: the predict value
        """
        return self._propagate(x_test)

    def _logloss(self, pred_y, true_y):
        """
        calculate the logloss

        :param pred_y: the predict value
        :param true_y: the true value
        :return: the logloss
        """
        assert pred_y.shape == true_y.shape
        return np.average(-true_y * np.log(pred_y) - (1 - true_y) * np.log(1 - pred_y))
