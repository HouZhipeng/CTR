# -*- coding:utf-8 -*-
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


class AucCallback(Callback):

    def __init__(self, train_generator=None, valid_generator=None):
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.history = {}
        if train_generator:
            self.history['auc'] = []
        if valid_generator:
            self.history['valid_auc'] = []
        super(AucCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        # TODO: 看一下self.validation_data在generator 的时候是什么值
        logs = logs or {}
        if self.train_generator:
            train_true = []
            train_pred = []
            for index in range(len(self.train_generator)):
                batch_x, batch_y = self.train_generator[index]
                train_pred.append(self.model.predict_on_batch(batch_x))
                train_true.append(batch_y)
            train_pred = np.vstack(train_pred)
            train_true = np.vstack(train_true)
            train_auc = roc_auc_score(train_true, train_pred)
            self.history['auc'].append(train_auc)
            logs['auc'] = train_auc
            print('Train auc: %s' % str(round(train_auc, 4)), end=' - ')

        if self.valid_generator:
            valid_true = []
            valid_pred = []
            for index in range(len(self.valid_generator)):
                batch_x, batch_y = self.valid_generator[index]
                valid_pred.append(self.model.predict_on_batch(batch_x))
                valid_true.append(batch_y)
            valid_pred = np.vstack(valid_pred)
            valid_true = np.vstack(valid_true)
            valid_auc = roc_auc_score(valid_true, valid_pred)
            self.history['valid_auc'].append(valid_auc)
            logs['valid_auc'] = valid_auc
            print('Valid auc: %s' % str(round(valid_auc, 4)))
        print('-' * 80)
