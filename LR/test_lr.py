# -*- coding:utf-8 -*-
import os

import numpy as np
from keras.callbacks import TensorBoard
from sklearn.metrics import log_loss, roc_auc_score

from LR.model import lr
from utils.callback import AucCallback
from utils.data import fill_nan, min_max_scalar, sparse_feat_encode, train_valid_test_split, SparseIdxDenseValSeq

# Data information
# ===================================================================================================
# data file path
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
dataset_name = 'cirteo'
full_data_file_name = 'sample.txt'
full_data_file_path = None
train_data_file_path = None
valid_data_file_path = None
test_data_file_path = None
log_dir = os.path.join(os.path.dirname(os.getcwd()), 'learn_curve')

# dataset info
target_col = 0
sparse_field_cols = [idx for idx in range(14, 40)]
dense_field_cols = [idx for idx in range(1, 14)]
sample_num = 200
sep = '\t'
# ===================================================================================================


# Model params
# ===================================================================================================
embed_size = 10
batch_size = 8
epochs = 20
train_ratio = 0.8
valid_ratio = 0.1
# ===================================================================================================


# Preprocess the data and generate data provider
# ===================================================================================================
# clear learning curve log files
if os.path.exists(log_dir):
    for log_file in os.listdir(log_dir):
        os.remove(os.path.join(log_dir, log_file))

# if the train, valid, test data file isn't given, do data pre-processing.
if full_data_file_path is None or train_data_file_path is None or \
        valid_data_file_path is None or test_data_file_path is None:
    full_data_file_path = os.path.join(data_dir, dataset_name, full_data_file_name)
    # fill the missing value
    full_data_file_path = fill_nan(full_data_file_path, sparse_field_cols, dense_field_cols,
                                   sep=sep, sparse_fill_val=-1, dense_fill_val=0)
    # do min & mac scalar on dense fields
    full_data_file_path = min_max_scalar(full_data_file_path, dense_field_cols, sep=sep)
    # split the full dataset to train, valid and test part
    train_data_file_path, valid_data_file_path, test_data_file_path = \
        train_valid_test_split(full_data_file_path, sample_num, train_ratio, valid_ratio)

# do label encoder on sparse fields
sparse_feat_map, sparse_index_num = \
    sparse_feat_encode(full_data_file_path, sparse_field_cols, sep=sep)

# generate the Keras Sequence of train, valid, test dataset
train_seq = SparseIdxDenseValSeq(train_data_file_path, target_col, sparse_field_cols, sparse_feat_map,
                                 dense_field_cols, sep=sep, batch_size=batch_size)
valid_seq = SparseIdxDenseValSeq(valid_data_file_path, target_col, sparse_field_cols, sparse_feat_map,
                                 dense_field_cols, sep=sep, batch_size=batch_size)
test_seq = SparseIdxDenseValSeq(test_data_file_path, target_col, sparse_field_cols, sparse_feat_map,
                                dense_field_cols, sep=sep, batch_size=batch_size)

sparse_field_num = len(sparse_field_cols)
dense_field_num = len(dense_field_cols)

print('Preparing data finish.')
# ===================================================================================================


# Generate model and training it
# ===================================================================================================
auc_callback = AucCallback(train_generator=train_seq, valid_generator=valid_seq)
model = lr(sparse_field_num, sparse_index_num, dense_field_num,
           kernel_initializer='glorot_uniform', kernel_regularizer=None,
           bias_initializer='zeros', bias_regularizer=None,
           output_use_bias=True, output_activation='sigmoid')
history = model.fit_generator(generator=train_seq, epochs=epochs, validation_data=valid_seq,
                              workers=1, use_multiprocessing=False, shuffle=True, verbose=2,
                              callbacks=[auc_callback, TensorBoard(log_dir=log_dir)])
# ===================================================================================================


# Test the model
# ===================================================================================================
test_true = []
test_pred = []
for index in range(len(test_seq)):
    batch_x, batch_y = test_seq[index]
    test_pred.append(model.predict_on_batch(batch_x))
    test_true.append(batch_y)
test_pred = np.vstack(test_pred)
test_true = np.vstack(test_true)
test_loss = log_loss(test_true, test_pred)
test_auc = roc_auc_score(test_true, test_pred)
print('=' * 80)
print('Test loss: %.4f - Test auc: %.4f' % (test_loss, test_auc))
# ===================================================================================================


# Clear temp files
# ===================================================================================================
train_seq.clear()
valid_seq.clear()
test_seq.clear()
# ===================================================================================================
