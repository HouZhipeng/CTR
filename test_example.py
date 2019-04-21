# -*- coding:utf-8 -*-
import numpy as np

from AFM import afm
from DCN import dcn
from DNN import dnn
from DeepFM import deep_fm
from NFM import nfm
from PNN import pnn

sparse_index = np.array([
    [0, 3],
    [1, 4],
    [2, 3],
    [0, 5],
    [2, 6]
])

sparse_value = np.array([
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0],
    [1.0, 1.0]
])

sparse_feat_num = 2
sparse_index_num = 7

# -------------------------------

dense_index = np.array([
    [7, 8, 9],
    [7, 8, 9],
    [7, 8, 9],
    [7, 8, 9],
    [7, 8, 9],
])

dense_value = np.array([
    [3.1, 2.2, 0],
    [2.1, 3.1, 0],
    [1.0, 3.4, 0],
    [2.1, 1.6, 0],
    [0.5, 1.8, 0]
])

dense_feat_num = 3
dense_index_num = 3

# -------------------------------

embed_size = 3
attn_fact_num = 4
cross_layer_num = 2
feat_num = sparse_feat_num + dense_feat_num
index_num = sparse_index_num + dense_index_num
input_index = np.hstack((sparse_index, dense_index))
input_value = np.hstack((sparse_value, dense_value))

# -------------------------------

model = dcn(dense_feat_num, sparse_feat_num, sparse_index_num, embed_size, cross_layer_num)
# result = model.predict([input_index, input_value])
result = model.predict([dense_value, sparse_index])

print(result)
