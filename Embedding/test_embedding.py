# -*- coding:utf-8 -*-

import numpy as np

from Embedding.model import embedding

sparse_field_num = 3
dense_field_num = 4
sparse_index_num = 100
sparse_feat_index = np.random.choice(range(sparse_index_num), (10, sparse_field_num))
dense_feat_value = np.ones((10, dense_field_num))

model = embedding(sparse_field_num, sparse_index_num, dense_field_num)
output = model.predict([sparse_feat_index, dense_feat_value])
