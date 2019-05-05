# -*- coding:utf-8 -*-
from keras.layers import Input, Embedding, Concatenate
from keras.models import Model

from utils.layer import DenseEmbedding


def embedding(sparse_field_num, sparse_index_num, dense_field_num, embed_size=8,
              embeddings_initializer='uniform', embeddings_regularizer=None):
    """
    An implementation of Embedding layer.

    :param sparse_field_num: The number of sparse field
    :param sparse_index_num: The total number index used to encode sparse features in all sparse field
    :param dense_field_num: The number of dense field
    :param embed_size: The embedding size
    :param embeddings_initializer: The initializer used to initialize kernels in embedding layer
    :param embeddings_regularizer: The regularizer used in embedding layer
    :return: A keras Model object
    """

    # 1.1 Sparse features embedding part => (batch_size, sparse_field_num, embed_size)
    # =============================================================================================
    sparse_feat_index = Input(shape=(sparse_field_num,), name='sparse_feat_index')
    embed_sparse_feat_index = Embedding(input_dim=sparse_index_num, output_dim=embed_size,
                                        embeddings_initializer=embeddings_initializer,
                                        embeddings_regularizer=embeddings_regularizer,
                                        name='embed_sparse_feat_index_layer'
                                        )(sparse_feat_index)

    # 1.2 Dense features embedding part => (batch_size, dense_field_num, embed_size)
    # =============================================================================================
    dense_feat_value = Input(shape=(dense_field_num,), name='dense_feat_value')
    embed_dense_feat_value = DenseEmbedding(embed_size=embed_size,
                                            embeddings_initializer=embeddings_initializer,
                                            embeddings_regularizer=embeddings_regularizer,
                                            name='embed_dense_feat_value_layer'
                                            )(dense_feat_value)

    # 2. Concatenate embedded sparse index and embedded dense value
    # =============================================================================================
    embed_output = Concatenate(axis=1, name='embed_output_layer')([embed_sparse_feat_index,
                                                                   embed_dense_feat_value])

    # 3. Build Model
    # =============================================================================================
    model = Model(inputs=[sparse_feat_index, dense_feat_value], outputs=embed_output)
    model.compile(optimizer='sgd', loss='mse')

    return model
