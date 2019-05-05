# -*- coding:utf-8 -*-
from keras.layers import Input, Embedding, Dense
from keras.models import Model

from utils.layer import sum_layer, OutputLayer


def lr(sparse_field_num, sparse_index_num, dense_field_num,
       kernel_initializer='glorot_uniform', kernel_regularizer=None,
       bias_initializer='zeros', bias_regularizer=None,
       output_use_bias=True, output_activation=None):
    """
    An implementation of Linear Regression.

    :param sparse_field_num: The number of sparse field
    :param sparse_index_num: The total number index used to encode sparse features in all sparse field
    :param dense_field_num: The number of dense field
    :param kernel_initializer: The initializer used to initialize kernel
    :param kernel_regularizer: The regularizer used on kernel
    :param bias_initializer: The initializer used to initialize bias
    :param bias_regularizer: The regularizer used on bias
    :param output_use_bias: In output layer, whether use bias or not.
    :param output_activation: The activation function used in output layer
    :return: A keras Model object
    """

    # 1. Sparse features linear sum part => (batch_size, 1)
    # =============================================================================================
    sparse_feat_index = Input(shape=(sparse_field_num,), name='sparse_feat_index')
    # (batch_size, sparse_field_num, 1)
    sparse_feat_weight = Embedding(input_dim=sparse_index_num, output_dim=1,
                                   embeddings_initializer=kernel_initializer,
                                   embeddings_regularizer=kernel_regularizer,
                                   name='sparse_feat_weight_layer'
                                   )(sparse_feat_index)
    # (batch_size, 1)
    sparse_lr_out = sum_layer(axis=1, name='sparse_lr_out_layer')(sparse_feat_weight)

    # 2. Dense features linear sum part => (batch_size, 1)
    # =============================================================================================
    dense_feat_value = Input(shape=(dense_field_num,), name='dense_feat_value')
    # (batch_size, dense_field_num, 1)
    dense_lr_out = Dense(units=1, activation=None, use_bias=False,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         name='dense_lr_out_layer'
                         )(dense_feat_value)

    # 3. Output layer => (batch_size, 1)
    # =============================================================================================
    outputs = OutputLayer(activation=output_activation, use_bias=output_use_bias,
                          bias_initializer=bias_initializer,
                          bias_regularizer=bias_regularizer,
                          name='output_layer'
                          )([sparse_lr_out, dense_lr_out])

    # 4. Build Model
    # =============================================================================================
    model = Model(inputs=[sparse_feat_index, dense_feat_value], outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mse'])

    return model
