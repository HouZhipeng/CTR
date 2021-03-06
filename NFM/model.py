# -*- coding:utf-8 -*-
from keras.layers import Input, Embedding, Dense, Activation, Concatenate
from keras.models import Model

from utils.layer import OutputLayer, fm_layer, sum_layer, DenseEmbedding


def nfm(sparse_field_num, sparse_index_num, dense_field_num, embed_size=8,
        embeddings_initializer='uniform', embeddings_regularizer=None,
        mlp_units=(100, 100), mlp_activation='relu',
        kernel_initializer='glorot_uniform', kernel_regularizer=None,
        bias_initializer='zeros', bias_regularizer=None,
        output_use_bias=True, output_activation=None):
    """
    An implementation of NFM model in CTR problem.

    :param sparse_field_num: The number of sparse field
    :param sparse_index_num: The total number index used to encode sparse features in all sparse field
    :param dense_field_num: The number of dense field
    :param embed_size: The embedding size
    :param embeddings_initializer: The initializer used to initialize kernels in embedding layer
    :param embeddings_regularizer: The regularizer used in embedding layer
    :param mlp_units: The number of hidden units used in each hidden layer.
                      The length of this list is equal to the number of hidden layers
    :param mlp_activation: The activate function used in multi-layer perception part
    :param kernel_initializer: The initializer used to initialize kernel
    :param kernel_regularizer: The regularizer used on kernel
    :param bias_initializer: The initializer used to initialize bias
    :param bias_regularizer: The regularizer used on bias
    :param output_use_bias: In output layer, whether use bias or not.
    :param output_activation: The activation function used in output layer
    :return: A keras Model object
    """

    # 1. Inputs
    # =============================================================================================
    sparse_feat_index = Input(shape=(sparse_field_num,), name='sparse_feat_index')
    dense_feat_value = Input(shape=(dense_field_num,), name='dense_feat_value')

    # 2. LR part => (batch_size, 1)
    # =============================================================================================
    # 2.1 Sparse features linear sum part => (batch_size, 1)
    # (batch_size, sparse_field_num, 1)
    sparse_feat_weight = Embedding(input_dim=sparse_index_num, output_dim=1,
                                   embeddings_initializer=kernel_initializer,
                                   embeddings_regularizer=kernel_regularizer,
                                   name='sparse_feat_weight_layer'
                                   )(sparse_feat_index)
    # (batch_size, 1)
    sparse_lr_out = sum_layer(axis=1, name='sparse_lr_out_layer')(sparse_feat_weight)

    # 2.2 Dense features embedding part => (batch_size, 1)
    # (batch_size, dense_field_num, 1)
    dense_lr_out = Dense(units=1, activation=None, use_bias=False,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         name='dense_lr_out_layer'
                         )(dense_feat_value)

    # 3. Embedding Layer => (batch_size, sparse_field_num + dense_field_num, embed_size)
    # =============================================================================================
    # 3.1 Sparse features embedding part => (batch_size, sparse_field_num, embed_size)
    embed_sparse_feat_index = Embedding(input_dim=sparse_index_num, output_dim=embed_size,
                                        embeddings_initializer=embeddings_initializer,
                                        embeddings_regularizer=embeddings_regularizer,
                                        name='embed_sparse_feat_index_layer'
                                        )(sparse_feat_index)

    # 3.2 Dense features embedding part => (batch_size, dense_field_num, embed_size)
    embed_dense_feat_value = DenseEmbedding(embed_size=embed_size,
                                            embeddings_initializer=embeddings_initializer,
                                            embeddings_regularizer=embeddings_regularizer,
                                            name='embed_dense_feat_value_layer'
                                            )(dense_feat_value)

    # 3.3 Concatenate embedded sparse index and embedded dense value
    embed_output = Concatenate(axis=1, name='embed_output_layer')([embed_sparse_feat_index,
                                                                   embed_dense_feat_value])

    # 4. NFM part => (batch_size, 1)
    # =============================================================================================
    # 4.1 FM layer => (batch_size, embed_size)
    fm_output = fm_layer(name='fm_layer')(embed_output)

    # 4.2 Multi-layer perception part => (batch_size, 1)
    active_out = fm_output
    for idx, units in enumerate(mlp_units):
        kernel_out = Dense(units=units, use_bias=True, name='dense_layer' + str(idx + 1),
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_initializer=bias_initializer,
                           bias_regularizer=bias_regularizer
                           )(active_out)
        active_out = Activation(mlp_activation, name='activation_layer' + str(idx + 1))(kernel_out)

    nfm_output = Dense(units=1, use_bias=False, name='nfm_output_layer',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer
                       )(active_out)

    # 5. Output layer => (batch_size, 1)
    # =============================================================================================
    outputs = OutputLayer(activation=output_activation, use_bias=output_use_bias,
                          bias_initializer=bias_initializer,
                          bias_regularizer=bias_regularizer,
                          name='output_layer'
                          )([sparse_lr_out, dense_lr_out, nfm_output])

    # 6. Build Model
    # =============================================================================================
    model = Model(inputs=[sparse_feat_index, dense_feat_value], outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mse'])

    return model
