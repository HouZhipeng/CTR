# -*- coding:utf-8 -*-
from keras.layers import Input, Multiply, Embedding, Dense, Activation, Softmax, Concatenate
from keras.models import Model

from utils.layer import OutputLayer, inner_product_layer, sum_layer, DenseEmbedding


def afm(sparse_field_num, sparse_index_num, dense_field_num, embed_size=8,
        embeddings_initializer='uniform', embeddings_regularizer=None,
        attn_fact_num=8, kernel_initializer='glorot_uniform', kernel_regularizer=None,
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
    :param attn_fact_num: The number of latent factories used in attention network part
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

    # 4. Pair-wise interaction => (batch_size, pairs, embed_size), pairs = (field_num * (field_num-1))/2
    # =============================================================================================
    pair_wise_feat = inner_product_layer(keepdims=True, name='inner_product_layer')(embed_output)

    # 5. Attention weight part => (batch_size, pairs, 1)
    # =============================================================================================
    # 5.1 (batch_size, pairs, attn_fact_num)
    attn_z1 = Dense(units=attn_fact_num, use_bias=True, name='attn_dense_layer1',
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_initializer=bias_initializer,
                    bias_regularizer=bias_regularizer
                    )(pair_wise_feat)
    attn_a1 = Activation('relu', name='attn_activation_layer1')(attn_z1)

    # 5.2 (batch_size, pairs, 1)
    attn_z2 = Dense(units=1, use_bias=False, name='attn_dense_layer2',
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer
                    )(attn_a1)
    attn_weight = Softmax(axis=1, name='attn_output_layer')(attn_z2)

    # 6. Attention based pooling => (batch_size, 1)
    # =============================================================================================
    # (batch_size, pairs, embed_size)
    pool_input = Multiply(name='pooling_input_layer')([pair_wise_feat, attn_weight])
    # (batch_size, embed_size)
    pool_output = sum_layer(1, name='pooling_output_layer')(pool_input)
    # (batch_size, 1)
    afm_output = Dense(units=1, use_bias=False, name='afm_output_layer',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer
                       )(pool_output)

    # 7. Output layer => (batch_size, 1)
    # =============================================================================================
    outputs = OutputLayer(activation=output_activation, use_bias=output_use_bias,
                          bias_initializer=bias_initializer,
                          bias_regularizer=bias_regularizer,
                          name='output_layer'
                          )([sparse_lr_out, dense_lr_out, afm_output])

    # 6. Build Model
    # =============================================================================================
    model = Model(inputs=[sparse_feat_index, dense_feat_value], outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mse'])

    return model
