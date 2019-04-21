# -*- coding:utf-8 -*-

from keras.layers import Input, Multiply, Reshape, Embedding, Dense, Activation, Add
from keras.models import Model

from utils.layers import linear_layer, pair_wise_interact_layer, axis_sum_layer


def deep_fm(index_num, feat_num, embed_size):
    """
    The DeepFM CTR prediction model

    :param index_num: the number of indices, which is used to encode the features
    :param feat_num: the number of features of each sample
    :param embed_size: the embedding size of each feature
    :return: the Keras Model object
    """

    # Model input
    # =============================================================================================
    feat_index = Input(shape=(feat_num,), name='feat_index')  # (batch_size, feat_num)

    feat_value = Input(shape=(feat_num,), name='feat_value')  # (batch_size, feat_num)
    # Preprocess the feat value to the shape (batch_size, feat_num, 1)
    reshape_input_value = Reshape(target_shape=(feat_num, 1), name='reshape_feat_value_layer')(feat_value)
    # =============================================================================================

    # Linear part (batch_size, 1)
    # =============================================================================================
    # 1. Get the weight of each feature for each samples
    linear_weight = Embedding(input_dim=index_num, output_dim=1, input_length=feat_num,
                              embeddings_initializer='uniform', embeddings_regularizer=None,
                              name='linear_weight_layer')(feat_index)  # (batch_size, feat_num, 1)
    # 2. Get linear part output
    linear_output = linear_layer(name='linear_output_layer')([linear_weight, reshape_input_value])
    # =============================================================================================

    # Embedding part (batch_size, feat_num, embed_size)
    # =============================================================================================
    # 1. Embedding feat index (batch_size, feat_num, embed_size)
    embed_feat_index = Embedding(input_dim=index_num, output_dim=embed_size, input_length=feat_num,
                                 embeddings_initializer='uniform', embeddings_regularizer=None,
                                 name='embed_feat_index_layer')(feat_index)

    # 2. Combine the embedded index and the fixed feat values (batch_size, feat_num, embed_size)
    embed_output = Multiply(name='embed_output_layer')([embed_feat_index, reshape_input_value])
    # =============================================================================================

    # FM cross features part (batch_size, 1)
    # =============================================================================================
    fm_cross_feat = pair_wise_interact_layer(name='fm_cross_feat_layer')(embed_output)
    fm_output = axis_sum_layer(1, keepdims=True, name='fm_output_layer')(fm_cross_feat)
    # =============================================================================================

    # Deep part (batch_size, 1)
    # =============================================================================================
    deep_input = Reshape(target_shape=(feat_num * embed_size,), name='deep_input_layer')(embed_output)
    # ------------------------------------------------------------------------
    z1 = Dense(units=200, use_bias=True, name='dense_layer1',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None)(deep_input)
    a1 = Activation('relu', name='activation_layer1')(z1)
    # ------------------------------------------------------------------------
    z2 = Dense(units=200, use_bias=True, name='dense_layer2',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None)(a1)
    a2 = Activation('relu', name='activation_layer2')(z2)
    # ------------------------------------------------------------------------
    z3 = Dense(units=200, use_bias=True, name='dense_layer3',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None)(a2)
    a3 = Activation('relu', name='activation_layer3')(z3)
    # ------------------------------------------------------------------------
    deep_output = Dense(units=1, use_bias=False, name='deep_output_layer',
                        kernel_initializer='glorot_uniform', kernel_regularizer=None)(a3)
    # =============================================================================================

    # Final output
    # =============================================================================================
    final_input = Add(name='final_input_layer')([linear_output, fm_output, deep_output])
    outputs = Activation('sigmoid', name='final_output_layer')(final_input)
    # =============================================================================================

    # Build model
    # =============================================================================================
    model = Model(inputs=[feat_index, feat_value], outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model
    # =============================================================================================
