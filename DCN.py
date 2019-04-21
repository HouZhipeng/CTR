# -*- coding:utf-8 -*-

from keras.layers import Input, Reshape, Embedding, Dense, Activation, Concatenate
from keras.models import Model

from utils.layers import CrossLayer


def dcn(dense_feat_num, sparse_feat_num, sparse_index_num, embed_size, cross_layer_num):
    """
    The DCN CTR prediction model

    :param dense_feat_num: the number of dense features of each sample
    :param sparse_feat_num: the number of sparse features of each sample
    :param sparse_index_num: the number of indices, which is used to encode the sparse features
    :param embed_size: the embedding size of sparse features
    :param cross_layer_num: the number of layer used in cross-net part
    :return: the Keras Model object
    """

    # Model input (batch_size, dense_feat_num + sparse_feat_num * embed_size)
    # =============================================================================================
    # Dense features (batch_size, dense_feat_num)
    dense_feat_value = Input(shape=(dense_feat_num,), name='dense_feat_value')

    # Sparse features embedding part (batch_size, sparse_feat_num * embed_size)
    sparse_feat_index = Input(shape=(sparse_feat_num,), name='sparse_feat_index')
    embed_sparse_feat_index = Embedding(input_dim=sparse_index_num, output_dim=embed_size, input_length=sparse_feat_num,
                                        embeddings_initializer='uniform', embeddings_regularizer=None,
                                        name='embed_sparse_feat_index_layer')(sparse_feat_index)
    reshape_embed_sparse_feat_index = Reshape(target_shape=(sparse_feat_num * embed_size,),
                                              name='reshape_embed_sparse_feat_index')(embed_sparse_feat_index)

    # Concatenate dense part and sparse part
    model_input = Concatenate(axis=-1, name='model_input_layer')([dense_feat_value, reshape_embed_sparse_feat_index])
    # =============================================================================================

    # Cross network part (batch_size, cross_net_units)
    # =============================================================================================
    cross_output = model_input
    for layer_idx in range(1, cross_layer_num + 1):
        # initial input & last output
        cross_output = CrossLayer(name='cross_layer' + str(layer_idx),
                                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                  kernel_regularizer=None, bias_regularizer=None)([model_input, cross_output])
    # =============================================================================================

    # Deep part (batch_size, last_dense_units)
    # =============================================================================================
    z1 = Dense(units=200, use_bias=True, name='dense_layer1',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None)(model_input)
    a1 = Activation('relu', name='activation_layer1')(z1)
    # ------------------------------------------------------------------------
    z2 = Dense(units=200, use_bias=True, name='dense_layer2',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None)(a1)
    a2 = Activation('relu', name='activation_layer2')(z2)
    # ------------------------------------------------------------------------
    z3 = Dense(units=200, use_bias=True, name='deep_output_dense_layer',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None)(a2)
    deep_output = Activation('relu', name='deep_output_activation_layer')(z3)
    # =============================================================================================

    # Final output
    # =============================================================================================
    cat_cross_deep = Concatenate(axis=-1, name='final_input_layer')([cross_output, deep_output])
    final_z = Dense(units=1, use_bias=True, name='final_output_dense_layer',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None)(cat_cross_deep)
    outputs = Activation('sigmoid', name='final_output_activation_layer')(final_z)
    # =============================================================================================

    # Build model
    # =============================================================================================
    model = Model(inputs=[dense_feat_value, sparse_feat_index], outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model
    # =============================================================================================
