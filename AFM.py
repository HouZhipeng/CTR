# -*- coding:utf-8 -*-
from keras.layers import Input, Multiply, Reshape, Embedding, Dense, Activation, Add, Softmax
from keras.models import Model

from utils.layers import AddBiasLayer, linear_layer, pair_wise_interact_layer, axis_sum_layer


def afm(index_num, feat_num, embed_size, attn_fact_num):
    """
    The AFM CTR prediction model

    :param index_num: the number of indices, which is used to encode the features
    :param feat_num: the number of features of each sample
    :param embed_size: the embedding size of each feature
    :param attn_fact_num: the number of latent factors used in attention network
    :return: the Keras Model object
    """

    # Model input
    # =============================================================================================
    feat_index = Input(shape=(feat_num,), name='feat_index')  # (batch_size, feat_num)

    feat_value = Input(shape=(feat_num,), name='feat_value')  # (batch_size, feat_num)
    # Preprocess the feat value to the shape (batch_size, feat_num, 1)
    reshape_input_value = Reshape(target_shape=(feat_num, 1), name='reshape_feat_value_layer')(feat_value)
    # =============================================================================================

    # Linear part
    # =============================================================================================
    # 1. Get the weight of each feature for each samples
    linear_weight = Embedding(input_dim=index_num, output_dim=1, input_length=feat_num,
                              embeddings_initializer='uniform', embeddings_regularizer=None,
                              name='linear_weight_layer')(feat_index)  # (batch_size, feat_num, 1)

    # 2. Get linear part output (batch_size, 1)
    linear_output = linear_layer(name='linear_output_layer')([linear_weight, reshape_input_value])
    # =============================================================================================

    # AFM part
    # =============================================================================================
    # 1. Embedding feat index (batch_size, feat_num, embed_size)
    embed_feat_index = Embedding(input_dim=index_num, output_dim=embed_size, input_length=feat_num,
                                 embeddings_initializer='uniform', embeddings_regularizer=None,
                                 name='embed_feat_index_layer')(feat_index)

    # 2. Combine the embedded index and the fixed feat values (batch_size, feat_num, embed_size)
    embed_output = Multiply(name='embed_output_layer')([embed_feat_index, reshape_input_value])

    # 3. Pair-wise interaction layer (batch_size, pairs, embed_size), pairs = (feat_num * (feat_num-1))/2
    pair_wise_feat = pair_wise_interact_layer(keepdims=True, name='pair_wise_interact_layer')(embed_output)

    # 4. Attention weight part (batch_size, pairs, 1)
    # ------------------------------------------------------------------------
    # (batch_size, pairs, attn_fact_num)
    attn_z1 = Dense(units=attn_fact_num, use_bias=True, name='attn_dense_layer1',
                    kernel_initializer='glorot_uniform', bias_initializer='zeros',
                    kernel_regularizer=None, bias_regularizer=None)(pair_wise_feat)
    attn_a1 = Activation('relu', name='attn_activation_layer1')(attn_z1)
    # ------------------------------------------------------------------------
    # (batch_size, pairs, 1)
    attn_output = Dense(units=1, use_bias=False, name='attn_output_dense_layer',
                        kernel_initializer='glorot_uniform', kernel_regularizer=None)(attn_a1)
    attn_weight = Softmax(axis=1, name='attn_weight_layer')(attn_output)
    # ------------------------------------------------------------------------

    # 5. Attention based pooling (batch_size, embed)
    pool_input = Multiply(name='pooling_input_layer')([pair_wise_feat, attn_weight])
    pool_output = axis_sum_layer(1, name='pooling_output_layer')(pool_input)

    # 6. AFM output
    afm_output = Dense(units=1, use_bias=False, name='afm_output_layer',
                       kernel_initializer='glorot_uniform', kernel_regularizer=None)(pool_output)
    # =============================================================================================

    # Final output
    # =============================================================================================
    outputs = AddBiasLayer(activation='sigmoid', name='output_layer',
                           initializer='zeros', regularizer=None)(
        Add(name='add_linear_afm_layer')([linear_output, afm_output])
    )
    # =============================================================================================

    # Build model
    # =============================================================================================
    model = Model(inputs=[feat_index, feat_value], outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model
    # =============================================================================================
