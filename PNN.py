# -*- coding:utf-8 -*-
from keras.layers import Input, Multiply, Reshape, Concatenate, Dense, Activation, Embedding
from keras.models import Model

from utils.layers import pair_wise_interact_layer


def pnn(index_num, feat_num, embed_size):
    """
    The PNN CTR prediction model

    :param index_num: the number of indices, which is used to encode the features
    :param feat_num: the number of features of each sample
    :param embed_size: the embedding size of each feature
    :return: the Keras Model object
    """
    # 1.1 Embedding feat index part (batch_size, feat_num, embed_size)
    feat_index = Input(shape=(feat_num,), name='feat_index')
    embed_feat_index = Embedding(input_dim=index_num, output_dim=embed_size, input_length=feat_num,
                                 embeddings_initializer='uniform', embeddings_regularizer=None,
                                 name='embed_feat_index_layer')(feat_index)

    # 1.2 Preprocess the feat value (batch_size, feat_num, 1)
    feat_value = Input(shape=(feat_num,), name='feat_value')
    reshape_input_value = Reshape(target_shape=(feat_num, 1), name='reshape_feat_value_layer')(feat_value)

    # 2. Combine the embedded index and the fixed feat values (batch_size, feat_num, embed_size)
    embed_output = Multiply(name='embed_output_layer')([embed_feat_index, reshape_input_value])

    # 3.1 First order part (batch_size, feat_num * embed_size)
    first_order_output = Reshape(target_shape=(feat_num * embed_size,), name='first_order_output_layer')(embed_output)

    # 3.2 Product Layer (batch_size, pairs), pairs = (feat_num * (feat_num-1))/2
    second_order_output = pair_wise_interact_layer(name='product_layer')(embed_output)

    # 4. Concatenate the first and second order input (batch_size, feat_num * embed_size + pairs)
    mlp_input = Concatenate(axis=-1, name='mlp_input_layer')([first_order_output, second_order_output])

    # 5. Multi-layer perception
    # ------------------------------------------------------------------------
    z1 = Dense(units=200, use_bias=True, name='dense_layer1',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None)(mlp_input)
    a1 = Activation('relu', name='activation_layer1')(z1)
    # ------------------------------------------------------------------------
    z2 = Dense(units=200, use_bias=True, name='dense_layer2',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None)(a1)
    a2 = Activation('relu', name='activation_layer2')(z2)
    # ------------------------------------------------------------------------
    z3 = Dense(units=1, use_bias=True, name='output_dense_layer',
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None)(a2)
    outputs = Activation('sigmoid', name='output_activation_layer')(z3)
    # ------------------------------------------------------------------------

    # 6. Build Model
    model = Model(inputs=[feat_index, feat_value], outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy')

    return model
