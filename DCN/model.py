# -*- coding:utf-8 -*-

from keras.layers import Input, Embedding, Dense, Activation, Concatenate, Flatten
from keras.models import Model

from utils.layer import CrossLayer, OutputLayer


def dcn(sparse_field_num, sparse_index_num, dense_field_num, embed_size=8,
        embeddings_initializer='uniform', embeddings_regularizer=None,
        cross_layer_num=2, mlp_units=(100, 100), mlp_activation='relu',
        kernel_initializer='glorot_uniform', kernel_regularizer=None,
        bias_initializer='zeros', bias_regularizer=None,
        output_use_bias=True, output_activation=None):
    """
    An implementation of DCN model in CTR problem.

    :param sparse_field_num: The number of sparse field
    :param sparse_index_num: The total number index used to encode sparse features in all sparse field
    :param dense_field_num: The number of dense field
    :param embed_size: The embedding size
    :param embeddings_initializer: The initializer used to initialize kernels in embedding layer
    :param embeddings_regularizer: The regularizer used in embedding layer
    :param cross_layer_num: The number of cross layers
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

    # 1. Inputs => (batch_size, sparse_field_num * embed_size + dense_field_num)
    # =============================================================================================
    # 1.1 Embedded the sparse features => (batch_size, sparse_field_num, embed_size)
    sparse_feat_index = Input(shape=(sparse_field_num,), name='sparse_feat_index')
    embed_sparse_feat_index = Embedding(input_dim=sparse_index_num, output_dim=embed_size,
                                        embeddings_initializer=embeddings_initializer,
                                        embeddings_regularizer=embeddings_regularizer,
                                        name='embed_sparse_feat_index_layer'
                                        )(sparse_feat_index)
    # 1.2 Flatten the embedded sparse features
    embed_sparse_feat_index = Flatten(name='flatten_embed_sparse_feat_layer')(embed_sparse_feat_index)
    # 1.3 Dense features
    dense_feat_value = Input(shape=(dense_field_num,), name='dense_feat_value')

    # 1.4 Concatenate sparse part and dense part
    model_input = Concatenate(axis=-1, name='model_input_layer')([embed_sparse_feat_index, dense_feat_value])

    # 2. Cross network part (batch_size, cross_net_unit_num)
    # =============================================================================================
    cross_output = model_input
    for layer_idx in range(cross_layer_num):
        # initial input & last output
        cross_output = CrossLayer(name='cross_layer' + str(layer_idx + 1),
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer,
                                  bias_initializer=bias_initializer,
                                  bias_regularizer=bias_regularizer
                                  )([model_input, cross_output])

    # 3. Multi-layer perception => (batch_size, last_hidden_unit_num)
    # =============================================================================================
    active_out = model_input
    for idx, units in enumerate(mlp_units):
        kernel_out = Dense(units=units, use_bias=True, name='dense_layer' + str(idx + 1),
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_initializer=bias_initializer,
                           bias_regularizer=bias_regularizer
                           )(active_out)
        active_out = Activation(mlp_activation, name='activation_layer' + str(idx + 1))(kernel_out)
    mlp_output = active_out

    # 4. Output layer => (batch_size, 1)
    # =============================================================================================
    final_input = Concatenate(name='final_input_layer')([cross_output, mlp_output])
    outputs = Dense(units=1, activation=output_activation, use_bias=output_use_bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_initializer=bias_initializer,
                    bias_regularizer=bias_regularizer,
                    name='final_output_layer'
                    )(final_input)

    # 6. Build Model
    # =============================================================================================
    model = Model(inputs=[sparse_feat_index, dense_feat_value], outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mse'])

    return model
