# -*- coding:utf-8 -*-
from keras.layers import Input, Concatenate, Dense, Activation, Embedding, Flatten
from keras.models import Model

from utils.layer import inner_product_layer, DenseEmbedding, OutputLayer


def pnn(sparse_field_num, sparse_index_num, dense_field_num, embed_size=8,
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
    # 3. Combine layer
    # =============================================================================================
    # 3.1 First order part => (batch_size, field_num * embed_size)
    first_order_output = Flatten(name='first_order_output_layer')(embed_output)

    # 3.2 Product Layer => (batch_size, pairs), pairs = (field_num * (field_num-1))/2
    second_order_output = inner_product_layer(name='product_layer')(embed_output)

    # 3.3 Concatenate the first and second order output => (batch_size, field_num * embed_size + pairs)
    mlp_input = Concatenate(axis=-1, name='mlp_input_layer')([first_order_output, second_order_output])

    # 4. Multi-layer perception => (batch_size, 1)
    # =============================================================================================
    active_out = mlp_input
    for idx, units in enumerate(mlp_units):
        kernel_out = Dense(units=units, use_bias=True, name='dense_layer' + str(idx + 1),
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_initializer=bias_initializer,
                           bias_regularizer=bias_regularizer
                           )(active_out)
        active_out = Activation(mlp_activation, name='activation_layer' + str(idx + 1))(kernel_out)

    mlp_output = Dense(units=1, use_bias=False, name='mlp_output_layer',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer
                       )(active_out)

    # 5. Output layer => (batch_size, 1)
    # =============================================================================================
    outputs = OutputLayer(activation=output_activation, use_bias=output_use_bias,
                          bias_initializer=bias_initializer,
                          bias_regularizer=bias_regularizer,
                          name='output_layer'
                          )(mlp_output)

    # 6. Build Model
    # =============================================================================================
    model = Model(inputs=[sparse_feat_index, dense_feat_value], outputs=outputs)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mse'])

    return model
