# -*- coding:utf-8 -*-

from keras import backend as K
from keras.layers import Layer, initializers, regularizers, activations, Lambda


class DenseEmbedding(Layer):
    """
    Keras Layer object, which defines an embedding layer used to embed the dense input features.
    Each field is correlated with a embedding tensor. The output of this field
    is equal to the field value multiply with the embedding tensor.

    For detail:
    If the shape of input dense feature value is: (batch_size, field_num).
    The shape of embedding matrix is: (field_num, embed_size). Different with sparse feature index,
    lookup operator is no longer needed here. Because each dense field have only one embedding tensor.
    Then, the value of each field will multiply with its embedding tensor.
    The output shape will be: (batch_size, field_size, embed_size)
    """

    def __init__(self, embed_size, embeddings_initializer='uniform',
                 embeddings_regularizer=None, **kwargs):
        self.embed_size = embed_size
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        super(DenseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(name='embeddings',
                                          shape=(input_shape[-1], self.embed_size),
                                          initializer=self.embeddings_initializer,
                                          regularizer=self.embeddings_regularizer,
                                          trainable=True)
        super(DenseEmbedding, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return input_shape + (self.embed_size,)

    def call(self, inputs, **kwargs):
        return K.expand_dims(inputs) * self.embeddings

    def get_config(self):
        config = {'embed_size': self.embed_size,
                  'embeddings_initializer':
                      initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer':
                      regularizers.serialize(self.embeddings_regularizer)}
        base_config = super(DenseEmbedding, self).get_config()
        return dict(**base_config, **config)


class OutputLayer(Layer):
    """
    Keras Layer object, which defines the output layer.
    In this layer, the inputs can be a tensor or tensor list and the shape of each tensor should
    be: (batch_size, 1).
    This layer first calculates the sum of these tensor.
    If `use_bias` is True, a global bias will be added in this layer.
    If `activation` is not None, the activation function will be used on the final output.
    The shape of output is: (batch_size, 1)
    """

    def __init__(self, activation=None, use_bias=True,
                 bias_initializer='zeros', bias_regularizer=None, **kwargs):
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        super(OutputLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        shape_list = list(input_shapes) if isinstance(input_shapes, list) else [input_shapes]
        # all of the input have the same shape, (batch_size, 1)
        first_shape = shape_list[0]
        assert first_shape[-1] == 1 and len(first_shape) == 2
        for shape in shape_list:
            assert shape == first_shape
        # Create a trainable bias weight variable for this layer.
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(1,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        trainable=True)
        super(OutputLayer, self).build(input_shapes)

    def call(self, inputs, **kwargs):
        input_list = list(inputs) if isinstance(inputs, list) else [inputs]
        output = input_list[0]
        for idx in range(1, len(input_list)):
            output = output + input_list[idx]
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shapes):
        shape_list = list(input_shapes) if isinstance(input_shapes, list) else [input_shapes]
        return shape_list[0]

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        }
        base_config = super(OutputLayer, self).get_config()
        return dict(**base_config, **config)


class CrossLayer(Layer):
    """
    A Keras Layers object.
    In this layer, the cross-net is implemented, which is defined in DCN CTR prediction model
    The calculate equation is:
    `output = x0 * last_output.T * w + b + last_output`
    [Note]: the first input of this layer is the x0, the second input is the output of last layer.
    """

    def __init__(self, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        # check the shapes of inputs
        assert isinstance(input_shapes, list)
        x0_shape, xl_shape = input_shapes
        assert x0_shape == xl_shape and len(xl_shape) == 2
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shapes[0][1], 1),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shapes[0][1], 1),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    trainable=True)
        super(CrossLayer, self).build(input_shapes)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        x0, xl = inputs
        x0 = K.expand_dims(x0, axis=2)  # (batch_size, n, 1)
        xl_t = K.expand_dims(xl, axis=1)  # (batch_size, 1, n)
        xl = K.expand_dims(xl, axis=2)  # (batch_size, n, 1)
        # (batch_size, n, 1)
        output = K.dot(K.batch_dot(x0, xl_t, axes=(2, 1)), self.kernel) + self.bias + xl
        return K.squeeze(output, axis=2)

    def compute_output_shape(self, input_shapes):
        assert isinstance(input_shapes, list)
        return input_shapes[0]

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        }
        base_config = super(CrossLayer, self).get_config()
        return dict(**base_config, **config)


class CompressInteractLayer(Layer):
    """
    A Keras Layers object.
    In this layer, the compress-interact layer (CIN) is implemented, which is defined in xDeepFM CTR prediction model
    [Note]: the first input of this layer is the x0, the second input is the output of last layer.
    """

    def __init__(self, feature_map_num, kernel_initializer='glorot_uniform', kernel_regularizer=None, **kwargs):
        self.feature_map_num = feature_map_num
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        super(CompressInteractLayer, self).__init__(**kwargs)

    def build(self, input_shapes):
        # check the shapes of inputs
        assert isinstance(input_shapes, list)
        x0_shape, xk_shape = input_shapes
        assert len(x0_shape) == len(xk_shape) == 3 and x0_shape[2] == xk_shape[2]
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(x0_shape[1] * xk_shape[1], self.feature_map_num),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(CompressInteractLayer, self).build(input_shapes)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        x0, xk = inputs
        _, field_num, embed_size = K.int_shape(x0)
        _, h_k, embed_size = K.int_shape(xk)
        # (batch_size, embed_size, feat_num, 1)
        x0 = K.expand_dims(K.permute_dimensions(x0, pattern=(0, 2, 1)), axis=3)
        # (batch_size, embed_size, 1, h_k)
        xk = K.expand_dims(K.permute_dimensions(xk, pattern=(0, 2, 1)), axis=2)
        # (batch_size, embed_size, feat_num, h_k)
        z_k_1 = K.batch_dot(x0, xk, axes=(3, 2))
        # (batch_size, embed_size, feature_map_num)
        output = K.dot(K.reshape(z_k_1, shape=(-1, embed_size, field_num * h_k)), self.kernel)
        # (batch_size, feature_map_size, embed_size)
        return K.permute_dimensions(output, pattern=(0, 2, 1))

    def compute_output_shape(self, input_shapes):
        assert isinstance(input_shapes, list) and len(input_shapes[0]) == 3
        return input_shapes[0][0], self.feature_map_num, input_shapes[0][2]

    def get_config(self):
        config = {
            'feature_map_num': self.feature_map_num,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        }
        base_config = super(CompressInteractLayer, self).get_config()
        return dict(**base_config, **config)


def inner_product_layer(keepdims=False, **kwargs):
    """
    The implementation of inner product layer defined in PNN & AFM CTR prediction model.
    This layer calculate the element-wise-product of each pair of embedded feature of the given embedded matrix.
    The shape of the input is assumed as (batch_size, field_num, embed_size).
    After this layer:
    If `keepdims` is `False`: A tensor with the shape of (batch_size, pairs) will be return.
                              Here, the pairs is equal to (field_num * (field_num-1)) / 2
    If `keepdims` is `True`: A tensor with the shape of (batch_size, pairs, embed_size) will be return.
                             Here, the pairs is equal to (field_num * (field_num-1)) / 2

    :param keepdims: A boolean, whether to keep the embed dimensions or not.
                     If `keepdims` is `False`, the embed dimension will be reduced.
                     If `keepdims` is `True`, the embed dimension will be preserved.
    :return: the Keras Layers object which implements the product operation
    """

    def _pair_wise_element_product(tensor):
        # get the first and second feature index in given tensor
        n_feat = K.int_shape(tensor)[1]
        first_index = []
        second_index = []
        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                first_index.append(i)
                second_index.append(j)
        assert len(first_index) == len(second_index) == (n_feat * (n_feat - 1)) // 2

        # using `K.gather` to generate the first part and the second part
        reference = K.permute_dimensions(tensor, pattern=(1, 0, 2))  # (field_num, batch_size, embed_size)
        first_part = K.permute_dimensions(
            K.gather(reference, first_index),  # (first_index, batch_size, embed_size)
            pattern=(1, 0, 2))  # (batch_size, first_index, embed_size)
        second_part = K.permute_dimensions(
            K.gather(reference, second_index),  # (second_index, batch_size, embed_size)
            pattern=(1, 0, 2))  # (batch_size, second_index, embed_size)

        output = first_part * second_part  # (batch_size, pairs, embed_size)
        if not keepdims:
            output = K.sum(output, axis=2)  # (batch_size, pairs)
        return output

    def _cal_output_shape(input_shape):
        # assume the shape is (batch_size, field_num, embed_size)
        assert len(input_shape) == 3
        n_feat = input_shape[1]
        if keepdims:
            # (batch_size, pairs, embed_size)
            return tuple([input_shape[0], (n_feat * (n_feat - 1)) // 2, input_shape[2]])
        else:
            return tuple([input_shape[0], (n_feat * (n_feat - 1)) // 2])

    return Lambda(function=_pair_wise_element_product, output_shape=_cal_output_shape, **kwargs)


def fm_layer(**kwargs):
    """
    The implementation of fm layer defined in NFM & DeepFM CTR prediction model.

    This layer calculate transfer features by the fast linear time computational cost equation
    defined in FM paper:
    $\frac{1}{2}{\sum_{f=1}^{k}{((\sum^{n}_{i=1}{v_{i,f}x_i})^2-\sum_{i=1}^{n}v^2_{i.f}x_i^2)}}$

    The shape of the input is assumed as (batch_size, field_num, embed_size).
    After this layer, a tensor with the shape of (batch_size, embed_size) will be return.

    :return: the Keras Layers object which implements the bi-interaction operation
    """

    def _bi_interact(tensor):
        sum_square = K.square(K.sum(tensor, axis=1))
        square_sum = K.sum(K.square(tensor), axis=1)
        return 0.5 * (sum_square - square_sum)  # (batch_size, embed_size)

    def _cal_output_shape(input_shape):
        assert len(input_shape) == 3
        return tuple([input_shape[0], input_shape[2]])  # (batch_size, embed_size)

    return Lambda(function=_bi_interact, output_shape=_cal_output_shape, **kwargs)


def sum_layer(axis, keepdims=False, **kwargs):
    """
    This layer does the K.sum operation

    :param axis: An integer or list of integers in [-rank(input tensor), rank(input tensor)),
                 the axes to sum over. If `None` (default), sums over all dimensions.
    :param keepdims: A boolean, whether to keep the dimensions or not.
                     If `keepdims` is `False`, the rank of the tensor is reduced by 1.
                     If `keepdims` is `True`, the reduced dimension is retained with length 1.

    :return: the Keras Layers object which implements the sum operation
    """

    def _operation(tensor):
        return K.sum(tensor, axis=axis, keepdims=keepdims)

    def _cal_output_shape(input_shape):
        # check axis
        axes = list(axis) if isinstance(axis, (list, tuple)) else [axis]
        for ax in axes:
            if ax not in range(-len(input_shape), len(input_shape)):
                raise ValueError('The given axis: ' + str(axes) + ' is illegal. ' +
                                 'The axis should be an integer or list of integers' +
                                 ' in [-rank(input tensor), rank(input tensor)))')
        if keepdims:
            output_shape = list(input_shape)
            for ax in axes:
                output_shape[ax] = 1
            return tuple(output_shape)
        else:
            output_shape = list()
            for pos_rank, neg_rank in zip(range(0, len(input_shape)), range(-len(input_shape), 0)):
                if pos_rank not in axes and neg_rank not in axes:
                    output_shape.append(input_shape[pos_rank])
            return tuple(output_shape)

    return Lambda(function=_operation, output_shape=_cal_output_shape, **kwargs)
