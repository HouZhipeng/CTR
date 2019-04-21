# -*- coding:utf-8 -*-

from keras import backend as K
from keras.layers import Layer, initializers, regularizers, activations, Lambda


class AddBiasLayer(Layer):
    """
    A Keras Layers object.
    In this layer, a trainable bias will be added to the given inputs
    """

    def __init__(self, activation=None, initializer='zeros', regularizer=None, **kwargs):
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        super(AddBiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2 and input_shape[1] == 1  # (batch_size, 1)

        # Create a trainable bias weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,),
                                      initializer=self.initializer,
                                      regularizer=self.regularizer,
                                      trainable=True)
        super(AddBiasLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        output = K.bias_add(inputs, self.kernel)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], 1

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'initializer': initializers.serialize(self.initializer),
            'regularizer': regularizers.serialize(self.regularizer),
        }
        base_config = super(AddBiasLayer, self).get_config()
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


def linear_layer(**kwargs):
    """
    The implementation of linear layer used in CTR prediction model.
    This layer does element-multiply and reduce-sum operation.
    The shape of the two input are assumed to be same and equal to (batch_size, feat_num, 1).
    After this layer, a tensor with the shape of (batch_size,1) will be return.

    :return: the Keras Layers object which implements the linear operation
    """

    def _operation(tensors):
        tensor1 = K.batch_flatten(tensors[0])
        tensor2 = K.batch_flatten(tensors[1])
        return K.sum(tensor1 * tensor2, axis=1, keepdims=True)  # (batch_size, 1)

    def _cal_output_shape(input_shapes):
        shape1 = list(input_shapes[0])
        shape2 = list(input_shapes[1])
        assert shape1 == shape2  # check the equality of dimensions
        return tuple([shape1[0], 1])

    return Lambda(function=_operation, output_shape=_cal_output_shape, **kwargs)


def pair_wise_interact_layer(keepdims=False, **kwargs):
    """
    The implementation of pair wise interact layer defined in PNN & AFM CTR prediction model.
    This layer calculate the element-wise-product of each pair of embedded feature of the given embedded matrix.
    The shape of the input is assumed as (batch_size, feat_num, embed_size).
    After this layer:
    If `keepdims` is `False`: A tensor with the shape of (batch_size, pairs) will be return.
                              Here, the pairs is equal to (feat_num * (feat_num-1)) / 2
    If `keepdims` is `True`: A tensor with the shape of (batch_size, pairs, embed_size) will be return.
                             Here, the pairs is equal to (feat_num * (feat_num-1)) / 2

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
        reference = K.permute_dimensions(tensor, pattern=(1, 0, 2))  # (feat_num, batch_size, embed_size)
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
        # assume the shape is (batch_size, feat_num, embed_size)
        assert len(input_shape) == 3
        n_feat = input_shape[1]
        if keepdims:
            # (batch_size, pairs, embed_size)
            return tuple([input_shape[0], (n_feat * (n_feat - 1)) // 2, input_shape[2]])
        else:
            return tuple([input_shape[0], (n_feat * (n_feat - 1)) // 2])

    return Lambda(function=_pair_wise_element_product, output_shape=_cal_output_shape, **kwargs)


def bi_interact_layer(**kwargs):
    """
    The implementation of bi-interaction layer defined in NFM CTR prediction model.

    This layer calculate transfer features by the fast linear time computational cost equation
    defined in FM paper:
    $\frac{1}{2}{\sum_{f=1}^{k}{((\sum^{n}_{i=1}{v_{i,f}x_i})^2-\sum_{i=1}^{n}v^2_{i.f}x_i^2)}}$

    The shape of the input is assumed as (batch_size, feat_num, embed_size).
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


def axis_sum_layer(axis, keepdims=False, **kwargs):
    """
    This layer does the K.sum operation, which sum the values in a tensor alongside the specified axis.

    :param axis: An integer, the axis to sum over.
    :param keepdims: A boolean, whether to keep the dimensions or not.
                     If `keepdims` is `False`, the rank of the tensor is reduced by 1.
                     If `keepdims` is `True`, the reduced dimension is retained with length 1.
    :return: the Keras Layers object which implements the sum operation
    """

    def _operation(tensor):
        return K.sum(tensor, axis=axis, keepdims=keepdims)

    def _cal_output_shape(input_shape):
        if keepdims:
            output_shape = list(input_shape)
            output_shape[axis] = 1
            return tuple(output_shape)
        else:
            output_shape = list(input_shape)
            output_shape.pop(axis)
            return tuple(output_shape)

    return Lambda(function=_operation, output_shape=_cal_output_shape, **kwargs)
