from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
import numpy as np
from keras.backend.common import floatx


def is_tensor(x):
    return isinstance(x, tf_ops._TensorLike) or \
           tf_ops.is_dense_tensor_like(x) or \
           isinstance(x, tf.SparseTensor)


def is_keras_tensor(x):
    """Returns whether `x` is a Keras tensor.

    A "Keras tensor" is a tensor that was returned by a Keras layer,
    (`Layer` class) or by `Input`.

    # Arguments
        x: A candidate tensor.

    # Returns
        A boolean: Whether the argument is a Keras tensor.

    # Raises
        ValueError: In case `x` is not a symbolic tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> from keras.layers import Input, Dense
        >>> np_var = numpy.array([1, 2])
        >>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic tensor.
        ValueError
        >>> k_var = tf.placeholder('float32', shape=(1,1))
        >>> K.is_keras_tensor(k_var) # A variable indirectly created outside of keras is not a Keras tensor.
        False
        >>> keras_var = K.variable(np_var)
        >>> K.is_keras_tensor(keras_var)  # A variable created with the keras backend is not a Keras tensor.
        False
        >>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
        >>> K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras tensor.
        False
        >>> keras_input = Input([10])
        >>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
        True
        >>> keras_layer_output = Dense(10)(keras_input)
        >>> K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a Keras tensor.
        True
    ```
    """
    if not is_tensor(x):
        raise ValueError('Unexpectedly found an instance of type `' +
                         str(type(x)) + '`. '
                         'Expected a symbolic tensor instance.')
    return hasattr(x, '_keras_history')


def make_sparse(value, dtype=None):
    if dtype is None:
        dtype = floatx()
    assert hasattr(value, 'tocoo')
    sparse_coo = value.tocoo()
    indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
                                np.expand_dims(sparse_coo.col, 1)), 1)
    v = tf.SparseTensor(indices=indices,
                        values=sparse_coo.data,
                        dense_shape=sparse_coo.shape)
    return v


def sparse_mean(x, axis=None):
    """Mean of a tensor, alongside the specified axis.
    """
    raise NotImplementedError()


def sparse_mean_squared_error(y_true, y_pred):
    raise NotImplementedError()

def dumps_sparse(matrix):
    pass

def loads_sparse(matrix):
    pass

def sp_floor(x):
    pass