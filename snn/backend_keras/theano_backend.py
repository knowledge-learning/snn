from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.backend.common import floatx
from theano import tensor as T
from theano.tensor.basic import shape, cast, true_div
from theano.tensor.basic import discrete_dtypes, integer_types
import numpy as np
import codecs
try:
    import theano.sparse as th_sparse_module
except ImportError:
    th_sparse_module = None
try:
    import cPickle as pickle
except ImportError:
    import pickle



def _assert_sparse_module():
    if not th_sparse_module:
        raise ImportError("Failed to import theano.sparse\n"
                          "You probably need to pip install nose-parameterized")


def is_tensor(x):
    return isinstance(x, (T.TensorVariable,
                          T.sharedvar.TensorSharedVariable)) or \
           isinstance(x.type, th_sparse_module.SparseType)


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
    _assert_sparse_module()
    var = th_sparse_module.as_sparse_variable(value)
    return var


def sparse_mean(x, axis=None):
    """Mean of a tensor, alongside the specified axis.
    """
    # bool is available since theano v0.9dev
    if 'int' in x.dtype or x.dtype == 'bool':
        dtype = floatx()
    else:
        dtype = x.dtype

    if isinstance(axis, (integer_types, np.integer)):
        if axis == -1:
            axis = max(x.ndim-1, 0)
    s = th_sparse_module.sp_sum(x, axis, True)
    shp = shape(x)

    if s.dtype in ('float16', 'float32', 'complex64'):
        shp = cast(shp, 'float32')
    else:
        shp = cast(shp, 'float64')

    if axis is None:
        axis = list(range(len(x.data.shape)))
    elif isinstance(axis, (integer_types, np.integer)):
        axis = [axis]
    elif isinstance(axis, np.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]

    for i in axis:
        s = true_div(s, shp[i])

    if s.dtype != shp.dtype and s.dtype in discrete_dtypes:
        s = cast(s, shp.dtype)

    if dtype == 'float16' or (dtype is None and x.dtype == 'float16'):
        s = cast(s, 'float16')
    s.name = 'mean'
    return s


def sparse_mean_squared_error(y_true, y_pred):
    _assert_sparse_module()
    T.mean
    return sparse_mean(th_sparse_module.sqr(th_sparse_module.sub(y_true, y_pred)), axis=-1)

def dumps_sparse(matrix):
    data = matrix.data
    data = pickle.dumps(data)
    return codecs.encode(data, 'base64').decode('ascii')

def loads_sparse(matrix: str):
    data = codecs.decode(matrix.encode('ascii'), 'base64')
    return pickle.loads(data)

def sp_floor(x):
    return th_sparse_module.basic.floor(x)