from keras.engine import Layer
from keras import backend as K
from keras.metrics import binary_accuracy
from keras.utils.vis_utils import model_to_dot
from keras.models import Model
import functools
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras.utils.generic_utils import slice_arrays
from keras.utils.generic_utils import Progbar
from keras.engine.training_utils import check_num_samples
from keras.engine.training_utils import make_batches
from .backend_keras import is_keras_tensor
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, issparse, vstack


# contante de normalizaci\'on dela funci\'on de p\'erdida
# cte = (e-1)/e
#cte = 0.6321205588285577
bin_acc = binary_accuracy

def isLayer(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        assert issubclass(result.__class__, Layer), repr(result) + \
            " is not subclass of keras.engine.base_layer.Layer ."
        return result
    return wrapper


def isModel(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        assert issubclass(result.__class__, Model), repr(result) + \
            " is not subclass of keras.engine.training.Model ."
        return result
    return wrapper


def print_model(model, name: str, prog='dot', formatt='pdf', direction=0):
    if direction == 0:
        rankdir = 'LR'
    else:
        rankdir = 'UD'
    mm = model_to_dot(model, rankdir=rankdir).create(prog=prog, format=formatt)
    with open(name+'.'+formatt, 'wb') as f:
        f.write(mm)


def tnorm(x, axis=-1):
    return K.sum(K.square(x), axis=axis, keepdims=True)

def predict_loop(model, f, ins, batch_size=32, verbose=0, steps=None):
    """Abstract method to loop over some data in batches.

    # Arguments
        model: Keras model instance.
        f: Keras function returning a list of tensors.
        ins: list of tensors to be fed to `f`.
        batch_size: integer batch size.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring `predict_loop` finished.
            Ignored with the default value of `None`.

    # Returns
        Array of predictions (if the model has a single output)
        or list of arrays of predictions
        (if the model has multiple outputs).
    """
    num_samples = check_num_samples(ins,
                                    batch_size=batch_size,
                                    steps=steps,
                                    steps_name='steps')
    if verbose == 1:
        if steps is not None:
            progbar = Progbar(target=steps)
        else:
            progbar = Progbar(target=num_samples)

    indices_for_conversion_to_dense = []
    is_sparse = False
    for i in range(len(model._feed_inputs)):
        if issparse(ins[i]) and not K.is_sparse(model._feed_inputs[i]):
            indices_for_conversion_to_dense.append(i)
        elif issparse(ins[i]) and K.is_sparse(model._feed_inputs[i]):
            is_sparse = True
    if steps is not None:
        # Step-based predictions.
        # Since we do not know how many samples
        # we will see, we cannot pre-allocate
        # the returned Numpy arrays.
        # Instead, we store one array per batch seen
        # and concatenate them upon returning.
        unconcatenated_outs = []
        for step in range(steps):
            batch_outs = f(ins)
            batch_outs = to_list(batch_outs)
            if step == 0:
                for batch_out in batch_outs:
                    unconcatenated_outs.append([])
            for i, batch_out in enumerate(batch_outs):
                unconcatenated_outs[i].append(batch_out)
            if verbose == 1:
                progbar.update(step + 1)
        if is_sparse:
            if len(unconcatenated_outs) == 1:
                return vstack(unconcatenated_outs[0],'csr')
            return [vstack(unconcatenated_outs[i],'csr')
                    for i in range(len(unconcatenated_outs))]
        if len(unconcatenated_outs) == 1:
            return np.concatenate(unconcatenated_outs[0], axis=0)
        return [np.concatenate(unconcatenated_outs[i], axis=0)
                for i in range(len(unconcatenated_outs))]
    else:
        # Sample-based predictions.
        outs = []
        batches = make_batches(num_samples, batch_size)
        index_array = np.arange(num_samples)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if ins and isinstance(ins[-1], float):
                # Do not slice the training phase flag.
                ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_arrays(ins, batch_ids)
            for i in indices_for_conversion_to_dense:
                ins_batch[i] = ins_batch[i].toarray()

            batch_outs = f(ins_batch)
            batch_outs = to_list(batch_outs)
            if batch_index == 0:
                # Pre-allocate the results arrays.
                for batch_out in batch_outs:
                    shape = (num_samples,) + batch_out.shape[1:]
                    if is_sparse:
                        outs.append(lil_matrix(shape, dtype=batch_out.dtype))
                    else:
                        outs.append(np.zeros(shape, dtype=batch_out.dtype))
            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_start:batch_end] = batch_out
            if verbose == 1:
                progbar.update(batch_end)
        if is_sparse:
            return unpack_singleton(list(map(lambda oo: oo.tocsr(), outs)))
        return unpack_singleton(outs)


def tnorm_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] = 1
    return tuple(shape)


def tnorm_loss(y_true, y_pred):
    fp = (1-K.exp(-K.square(y_pred)))/0.6321205588285577
    fp1 = (1-K.exp(-K.square(y_pred-1)))/0.6321205588285577
    return K.mean(y_true*fp1+(1-y_true)*fp, axis=-1)


class Graph:
    def __init__(self, model):
        self.model = model

    def _repr_svg_(self):
        return model_to_dot(self.model).create_svg().decode('utf8')


def draw(model):
    return Graph(model)


def patch_assert_input_compatibility(cls):

    def assert_input_compatibility(self, inputs):
        """Checks compatibility between the layer and provided inputs.

        This checks that the tensor(s) `input`
        verify the input assumptions of the layer
        (if any). If not, exceptions are raised.

        # Arguments
            inputs: input tensor or list of input tensors.

        # Raises
            ValueError: in case of mismatch between
                the provided inputs and the expectations of the layer.
        """
        inputs = to_list(inputs)
        for x in inputs:
            try:
                # patch function with our function that accept sparse tensors
                is_keras_tensor(x)
            except ValueError:
                raise ValueError('Layer ' + self.name + ' was called with '
                                 'an input that isn\'t a symbolic tensor. '
                                 'Received type: ' +
                                 str(type(x)) + '. Full input: ' +
                                 str(inputs) + '. All inputs to the layer '
                                 'should be tensors.')

        if not self.input_spec:
            return
        if not isinstance(self.input_spec, (list, tuple)):
            input_spec = to_list(self.input_spec)
        else:
            input_spec = self.input_spec
        if len(inputs) != len(input_spec):
            raise ValueError('Layer ' + self.name + ' expects ' +
                             str(len(input_spec)) + ' inputs, '
                             'but it received ' + str(len(inputs)) +
                             ' input tensors. Input received: ' +
                             str(inputs))
        for input_index, (x, spec) in enumerate(zip(inputs, input_spec)):
            if spec is None:
                continue

            # Check ndim.
            if spec.ndim is not None:
                if K.ndim(x) != spec.ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected ndim=' +
                                     str(spec.ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            if spec.max_ndim is not None:
                ndim = K.ndim(x)
                if ndim is not None and ndim > spec.max_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected max_ndim=' +
                                     str(spec.max_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            if spec.min_ndim is not None:
                ndim = K.ndim(x)
                if ndim is not None and ndim < spec.min_ndim:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected min_ndim=' +
                                     str(spec.min_ndim) + ', found ndim=' +
                                     str(K.ndim(x)))
            # Check dtype.
            if spec.dtype is not None:
                if K.dtype(x) != spec.dtype:
                    raise ValueError('Input ' + str(input_index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected dtype=' +
                                     str(spec.dtype) + ', found dtype=' +
                                     str(K.dtype(x)))
            # Check specific shape axes.
            if spec.axes:
                try:
                    x_shape = K.int_shape(x)
                except TypeError:
                    x_shape = None
                if x_shape is not None:
                    for axis, value in spec.axes.items():
                        if (value is not None and
                                x_shape[int(axis)] not in {value, None}):
                            raise ValueError(
                                'Input ' + str(input_index) +
                                ' is incompatible with layer ' +
                                self.name + ': expected axis ' +
                                str(axis) + ' of input shape to have '
                                'value ' + str(value) +
                                ' but got shape ' + str(x_shape))
            # Check shape.
            if spec.shape is not None:
                try:
                    x_shape = K.int_shape(x)
                except TypeError:
                    x_shape = None
                if x_shape is not None:
                    for spec_dim, dim in zip(spec.shape, x_shape):
                        if spec_dim is not None and dim is not None:
                            if spec_dim != dim:
                                raise ValueError(
                                    'Input ' + str(input_index) +
                                    ' is incompatible with layer ' +
                                    self.name + ': expected shape=' +
                                    str(spec.shape) + ', found shape=' +
                                    str(x_shape))
    # override the original class with patch function
    cls.assert_input_compatibility = assert_input_compatibility
    return cls

def patch_predict_model(cls):

    def predict(self, x,
                batch_size=None,
                verbose=0,
                steps=None):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            x: The input data, as a Numpy array
                (or list of Numpy arrays if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            verbose: Verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.

        # Returns
            Numpy array(s) of predictions.

        # Raises
            ValueError: In case of mismatch between the provided
                input data and the model's expectations,
                or in case a stateful model receives a number of samples
                that is not a multiple of the batch size.
        """
        # Backwards compatibility.
        if batch_size is None and steps is None:
            batch_size = 32
        if x is None and steps is None:
            raise ValueError('If predicting from data tensors, '
                             'you should specify the `steps` '
                             'argument.')
        # Validate user data.
        x, _, _ = self._standardize_user_data(x)
        if self.stateful:
            if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
                raise ValueError('In a stateful network, '
                                 'you should only pass inputs with '
                                 'a number of samples that can be '
                                 'divided by the batch size. Found: ' +
                                 str(x[0].shape[0]) + ' samples. '
                                 'Batch size: ' + str(batch_size) + '.')

        # Prepare inputs, delegate logic to `predict_loop`.
        if self._uses_dynamic_learning_phase():
            ins = x + [0.]
        else:
            ins = x
        self._make_predict_function()
        f = self.predict_function
        return predict_loop(self, f, ins,
                            batch_size=batch_size,
                            verbose=verbose,
                            steps=steps)
    # override the original class with patch function
    cls.predict = predict
    return cls

