from keras.activations import get as get_activation
from keras.initializers import get as get_initializer
from keras.constraints import get as get_constraint
from keras.regularizers import get as get_regularizer
from keras.engine import Layer
from keras.engine import InputSpec
from keras.legacy.interfaces import generate_legacy_interface
from keras import backend as K
from keras.metrics import binary_accuracy
from keras.utils.vis_utils import model_to_dot
from keras.models import Model
import functools

legacy_entitie_support = generate_legacy_interface(
    allowed_positional_args=['units'],
    conversions=[('output_dim', 'units')])
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


def tnorm_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] = 1
    return tuple(shape)


def tnorm_loss(y_true, y_pred):
    fp = (1-K.exp(-y_pred))/0.6321205588285577
    fp1 = (1-K.exp(K.abs(y_pred-1)))/0.6321205588285577
    return y_true*fp1+(1-y_true)*fp


# esto solo aporta claridad a los diagramas de las redes ya que todo queda encapsulado
# como una clase de keras y se ve como un nodo en el grafo, no es m\'as que una capa
# densa por cada entidad(2 entidades) y otra capa densa mas grande conectada al concat
# de las capas antes mencionadas
class Relation(Layer):
    @legacy_entitie_support
    def __init__(self, units, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Relation, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)
        self.units = units
        self.supports_masking = True
        #self.activation = get_activation('relu')
        self.activation = get_activation(None)
        self.input_spec = [InputSpec(min_ndim=2), InputSpec(min_ndim=2)]
        self.kernel_initializer = get_initializer('glorot_uniform')
        self.bias_initializer = get_initializer('zeros')
        self.kernel_regularizer = get_regularizer(None)
        self.bias_regularizer = get_regularizer(None)
        self.kernel_constraint = get_constraint(None)
        self.bias_constraint = get_constraint(None)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2 or len(input_shape) != 2:
            raise ValueError('A `Relation` layer should be called '
                             'on a list of 2 inputs')
        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][-1]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError('A `Relation` layer requires '
                             'inputs with matching shapes '
                             'except for the concat axis. '
                             'Got inputs shapes: %s' % (input_shape))
        input_dim = input_shape[0][-1]

        self.kernel1 = self.add_weight(shape=(input_dim, self.units//2),
                                       initializer=self.kernel_initializer,
                                       name='kernel1',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)

        self.kernel2 = self.add_weight(shape=(input_dim, self.units//2),
                                       initializer=self.kernel_initializer,
                                       name='kernel2',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)

        self.kernel3 = self.add_weight(shape=((self.units//2)*2, self.units),
                                       initializer=self.kernel_initializer,
                                       name='kernel3',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)

        self.bias1 = self.add_weight(shape=(self.units//2,),
                                     initializer=self.bias_initializer,
                                     name='bias1',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        self.bias2 = self.add_weight(shape=(self.units//2,),
                                     initializer=self.bias_initializer,
                                     name='bias2',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        self.bias3 = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias3',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)

        self.input_spec = [InputSpec(min_ndim=2, axes={-1: input_dim}),
                           InputSpec(min_ndim=2, axes={-1: input_dim})]
        self.built = True

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError('A `Entitie` layer should be called '
                             'on a list of 2 inputs')
        output1 = K.dot(inputs[0], self.kernel1)
        output1 = K.bias_add(output1, self.bias1)
        output2 = K.dot(inputs[1], self.kernel2)
        output2 = K.bias_add(output2, self.bias2)
        cc = K.concatenate([output1, output2], axis=-1)
        output3 = K.dot(cc, self.kernel3)
        output3 = K.bias_add(output3, self.bias3)
        if self.activation is not None:
            output3 = self.activation(output3)
        return output3

    def compute_output_shape(self, input_shape):
        assert input_shape[0] and len(input_shape) == 2
        assert input_shape[0][-1]
        output_shape = list(input_shape[0])
        output_shape[-1] = self.units
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        masks = [K.expand_dims(m, 0) for m in mask if m is not None]
        return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)

    def get_config(self):
        config = {
            'units': self.units
        }
        base_config = super(Relation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# esto solo aporta claridad a los diagramas de las redes ya que todo queda encapsulado
# como una clase de keras y se ve como un nodo en el grafo, no es m\'as que dos capas
# densas consecutivas
class Entitie(Layer):
    @legacy_entitie_support
    def __init__(self, units, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Entitie, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)
        self.units = units
        self.supports_masking = True
        #self.activation = get_activation('relu')
        self.activation = get_activation(None)
        self.input_spec = InputSpec(min_ndim=2)
        self.kernel_initializer = get_initializer('glorot_uniform')
        self.kernel_initializer2 = get_initializer('orthogonal')
        self.bias_initializer = get_initializer('zeros')
        self.kernel_regularizer = get_regularizer(None)
        self.bias_regularizer = get_regularizer(None)
        self.kernel_constraint = get_constraint(None)
        self.bias_constraint = get_constraint(None)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel1 = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel1',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias1 = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias1',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        self.kernel2 = self.add_weight(shape=(self.units, self.units),
                                       initializer=self.kernel_initializer2,
                                       name='kernel2',
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)
        self.bias2 = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name='bias2',
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel1)
        output = K.bias_add(output, self.bias1)
        output = K.dot(output, self.kernel2)
        output = K.bias_add(output, self.bias2)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units
        }
        base_config = super(Entitie, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
