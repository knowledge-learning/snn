from keras.activations import get as get_activation
from keras.initializers import get as get_initializer
from keras.constraints import get as get_constraint
from keras.regularizers import get as get_regularizer
from keras import optimizers, losses
from keras.engine import Layer
from keras.layers import Lambda
from keras import Model as kModel
from keras.engine import InputSpec
from keras import backend as K
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy.interfaces import generate_legacy_interface

from scipy.sparse import csr_matrix, issparse
from functools import partial

from .backend_keras import is_tensor, dumps_sparse, loads_sparse
from .backend_keras import make_sparse, sp_floor
from .utils import patch_assert_input_compatibility
from .utils import patch_predict_model
import numpy as np


legacy_entitie_support = generate_legacy_interface(
    allowed_positional_args=['units'],
    conversions=[('output_dim', 'units')])

# esto solo aporta claridad a los diagramas de las redes ya que todo queda encapsulado
# como una clase de keras y se ve como un nodo en el grafo, no es m\'as que una capa
# densa por cada entidad(2 entidades) y otra capa densa mas grande conectada al concat
# de las capas antes mencionadas
@patch_assert_input_compatibility
class RelationLayer(Layer):
    @legacy_entitie_support
    def __init__(self, units, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RelationLayer, self).__init__(**kwargs)
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
        base_config = super(RelationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# esto solo aporta claridad a los diagramas de las redes ya que todo queda encapsulado
# como una clase de keras y se ve como un nodo en el grafo, no es m\'as que dos capas
# densas consecutivas
@patch_assert_input_compatibility
class EntityLayer(Layer):
    @legacy_entitie_support
    def __init__(self, units, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(EntityLayer, self).__init__(**kwargs)
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
        base_config = super(EntityLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# por revisar
# esta capa se le deben setearser los pesos despues de contruida
# por si sola no hace nada
# es necesario computar los pesos aparte y luego setearlo para no tener que guardar en la
# capa las instancias de entidades
# solo se le pasa la matriz de pesos de dimension entrada por cantidad de instancias de entidades
# preferiblemente una matriz esparcida de scipy a la hora de setear los pesos
@patch_assert_input_compatibility
class EntityEmbeding(Layer):

    def __init__(self, matrix, **kwargs):

        super(EntityEmbeding, self).__init__(**kwargs)

        if issparse(matrix):
            self.matrix = make_sparse(matrix)
        elif is_tensor(matrix):
            self.matrix = matrix
        else:
            raise ValueError("matrix must be a scipy.sparse matrix or Tensor")

        self.trainable = False

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] = self.matrix.data.shape[-1]
        return tuple(shape)

    def call(self, inputs, mask=None):
        return sp_floor(K.dot(inputs, self.matrix))

    def get_config(self):
        config = {'matrix': dumps_sparse(self.matrix)}
        base_config = super(EntityEmbeding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        globs = globals()
        if custom_objects:
            globs = dict(list(globs.items()) + list(custom_objects.items()))

        config['matrix'] = loads_sparse(config['matrix'])
        return cls(**config)

@patch_predict_model
class Model(kModel):
    pass