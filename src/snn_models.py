from keras.layers import Dense, Input, Concatenate, PReLU
from keras.models import Model
from keras.activations import get as get_activation
from keras.initializers import get as get_initializer
from keras.constraints import get as get_constraint
from keras.regularizers import get as get_regularizer
from keras.engine import Layer
from keras.engine import InputSpec
from keras.legacy.interfaces import generate_legacy_interface
from keras import backend as K

legacy_entie_support = generate_legacy_interface(
    allowed_positional_args=['units'],
    conversions=[('output_dim', 'units')])

def srelu(x):
    alpha = 0.8
    return K.relu(x, alpha=alpha, max_value=None)


class EntitiesEmbeding:

    def __init__(self,out_size,name):
        self.out_size = out_size
        self.name = name

    def __call__(self,input):
        tt = Dense(self.out_size,name=self.name)(input)
        return tt

class RelationsEmbeding:

    def __init__(self,out_size,name):
        self.out_size = out_size
        self.name = name

    def __call__(self,input):
        tt = Dense(self.out_size,name=self.name)(input)
        return tt
    
class OntoEmbeding:
    """Embeber ontologias
    entities: tupla o lista de strings
    relations: diccionario de: llave relacion, valor tupla de entidades relacionadas
    las relaciones tienen que ser binarias
    ej:
    ents = ['persona','pelic\\\'ula']
    rels = {'director':('persona','pelic\\\'ula')}
    """

    def __init__(self,entities,relations):
        self.entities = entities
        self.relations = relations

    def __call__(self,input, out_entitie = 10, out_relation = 20, out_sigmoid = False):
        """
        input: clase de keras para connectar con embeding
        out_entitie: taman\\~o de salida del embeding de entidades
        out_relation: taman\\~o de salida del embeding de relaciones
        out_sigmoid: booleano 
        True retorna el concat la sigmoidal de cada entidad y relacion, esto
        es para entrenar el embeding
        False retorna el concat de las entidades y relaciones
        """
        ents = {}
        for i in self.entities:
            ents[i] = EntitiesEmbeding(out_entitie, name = i)(input)
        rels = {}
        concats = {}
        for rel,(e1,e2) in self.relations.items():
            ee = e1+e2
            if not(ee in concats):
                concats[ee] = Concatenate(name= 'c_'+e1+'_'+e2)([ents[e1],ents[e2]])
            rels[rel] = RelationsEmbeding(out_relation, name=rel)(concats[ee])

        if out_sigmoid:
            out = []
            for i in sorted(ents.keys()):
                out.append(Dense(1, activation = 'sigmoid', name=i+'-out')(ents[i]))
            for i in sorted(rels.keys()):
                out.append(Dense(1, activation = 'sigmoid', name=i+'-out')(rels[i]))
            outt = Concatenate(name = 'out_embeding')(out)
            return outt

        out = []
        for i in sorted(ents.keys()):
            out.append(ents[i])
        for i in sorted(rels.keys()):
            out.append(rels[i])
        outt = Concatenate(name = 'out_embeding')(out)
        return outt


class RelationsEmbeding2:

    def __init__(self,input_size,out_size,name):
        self.out_size = out_size
        self.name = name
        self.input_size = input_size

    def __call__(self,inputs):
        if len(inputs)!=2:
            raise Exception('Bad input, 2 inpust are requeried')
        inn1 = Input(shape=(self.input_size,), name='input1')
        inn2 = Input(shape=(self.input_size,), name='input2')
        t1 = Dense(self.out_size//2,name=self.name+'_d1')(inn1)
        t2 = Dense(self.out_size//2,name=self.name+'_d2')(inn2)
        cc = Concatenate(name=self.name+'_c')([t1,t2])
        tt = Dense(self.out_size,name=self.name+'_out')(cc)
        mm = Model(inputs=[inn1,inn2], outputs=tt)
        return mm

class OntoEmbeding2:
    """Embeber ontologias
    entities: tupla o lista de strings
    relations: diccionario de: llave relacion, valor tupla de entidades relacionadas
    las relaciones tienen que ser binarias
    ej:
    ents = ['persona','pelic\\\'ula']
    rels = {'director':('persona','pelic\\\'ula')}
    """

    def __init__(self,entities,relations):
        self.entities = entities
        self.relations = relations

    def __call__(self,input, out_entitie = 10, out_relation = 20, out_sigmoid = False):
        """
        input: clase de keras para connectar con embeding
        out_entitie: taman\\~o de salida del embeding de entidades
        out_relation: taman\\~o de salida del embeding de relaciones
        out_sigmoid: booleano 
        True retorna el concat la sigmoidal de cada entidad y relacion, esto
        es para entrenar el embeding
        False retorna el concat de las entidades y relaciones
        """
        ents = {}
        for i in self.entities:
            ents[i] = EntitiesEmbeding(out_entitie, name = i)(input)
        rels = {}
        for rel,(e1,e2) in self.relations.items():
            rels[rel] = Relation(out_relation, name=rel)([ents[e1],ents[e2]])

        if out_sigmoid:
            out = []
            for i in sorted(ents.keys()):
                out.append(Dense(1, activation = 'sigmoid', name=i+'-out')(ents[i]))
            for i in sorted(rels.keys()):
                out.append(Dense(1, activation = 'sigmoid', name=i+'-out')(rels[i]))
            outt = Concatenate(name = 'out_embeding')(out)
            return outt

        out = []
        for i in sorted(ents.keys()):
            out.append(ents[i])
        for i in sorted(rels.keys()):
            out.append(rels[i])
        outt = Concatenate(name = 'out_embeding')(out)
        return outt

# por terminar cuando se tenga el modelo de una entidad( No creo que sea necesario)
# esto solo aporta claridad a los diagramas de las redes ya que todo queda encapsulado
# como una clase de keras y se ve como un nodo en el grafo

class Relation(Layer):
    @legacy_entie_support
    def __init__(self,units,**kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Relation, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)
        self.units = units
        self.supports_masking = True
        #self.activation = get_activation('relu')
        self.activation = get_activation(None)
        self.input_spec = [InputSpec(min_ndim=2),InputSpec(min_ndim=2)]
        self.kernel_initializer = get_initializer('glorot_uniform')
        self.bias_initializer = get_initializer('zeros')
        self.kernel_regularizer = get_regularizer(None)
        self.bias_regularizer = get_regularizer(None)
        self.kernel_constraint = get_constraint(None)
        self.bias_constraint = get_constraint(None)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2 or len(input_shape)!=2:
            raise ValueError('A `Entitie` layer should be called '
                             'on a list of 2 inputs')
        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][-1]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError('A `Entitie` layer requires '
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

        self.kernel3 = self.add_weight(shape=(input_dim*2, self.units),
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
        if len(inputs)!=2:
            raise ValueError('A `Entitie` layer should be called '
                             'on a list of 2 inputs')
        output1 = K.dot(inputs[0], self.kernel1)
        output1 = K.bias_add(output1, self.bias1)
        output2 = K.dot(inputs[1], self.kernel2)
        output2 = K.bias_add(output2, self.bias2)
        cc = K.concatenate([output1,output2], axis=-1)
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
        'units':self.units
        }
        base_config = super(Relation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))