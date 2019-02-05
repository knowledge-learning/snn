from .base import SNN
from keras.layers import Input


class SNNModel:
    def __init__(self, entities, relations, entity_shape=32, relation_shape=64, input_type='instances', **kwargs):
        self.snn = SNN(entities, relations, entity_shape=entity_shape, relation_shape=relation_shape)
        self.input = self._build_input(input_type, **kwargs)
        self.output = self.snn(self.input)
        self.model = self.snn.build(self.input)

    def _build_input(self, input_type, **kwargs):
        if input_type == 'instances':
            instances = kwargs.pop('instances', None)
            if instances is None:
                raise ValueError("Must provide `instances:int` value for building the input.")

            x = Input(shape=(instances,), name="input")
            return x
        else:
            raise ValueError("Input type %s not valid." % repr(input_type))
