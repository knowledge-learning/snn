import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, maximum, concatenate
from keras.utils.vis_utils import model_to_dot

from .utils import EntityLayer, RelationLayer


class SNN:
    """Representa una SNN.

    Una SNN es una red neuronal que codifica el conocimiento en una base de conocimientos.
    Para definirla, es necesario tener una taxonomía de entidades y una lista de relaciones.

    `entities`: Lista de entidades de tipo `snn.Entity`.
    `relations`: Lista de relaciones de tipo `snn.Relation`.
    `entity_shape`: Tamaño final del embedding de las entidades.
    `relation_shape`: Tamaño final del embedding de las relaciones.

    Una SNN se usa como una capa de `keras` (aunque técnicamente no es una capa).
    Se puede usar en un modelo secuencial o en un modelo funcional.

    Ejemplo:

    >>> x = Input(shape=(100,))
    >>> snn = SNN(entities, relations)
    >>> y = snn(x)

    En este punto se puede continuar construyendo la red a partir de y.

    Para entrenar una SNN hay 2 pasos. Primero se entrena en la base de conocimiento.
    Para ello se usa un modelo interno que la SNN construye por sí sola.

    >>> snn_model = snn.build()

    Se debe entrenar con ejemplos de la forma:
    `(input, [entidades o relaciones])`

    Donde `input` tiene la forma correspondiente a `x`, mientras que `[entidades o relaciones]`
    es una lista de `Entity` y `Relation` que indican los hechos existentes en esa tupla, es decir,
    las clases de las entidades presentes y las relaciones entre ellas.

    El método `snn.map` recibe una lista de listas de `Entity` y `Relation` y devuelve la matriz one-hot
    necesaria para entrenar la SNN.
    """
    def __init__(self, entities, relations, entity_shape=32, relation_shape=64):
        self.entities = entities
        self.relations = relations
        self.entity_shape = entity_shape
        self.relation_shape = relation_shape

    def __call__(self, x):
        """Construye la arquitectura SNN a partir de la entrada `x`,
        y devuelve una capa de `keras` a partir de la cuál se puede seguir
        construyendo la red neuronal.
        """
        self.entity_capsules_ = self._build_entities(x)
        self.relation_capsules_ = self._build_relations()
        self.outputs_, self.indicators_, self.representations_, self.indices_ = self._build_outputs()
        return self.representations_

    def build(self, x, optimizer='rmsprop'):
        """Construye y compila un modelo que permite entrenar la SNN en una base
        de conocimiento. Devuelve el modelo ya compilado.
        """
        model = Model(inputs=x, outputs=self.indicators_)
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        return model

    def _build_entities(self, x):
        """Construye la parte de la SNN correspondiente a las entidades.
        """
        entity_capsules = {}

        for e in toposort(self.entities):
            entity_capsules[e] = self._build_entity_capsule(x, e, entity_capsules)

        return entity_capsules

    def _build_entity_capsule(self, x, entity, capsules):
        """Construye una cápsula de entidad particular para la entidad `entity`.
        Si la entidad es hoja (no tiene hijos) se conecta directamente a `x`,
        de lo contrario, se conecta a los respectivos hijos.
        """
        children = [capsules[c] for c in entity.children]
        inputs = maximum(children, name="max-%s" % entity.name) if len(children) > 1 else x
        return EntityLayer(self.entity_shape, name=entity.name)(inputs)

    def _build_relations(self):
        """Construye la parte de la SNN correspondiente a las relaciones.
        """
        relation_capsules = {}

        for r in self.relations:
            relation_capsules[r] = self._build_relation_capsule(r)

        return relation_capsules

    def _build_relation_capsule(self, relation):
        """Construye una cápsula de relación particular correspondiente a `relation`.
        Se conecta a las entidades correspondientes automáticamente.
        """
        src = self.entity_capsules_[relation.src]
        dst = self.entity_capsules_[relation.dst]
        return RelationLayer(self.relation_shape, name=relation.label)([src, dst])

    def _build_outputs(self):
        """Construye las capas finales, una concatenación de todos los embeddings
        y los indicadores.
        """
        outputs = {}
        outputs_indicators = []
        outputs_concat = []
        indices = {}

        for e in self.entities:
            outputs[e] = Dense(units=1, activation='sigmoid', name="indicator-%s" % e.name)(self.entity_capsules_[e])
            outputs_indicators.append(outputs[e])
            outputs_concat.append(self.entity_capsules_[e])
            indices[e] = len(indices)

        for r in self.relations:
            outputs[r] = Dense(units=1, activation='sigmoid', name="indicator-%s" % r.label)(self.relation_capsules_[r])
            outputs_indicators.append(outputs[r])
            outputs_concat.append(self.relation_capsules_[r])
            indices[r] = len(indices)

        indicators = concatenate(outputs_indicators, name="Indicators")
        concat = concatenate(outputs_concat, name="Representations")

        return outputs, indicators, concat, indices

    def map(self, annotations):
        """Convierte una lista de anotaciones (lista de listas de `Entity` o `Relation`)
        en una matriz one-hot, donde las componentes correspondientes a las relaciones
        y entidades marcadas tienen valor 1.

        Este método es útil para entrenar la SNN, pues de la base de conocimiento
        solo sabemos las entidades y relaciones, pero la SNN es quien sabe a qué índice
        se mapea cada una.
        """
        y = []

        for ann in annotations:
            row = [0] * len(self.indices_)
            for a in ann:
                row[self.indices_[a]] = 1.
            y.append(row)

        return np.asarray(y)


def toposort(entities):
    """Devuelve el orden topológico de las entidades, en el orden
    en que deben construirse las cápsulas para que todos los padres
    se construyan después de los hijos.
    """
    visited = set()
    path = []

    def visit(e):
        if e in visited:
            return

        for c in e.children:
            visit(c)

        visited.add(e)
        path.append(e)

    for e in entities:
        visit(e)

    return path


class Entity:
    """Representa una entidad de una base de conocimiento.
    """
    def __init__(self, name, *parents):
        self.name = name
        self.parents = parents
        self.children = []

        for p in self.parents:
            p.children.append(self)

    def __repr__(self):
        return "<%s>" % self.name


class Relation:
    """Representa una relación de una base de conocimiento, entre 2 entidades.
    """
    def __init__(self, label, src:Entity, dst:Entity):
        self.label = label
        self.src = src
        self.dst = dst

    def __repr__(self):
        return "%s(%s , %s)" % (self.label, self.src, self.dst)
