from keras.models import Model
from keras.layers import Input, Dense, maximum, concatenate
from keras.utils.vis_utils import model_to_dot

from .utils import EntityLayer, RelationLayer


class SNN:
    def __init__(self, entities, relations, entity_shape=32, relation_shape=64):
        self.entities = entities
        self.relations = relations
        self.entity_shape = entity_shape
        self.relation_shape = relation_shape

    def __call__(self, x):
        self.input_ = x
        self.entity_capsules_ = self._build_entities(x)
        self.relation_capsules_ = self._build_relations()
        self.outputs_, self.indicators_, self.representations_ = self._build_outputs()
        return self.representations_

    def build(self):
        return Model(inputs=self.input_, outputs=self.indicators_)

    def _build_entities(self, x):
        entity_capsules = {}

        for e in toposort(self.entities):
            entity_capsules[e] = self._build_entity_capsule(x, e, entity_capsules)

        return entity_capsules

    def _build_entity_capsule(self, x, entity, capsules):
        children = [capsules[c] for c in entity.children]

        if children:
            inputs = maximum(children, name="Max-%s" % entity.name)
        else:
            inputs = x

        return EntityLayer(self.entity_shape, name=entity.name)(inputs)

    def _build_relations(self):
        relation_capsules = {}

        for r in self.relations:
            relation_capsules[r] = self._build_relation_capsule(r)

        return relation_capsules

    def _build_relation_capsule(self, relation):
        src = self.entity_capsules_[relation.src]
        dst = self.entity_capsules_[relation.dst]

        return RelationLayer(self.relation_shape, name=relation.label)([src, dst])

    def _build_outputs(self):
        outputs = {}
        outputs_indicators = []
        outputs_concat = []

        for e in self.entities:
            outputs[e] = Dense(units=1, activation='sigmoid', name="Indicator-%s" % e.name)(self.entity_capsules_[e])
            outputs_indicators.append(outputs[e])
            outputs_concat.append(self.entity_capsules_[e])

        for r in self.relations:
            outputs[r] = Dense(units=1, activation='sigmoid', name="Indicator-%s" % r.label)(self.relation_capsules_[r])
            outputs_indicators.append(outputs[r])
            outputs_concat.append(self.relation_capsules_[r])

        indicators = concatenate(outputs_indicators, name="Indicators")
        concat = concatenate(outputs_concat, name="Representations")

        return outputs, indicators, concat


def toposort(entities):
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
    def __init__(self, name, *parents):
        self.name = name
        self.parents = parents
        self.children = []

        for p in self.parents:
            p.children.append(self)

    def __repr__(self):
        return "<%s>" % self.name

class Relation:
    def __init__(self, label, src:Entity, dst:Entity):
        self.label = label
        self.src = src
        self.dst = dst

    def __repr__(self):
        return "%s(%s , %s)" % (self.label, self.src, self.dst)
