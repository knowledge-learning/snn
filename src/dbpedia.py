from typing import List, Tuple, Dict
import re
from urllib.parse import unquote
try:
    from .base_model import SNN, Dense
    from .utils import Relation, Entitie, tnorm_loss, bin_acc, Model
    from .datainterface import DataInterface
    from .mypymongo import PyMongo
except ImportError:
    from base_model import SNN, Dense
    from utils import Relation, Entitie, tnorm_loss, bin_acc, Model
    from datainterface import DataInterface
    from mypymongo import PyMongo

mongo = None
db = None
relations = None
ents_props = None


def init_mongo(uri=None):
    global mongo
    global db
    global relations
    global ents_props
    if uri is None:
        mongo = PyMongo('mongodb://localhost:27017/dbpedia')
    else:
        mongo = PyMongo(uri)
    db = mongo.db
    relations = db['Relations']
    ents_props = db['EntitiesProperties']

init_mongo()


class dbpediaIF(DataInterface):
    pass

class dbpediaSNN(SNN):
    def __init__(self, entities: List[str],
     relations: Dict[str, List[Tuple[str]]], isar: List[Tuple[str]]):
        super(dbpediaSNN, self).__init__(entities, relations, isar)
        self.ent_units = 12
        self.rel_unit = 24
        self.isa_units = 12

    def entitie_capsule(self, name: str, inputt):\
        return Entitie(self.ent_units, name=name)(inputt)

    def relation_capsule(self, name: str, inputs: List):
        return Relation(self.rel_units, name=name)(inputs)

    def isa_capsule(self, name: str, inputt):
        return Dense(self.isa_units, name=name)(inputt)

    def _compile(self, inn, outt):
        model = Model(inputs=inn, outputs=outt)
        model.compile(optimizer='RMSprop',
                        loss=tnorm_loss, metrics=[bin_acc])
        return model
