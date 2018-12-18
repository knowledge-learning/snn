from typing import List, Tuple, Dict
import re
from urllib.parse import unquote
import numpy as np
from random import choice
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


def clean_ent(data):
    return unquote(data).replace('\\\'', '\'')

init_mongo()
# cols = set(db.list_collection_names())
# cols.remove('Relations')
# cols.remove('EntitiesProperties')


class dbpediaIF(DataInterface):

    def _build_vocab(self):
        ents = self.ents
        vocab = set()
        for i in range(len(ents)):
            data = db[ents[i]]
            for j in data.find({}):
                inst = clean_ent(j['instance']).split('_')
                for k in inst:
                    tt = re.sub('\(|\)|\!|\.|\,|\$|\@|\"|\?', '', k)
                    if len(tt) > 1 and re.search('\D', tt) and not bool(re.search('[\[\]\|\:]', tt)):
                        vocab.add(tt)
        return vocab

    def sample_entitie_name(self, name: str, n: int):
        assert name in self.ents, f'Entitie {name} not found.'
        tt = db[name]
        data = []
        while len(data) < n:
            for j in tt.aggregate([{'$sample': {'size': n-len(data)}}]):
                    inst = clean_ent(j['instance'])
                    inst = re.sub('\(|\)|\!|\.|\,|\$|\@|\"|\?', '', inst)
                    if len(inst) > 1 and re.search('\D', inst) and not bool(re.search('[\[\]\|\:]', inst)):
                        data.append(' '.join(inst.split('_')))
        inn = self.vect.transform(data)
        res = np.zeros((n, len(self.out_map)))
        res[:, self.out_map[name]] = 1
        return inn.toarray(), res

    def sample_relation_name(self, name: str, n: int):
        rel = relations.aggregate([{'$match': {'relation': name.lower()}},
            {'$match': {'$and': [{'e1':  {'$in': self.ents}}, {'e2':  {'$in': self.ents}}]}}, {
                       '$sample': {'size': 1}}]).next()
        #rel = relations.find_one({'relation': name.lower()})
        assert rel and name in self.rels, f'Relation {name} not found.'
        e1 = rel['e1']
        e2 = rel['e2']
        ee = rel['relFrom']
        assert e1 in self.ents
        assert e2 in self.ents
        if e1 == ee:
            samp = db[e1]
            other = e2
        else:
            samp = db[e2]
            other = e1
        data = []
        while len(data) < n:
            for j in samp.aggregate([{'$match': {other: {'$exists': 'true'}}}, {
                    '$sample': {'size': n - len(data)}}]):
                temp = ''
                inst = clean_ent(j['instance'])
                inst = re.sub('\(|\)|\!|\.|\,|\$|\@|\"|\?', '', inst)
                if len(inst) > 1 and re.search('\D', inst) and not bool(re.search('[\[\]\|\:]', inst)):
                    temp += ' '.join(inst.split('_'))+' '
                else:
                    continue
                rr = j[other]
                if isinstance(rr, list) or isinstance(rr, tuple):
                    rr = choice(rr)
                inst = clean_ent(rr)
                inst = re.sub('\(|\)|\!|\.|\,|\$|\@|\"|\?', '', inst)
                if len(inst) > 1 and re.search('\D', inst) and not bool(re.search('[\[\]\|\:]', inst)):
                    temp += ' '.join(inst.split('_'))
                else:
                    continue
                data.append(temp)
        inn = self.vect.transform(data)
        res = np.zeros((n, len(self.out_map)))
        res[:, self.out_map[name]] = 1
        res[:, self.out_map[e1]] = 1
        res[:, self.out_map[e2]] = 1
        return inn.toarray(), res


class dbpediaSNN(SNN):
    def __init__(self, entities: List[str],
     relations: Dict[str, List[Tuple[str]]], isar: List[Tuple[str]]):
        super(dbpediaSNN, self).__init__(entities, relations, isar)
        self.ent_units = 12
        self.rel_units = 24
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
