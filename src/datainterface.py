import abc
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import CountVectorizer


class DataInterface(metaclass=abc.ABCMeta):

    def __init__(self, entities, relations):
        try:
            assert isinstance(entities, list)
        except AssertionError:
            assert isinstance(
                entities, tuple), f"Expected list or tuple, but is {type(entities)}"
        try:
            assert isinstance(relations, list)
        except AssertionError:
            assert isinstance(
                relations, tuple), f"Expected list or tuple, but is {type(relations)}"
        self.ents = tuple(sorted(entities))
        self.rels = tuple(sorted(relations))
        self.vect = CountVectorizer(vocabulary=self.build_vocab(), binary=True)
        self.out_map = {i: n for n, i in enumerate(self.ents+self.rels)}

    @abc.abstractmethod
    def sample_entitie_name(self, name: str, n: int):
        raise NotImplementedError()

    def sample_entities_name(self, name: List[str], n: int):
        assert len(name) != 0
        samples_type = np.random.randint(len(name), size=n)
        samples_type = Counter(samples_type)
        data = None
        res = None
        for i, j in samples_type.items():
            if res is None:
                data, res = self.sample_entitie_name(name[i], j)
            else:
                dd, rr = self.sample_entitie_name(name[i], j)
                data = np.vstack([data, dd])
                res = np.vstack([res, rr])
        return data, res

    def sample_entities(self, n: int):
        return self.sample_entities_name(self.ents, n)

    @abc.abstractmethod
    def sample_relation_name(self, name: str, n: int):
        raise NotImplementedError()

    def sample_relations_name(self, name: List[str], n: int):
        assert len(name) != 0
        samples_type = np.random.randint(len(name), size=n)
        samples_type = Counter(samples_type)
        data = None
        res = None
        for i, j in samples_type.items():
            if res is None:
                data, res = self.sample_relation_name(name[i], j)
            else:
                dd, rr = self.sample_relation_name(name[i], j)
                data = np.vstack([data, dd])
                res = np.vstack([res, rr])
        return data, res

    def sample_relations(self, n: int):
        return self.sample_relations_name(self.rels, n)

    def build_vocab(self):
        tt = self._build_vocab()
        assert isinstance(tt, dict) or isinstance(tt, set)
        return tt

    @abc.abstractmethod
    def _build_vocab(self):
        raise NotImplementedError()
