import abc
import numpy as np
from collections import Counter
from typing import List, Tuple, Dict


class DataInterface(metaclass=abc.ABCMeta):

    def __init__(self, entities, relations):
        try:
            assert isinstance(ee, list)
        except AssertionError:
            assert isinstance(
                ee, tuple), f"Expected list or tuple, but is {type(ee)}"
        try:
            assert isinstance(ee, list)
        except AssertionError:
            assert isinstance(
                ee, tuple), f"Expected list or tuple, but is {type(ee)}"
        self.ents = entities
        self.resl = relations

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
                data = np.hstack([data, dd])
                res = np.hstack([res, rr])
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
                data = np.hstack([data, dd])
                res = np.hstack([res, rr])
        return data, res

    def sample_relation(self, n: int):
        return self.sample_entities_name(self.rels, n)

    @abc.abstractmethod
    def build_vocab(self):
        raise NotImplementedError()
