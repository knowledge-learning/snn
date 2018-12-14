from typing import List, Tuple, Dict
try:
    from .base_model import SNN, Dense
    from .utils import Relation, Entitie, tnorm_loss, bin_acc, Model
except ImportError:
    from base_model import SNN, Dense
    from utils import Relation, Entitie, tnorm_loss, bin_acc, Model


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

    def pretrain(self):
        pass