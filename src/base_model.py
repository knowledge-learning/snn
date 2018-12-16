import abc
from typing import List, Tuple, Dict
try:
    from .utils import tnorm, tnorm_output_shape, isLayer, isModel
except ImportError:
    from utils import tnorm, tnorm_output_shape, isLayer, isModel
from keras.layers import Lambda, Maximum, Dense, Concatenate


class SNN(metaclass=abc.ABCMeta):

    def __init__(self, entities: List[str],
                 relations: Dict[str, List[Tuple[str]]], isar: List[Tuple[str]]):
        """ redes semanticas
        entities: lista de strings
        relations: diccionario de: llave relacion, valor lista de tuplas de
        pares de entidades relacionadas
        ej:
        ents = ['persona','pelic\\\'ula']
        rels = {'director':[('persona','pelic\\\'ula')]}
        isar: lista de tuplas de pares de entidades relacionadas por relaciones
        parecidas a "is a" de la forma (hijo, padre)
        """
        eet = set(entities)
        # check for duplicate entities
        assert len(eet) == len(entities), "Exist duplicated entities"
        # avoid overhead of acces class attribute
        for rel_name, ents in relations.items():
            for ents_pair in ents:
                # check that relations have only pairs of entities
                assert len(ents_pair) == 2
                # check thath entities are in the entities provided after
                assert ents_pair[0] in eet
                assert ents_pair[1] in eet
        self.relations = relations
        for ents_pair in isar:
            # check that relations have only pairs of entities
            assert len(ents_pair) == 2
            # check thath entities are in the entities provided after
            assert ents_pair[0] in eet
            assert ents_pair[1] in eet
        self.isar = isar
        self.entities = list(sorted(entities))
        self.node_prior = self._make_proc_ord(isar)

    def _make_proc_ord(self, isar):
        """ Las relaciones isa forman una foresta, por lo que
            hay que conectar de los hijos hacia los padres y
            hasta que todos los hijos del nivel inferior no est\'en
            conectados no se pueden conectar al padre. Este m\'etodo
            genera el orden en el que hay que connectar los nodos
            de la ralacion isa.
            """
        parents = {}
        procord = []
        w = {}
        if not isar:
            self.porcord = []
            return w

        for child, parent in isar:
            if not(child in w):
                w[child] = 0
            if parent in parents:
                parents[parent][0].add(child)
                nw = max(procord[parents[parent][1]][2], w[child]+1)
                nw = max([w[i] for i in parents[parent][0]])+1
                if nw != w[parent]:
                    w[parent] = nw
                    self._update_priority(procord, nw, parent, w)
                procord.pop(parents[parent][1])
                procord.append([parents[parent][0], parent, w[parent]])
                parents[parent][1] = len(procord)-1

            else:
                w[parent] = w[child]+1
                self._update_priority(procord, w[parent], parent, w)
                procord.append([set([child]), parent, w[parent]])
                parents[parent] = [set([child]), len(procord)-1]
        self.porcord = list(map(lambda y: (tuple(y[0]), y[1]),
                                sorted(procord, key=lambda x: x[-1])))
        return w

    def _update_priority(self, procord, nval, p, w):
        """ M\'etodo auxiliar para generar el orden de
        conectar los nodos de las relaciones isa.
        """
        change = []
        for i in range(len(procord)):
            if p in procord[i][0]:
                ent = procord[i][1]
                nw = max(procord[i][2], nval+1)
                if w[ent] != nw:
                    w[ent] = nw
                    procord[i][2] = nw
                    change.append((nw, ent))

        for i, j in change:
            self._update_priority(procord, i, j, w)

    def _make_isa_layers(self, inputt):
        """ Generar los nodos entidad de las
        relaciones isa y los conecta seg\'un el orden
        computado anteriormente.
        """
        # avoid overhead of acces class attribute
        tt = self.ents
        procord = self.porcord
        if not procord:
            return
        # chs = set()
        for childs, parent in procord:
            # chs.update(childs)
            for child in childs:
                if not(child in tt):
                    tt[child] = self.entitie_capsule(child, inputt)
            if len(childs) == 1:
                rel = self.isa_capsule(childs[0]+'_isa_'+parent, tt[childs[0]])
            else:
                strch = '_'.join(childs)
                mm = Maximum(name='max_'+strch)([tt[i] for i in childs])
                rel = self.isa_capsule(strch+'_isa_'+parent, mm)
            tt[parent] = self.entitie_capsule(parent, rel)
        # sents = set(tt.keys())
        # np = sents.difference_update(chs)
        # return [tt[i] for i in np]

    def _build_model(self, inputt):
        """ Genera el mode, es decir, se crean los nodos entidad
            y los nodos relacion y se conectan
        """
        self.ents = {}
        ents = self.ents
        self._make_isa_layers(inputt)
        # clean procord
        self.procord = []

        for i in self.entities:
            if not(i in ents):
                ents[i] = self.entitie_capsule(i, inputt)
                self.node_prior[i] = 0
        self.rels = {}
        rels = self.rels
        for rel, ents_p in self.relations.items():
            if len(ents_p) == 1:
                e1, e2 = ents_p[0]
                rels[rel] = self.relation_capsule(
                    rel, [ents[e1], ents[e2]])
            for n, (e1, e2) in enumerate(ents_p):
                rels[rel+str(n)] = self.relation_capsule(
                    rel+'_'+str(n), [ents[e1], ents[e2]])

    def __call__(self, inputt, train=False, compilee=False):
        self._build_model(inputt)
        ents = self.ents
        rels = self.rels
        if train:
            out = []
            for i in self.entities:
                out.append(Lambda(tnorm,
                                  output_shape=tnorm_output_shape, name=i+'-out')(ents[i]))
            for i in sorted(rels.keys()):
                out.append(Lambda(tnorm,
                                  output_shape=tnorm_output_shape, name=i+'-out')(rels[i]))
            outt = Concatenate(name='out_embeding')(out)
            if compilee:
                return self.wrap_compile(inputt, outt)
            return outt
        outt = Concatenate(name='out_embeding')([ents[i] for i in self.entities] +
                                                [rels[j] for j in sorted(rels.keys())])
        if compilee:
            return self.wrap_compile(inputt, outt)
        return outt

    def pretrain(self):
        pass

    @isModel
    def wrap_compile(self, inn, outt):
        return self._compile(inn, outt)

    @abc.abstractmethod
    def _compile(self, inn, outt):
        raise NotImplementedError()

    @abc.abstractmethod
    def entitie_capsule(self, name: str, inputt):
        """
        name:  name of the entitie
        imput: is a keras symbolic tensor or input layer or a class that
        implement keras.Layer interface
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def relation_capsule(self, name: str, inputs: List):
        """
        name:  name of the entitie
        imputs: is a list of keras symbolic tensors or input layers or a
        classes that implement keras.Layer interface
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def isa_capsule(self, name: str, inputt):
        """
        name:  name of the entitie
        imput: is a keras symbolic tensor or input layer or a class that
        implement keras.Layer interface
        """
        raise NotImplementedError()
