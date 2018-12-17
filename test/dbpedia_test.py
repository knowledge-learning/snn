import unittest
import os
import sys
from keras.layers import Input
from keras.models import load_model
import json
from itertools import cycle
import numpy as np

try:
    MODULE = os.path.dirname(os.path.realpath(__file__))
except:
    MODULE = ""

sys.path.insert(0, os.path.join(MODULE, '..'))

from src import dbpediaSNN, dbpediaIF, Relation, Entitie, tnorm_loss
from src import dbpedia

sys.path.pop(0)


class mock_pyMongoCursor(list):

    def next(self):
        return self[0]


class mock_pyMongoCollection:

    def __init__(self, data):
        self.inst = data

    def find(self, *args):
        return self.inst[:]

    def aggregate(self, lst):
        n = 0
        for i in lst:
            if '$sample' in i:
                n = i['$sample']['size']
                break
        res = mock_pyMongoCursor()
        for i, j in enumerate(cycle(self.inst)):
            if i == n:
                break
            res.append(j)
        return res


class Test_dbpedia_isa(unittest.TestCase):

    def setUp(self):
        it = [('z', 'a'), ('c', 'a'), ('e', 'c'), ('d', 'b'), ('c', 'b'), ('f', 'c'),
              ('g', 'd'), ('h', 'd'), ('i', 'd'), ('b', 'x'), ('y', 'x'), ('a', 'w')]
        ents = [i for i in 'abcdefghiwxyz']

        inn = Input(shape=(10,), name='input')
        net = dbpediaSNN(ents, {}, it)
        model1 = net(inn, False, True)
        model2 = net(inn, True, True)

        self.ndp = json.load(open(os.path.join(MODULE, 'dbpedia_node_prior.json'), 'r'))
        self.net_ndp = net.node_prior
        model11 = load_model(os.path.join(MODULE, 'dbpedia_problem.model'), custom_objects={
                            'Relation': Relation, 'Entitie': Entitie,
                            'tnorm_loss': tnorm_loss})
        model22 = load_model(os.path.join(MODULE, 'dbpedia_train.model'), custom_objects={
                            'Relation': Relation, 'Entitie': Entitie,
                            'tnorm_loss': tnorm_loss})

        def layers(y):
            return set((map(lambda x: tuple(sorted(x.name.split('_'))), y.layers)))

        self.l1 = layers(model1)
        self.l2 = layers(model2)
        self.l11 = layers(model1)
        self.l22 = layers(model2)

    def test_layers_priotiry_for_training(self):
        self.assertEqual(self.ndp, self.net_ndp)

    def test_training_model_layers(self):
        self.assertEqual(self.l2, self.l2)

    def test_problem_model_layers(self):
        self.assertEqual(self.l1, self.l11)


class Test_dbpedia_DataInterface(unittest.TestCase):

    def setUp(self):
        global dbpedia
        self.old_db = dbpedia.db
        self.old_rels = dbpedia.relations
        dbpedia.db = {"Language": mock_pyMongoCollection([{'instance': 'aa'}, {'instance': 'bb'}]),
                      'Continent': mock_pyMongoCollection([{'instance': 'cc', 'Language': 'aa', 'has_millonarie': 'ee'},
                                                           {'instance': 'dd', 'Language': 'bb', 'has_millonarie': 'ff'}]),
                      'Millonarie': mock_pyMongoCollection([{'instance': 'ee', 'Language': 'aa'},
                                                           {'instance': 'ff', 'Language': 'bb'}]),
                      'Relations': mock_pyMongoCollection(
            [{'e1': 'Language', 'e2': 'Continent', 'relFrom': 'Continent',
                    'relation': 'speak_language'},
             {'e1': 'Millonarie', 'e2': 'Continent', 'relFrom': 'Continent',
              'relation': 'has_millonarie'}])}
        dbpedia.relations = dbpedia.db['Relations']
        self.dif = dbpediaIF(
            ["Language", 'Continent', 'Millonarie'], ['speak_language', 'has_millonarie'])

    def tearDown(self):
        global dbpedia
        dbpedia.db = self.old_db
        dbpedia.relations = self.old_rels

    def test_sample_entitie_name(self):
        data, res = self.dif.sample_entitie_name("Language", 2)
        datag = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0]])
        resg = np.array([[0., 1., 0., 0., 0.],
                        [0., 1., 0., 0., 0.]])
        self.assertLessEqual(np.abs(data-datag).flatten().sum(), 5e-16)
        self.assertLessEqual(np.abs(res-resg).flatten().sum(), 5e-16)

        data, res = self.dif.sample_entitie_name("Continent", 2)
        datag = np.array([[0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0]])
        resg = np.array([[1., 0., 0., 0., 0.],
                         [1., 0., 0., 0., 0.]])
        self.assertLessEqual(np.abs(data-datag).flatten().sum(), 5e-16)
        self.assertLessEqual(np.abs(res - resg).flatten().sum(), 5e-16)

    def test_sample_entities_name(self):
        for i in range(4):
            with self.subTest(f'Entities random sampling, iteration {i}'):
                data, res = self.dif.sample_entities_name(['Language', 'Continent'], 2)
                lang = data[0][0] != 0 or data[0][1] != 0
                cont = data[0][2] != 0 or data[0][3] != 0
                self.assertTrue(cont ^ lang, 'Bad enncoding of entitie')
                self.assertTrue(cont == res[0][0] and lang == res[0][1], 'Bad enncoding of result')

                lang = data[1][0] != 0 or data[1][1] != 0
                cont = data[1][2] != 0 or data[1][3] != 0
                self.assertTrue(cont ^ lang, 'Bad enncoding of entitie')
                self.assertTrue(cont == res[1][0] and lang ==
                                res[1][1], 'Bad enncoding of result')

                self.assertTrue(res[0][2] == 0 and res[1][2]
                                == 0, 'Bad enncoding of result')
                self.assertTrue(res[0][3] == 0 and res[1][3]
                                == 0, 'Bad enncoding of result')
                self.assertTrue(res[0][4] == 0 and res[1][4]
                                == 0, 'Bad enncoding of result')

    def test_sample_entities(self):
        for i in range(4):
            with self.subTest(f'Entities random sampling, iteration {i}'):
                data, res = self.dif.sample_entities(2)
                lang = data[0][0] != 0 or data[0][1] != 0
                cont = data[0][2] != 0 or data[0][3] != 0
                mill = data[0][4] != 0 or data[0][5] != 0
                self.assertTrue(cont ^ lang ^ mill, 'Bad enncoding of entitie')
                self.assertTrue(
                    cont == res[0][0] and lang == res[0][1] and mill == res[0][2], 'Bad enncoding of result')

                lang = data[1][0] != 0 or data[1][1] != 0
                cont = data[1][2] != 0 or data[1][3] != 0
                mill = data[1][4] != 0 or data[1][5] != 0
                self.assertTrue(cont ^ lang ^ mill, 'Bad enncoding of entitie')
                self.assertTrue(
                    cont == res[1][0] and lang == res[1][1] and mill == res[1][2], 'Bad enncoding of result')

                self.assertTrue(res[0][3] == 0 and res[1][3]
                                == 0, 'Bad enncoding of result')
                self.assertTrue(res[0][4] == 0 and res[1][4]
                                == 0, 'Bad enncoding of result')

    def test_sample_relation_name(self):
        data, res = self.dif.sample_relation_name("speak_language", 2)
        datag = np.array([[1, 0, 1, 0, 0, 0],
                       [0, 1, 0, 1, 0, 0]])
        resg = np.array([[1., 1., 0., 0., 1.],
                      [1., 1., 0., 0., 1.]])
        self.assertLessEqual(np.abs(data-datag).flatten().sum(), 5e-16)
        self.assertLessEqual(np.abs(res-resg).flatten().sum(), 5e-16)

        data, res = self.dif.sample_relation_name("has_millonarie", 2)
        datag = np.array([[1, 0, 1, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0]])
        resg = np.array([[1., 1., 0., 1., 0.],
                        [1., 1., 0., 1., 0.]])
        self.assertLessEqual(np.abs(data-datag).flatten().sum(), 5e-16)
        self.assertLessEqual(np.abs(res - resg).flatten().sum(), 5e-16)




if __name__ == '__main__':
    unittest.main()
