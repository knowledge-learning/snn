from keras import backend as K

_BACKEND = K.backend()

if _BACKEND == 'cntk':
    #sys.stderr.write(' CNTK backend\n')
    raise NotImplementedError('Not implemented backend for cntk in snn')
elif _BACKEND == 'theano':
    #sys.stderr.write('Using Theano backend.\n')
    from .theano_backend import *
elif _BACKEND == 'tensorflow':
    sys.stderr.write('Using TensorFlow backend.\n')
    from .tensorflow_backend import *