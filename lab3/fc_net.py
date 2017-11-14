import numpy as np

from layers import *
from layer_utils import *


class FullyConnectedNet:

    def __init__(self, hidden_dims, input_dim, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        for i, hidden_dim in enumerate(hidden_dims, 1):
            self.params['W%d' % i] = weight_scale * np.random.randn(input_dim, hidden_dim)
            self.params['b%d' % i] = np.zeros(hidden_dim)
            if self.use_batchnorm:
                self.params['gamma%d' % i] = np.ones(hidden_dim)
                self.params['beta%d' % i] = np.zeros(hidden_dim)
            input_dim = hidden_dim
        i += 1
        self.params['W%d' % i] = weight_scale * np.random.randn(input_dim, num_classes)
        self.params['b%d' % i] = np.zeros(num_classes)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None

        cache = []
        t = X
        cd = None
        for i in range(1, self.num_layers+1):
            w, b = self.params['W%d' % i], self.params['b%d' % i]
            if i == self.num_layers:
                t, c = affine_forward(t, w, b)
            else:
                if self.use_batchnorm:
                    gamma, beta = self.params['gamma%d' % i], self.params['beta%d' % i]
                    t, c = affine_bn_relu_forward(t, w, b, gamma, beta, self.bn_params[i-1])
                else:
                    t, c = affine_relu_forward(t, w, b)
                if self.use_dropout:
                    t, cd = dropout_forward(t, self.dropout_param)
            cache.append((c, cd))

        scores = t

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dx = softmax_loss(scores, y)

        for i in range(self.num_layers, 0, -1):
            w, b = self.params['W%d' % i], self.params['b%d' % i]
            loss += 0.5 * self.reg * np.sum(w * w)
            if i == self.num_layers:
                dx, dw, db = affine_backward(dx, cache[i-1][0])
            else:
                if self.use_dropout:
                    dx = dropout_backward(dx, cache[i-1][1])
                if self.use_batchnorm:
                    dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, cache[i-1][0])
                    grads['gamma%d' % i] = dgamma
                    grads['beta%d' % i] = dbeta
                else:
                    dx, dw, db = affine_relu_backward(dx, cache[i-1][0])
            grads['W%d' % i] = dw + self.reg * w
            grads['b%d' % i] = db


        return loss, grads

    def predict(self, X):
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
