import numpy as np


def affine_forward(x, w, b):
    N = x.shape[0]
    x_flat = x.reshape((N, -1))
    out = x_flat.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    xf = x.reshape((x.shape[0], -1))
    dw = np.dot(xf.T, dout).reshape(w.shape)
    dx = np.dot(dout, w.T).reshape(x.shape)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        mu = sample_mean
        xc = x - mu
        s2 = sample_var
        s = np.sqrt(s2 + eps)
        s1 = 1 / s
        xn = xc * s1
        out = xn * gamma + beta
        cache = (gamma, xc, xn, s1, eps)
    elif mode == 'test':
        out = gamma * (x - running_mean) / (np.sqrt(running_var) + eps) + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    (gamma, xc, xn, s1, eps) = cache

    dgamma = np.sum(xn * dout, axis=0)
    dbeta = np.sum(dout, axis=0)

    dxn = gamma * dout
    ds1 = xc * dxn
    dxc = s1 * dxn
    ds = -s1**2 * ds1
    ds2 = s1 / 2 * ds
    dxc += 2 * xc * ds2.mean(axis=0)
    dmu = dxc.mean(axis=0)
    dx = dxc - dmu

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) > p) / p
        out = mask * x
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    return dx


def svm_loss(x, y):
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
