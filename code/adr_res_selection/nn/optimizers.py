from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from nn_utils import build_shared_zeros


def grad_clipping(g, s):
    g_norm = T.abs_(g)
    return T.switch(g_norm > s, (s * g) / g_norm, g)


def optimizer_select(opt, params, grads, lr=0.001):
    if opt == 'adagrad':
        return ada_grad(grads=grads, params=params, lr=lr)
    elif opt == 'ada_delta':
        return ada_delta(grads=grads, params=params)
    elif opt == 'adam':
        return adam(grads=grads, params=params, lr=lr)
    return sgd(grads=grads, params=params, lr=lr)


def sgd(grads, params, lr=0.1):
    updates = OrderedDict()

    for p, g in zip(params, grads):
        updates[p] = p - lr * g
    return updates


def ada_grad(grads, params, lr=0.1, eps=1.):
    updates = OrderedDict()

    for p, g in zip(params, grads):
        r = build_shared_zeros(p.get_value(True).shape)
        r_t = r + T.sqr(g)
        p_t = p - (lr / (T.sqrt(r_t) + eps)) * g
        updates[r] = r_t
        updates[p] = p_t
    return updates


def ada_delta(grads, params, b=0.999, eps=1e-8):
    updates = OrderedDict()

    for p, g in zip(params, grads):
        r = build_shared_zeros(p.get_value(True).shape)
        v = build_shared_zeros(p.get_value(True).shape)
        s = build_shared_zeros(p.get_value(True).shape)
        r_t = b * r + (1 - b) * T.sqr(g)
        v_t = (T.sqrt(s) + eps) / (T.sqrt(r) + eps) * g
        s_t = b * s + (1 - b) * T.sqr(v_t)
        p_t = p - v_t
        updates[r] = r_t
        updates[v] = v_t
        updates[s] = s_t
        updates[p] = p_t
    return updates


def adam(grads, params, lr=0.001, b1=0.9, b2=0.999, e=1e-8):
    updates = OrderedDict()
    i = theano.shared(np.float32(0.))
    i_t = i + 1.

    for p, g in zip(params, grads):
        v = build_shared_zeros(p.get_value(True).shape)
        r = build_shared_zeros(p.get_value(True).shape)

        v_t = (b1 * v) + (1. - b1) * g
        r_t = (b2 * r) + (1. - b2) * T.sqr(g)

        r_hat = lr / (T.sqrt(r_t / (1. - b2 ** i_t)) + e)
        v_hat = v / (1. - b1 ** i_t)

        p_t = p - r_hat * v_hat
        updates[v] = v_t
        updates[r] = r_t
        updates[p] = p_t

    updates[i] = i_t
    return updates
