# encoding: UTF-8


from torch.nn import BatchNorm3d, Module
from models.sync_batchnorm import SynchronizedBatchNorm3d


def hijack_bn(m):
    if isinstance(m, BatchNorm3d):
        s = SynchronizedBatchNorm3d(num_features=m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine)
        s.running_mean = m.running_mean
        s.running_var = m.running_var
        if m.affine:
            s.weight = m.weight
            s.bias = m.bias
        return s
    else:
        return m


def hijack_cgbn(m):
    if isinstance(m, SynchronizedBatchNorm3d):
        s = BatchNorm3d(num_features=m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine)
        s.running_mean = m.running_mean
        s.running_var = m.running_var
        if m.affine:
            s.weight = m.weight
            s.bias = m.bias
        return s
    else:
        return m


def super_setattr(m, name, value):
    crumbs = name.split('.')
    for crumb in crumbs[:-1]:
        m = getattr(m, crumb)
    setattr(m, crumbs[-1], value)


def hijack(m):
    for name, module in m.named_modules():
        if isinstance(module, BatchNorm3d):
            super_setattr(m, name, hijack_bn(module))


def restore(m):
    for name, module in m.named_modules():
        if isinstance(module, SynchronizedBatchNorm3d):
            super_setattr(m, name, hijack_bn(module))
