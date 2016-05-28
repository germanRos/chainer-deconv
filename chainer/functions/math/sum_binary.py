import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class SumBinary(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

    def forward(self, xs):
        return xs[0] + xs[1],

    def backward(self, xs, gy):
        gy = gy[0]
	return (gy, gy)


def sum_binary(x1, x2):
    """

    """
    return SumBinary()(x1, x2)
