import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Mult(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)

    def forward(self, xs):
        return xs[0] * xs[1],

    def backward(self, xs, gy):
        gy = grad_outputs[0]
	return (xs[1]*gy, xs[0]*gy)


def mult(x1, x2):
    """

    """
    return Mult()(x1, x2)
