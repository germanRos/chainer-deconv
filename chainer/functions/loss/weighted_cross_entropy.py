import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import ipdb

class WeightedCrossEntropy(function.Function):

    """weighted cross entropy loss."""

    ignore_label = -1

    def __init__(self, weights, normalize=True, debug=False):
        self.use_cudnn = False
	self.normalize = normalize
	self.debug = debug
	self.weights = weights.data

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    def forward_cpu(self, inputs):
        raise Exception('Method not implemented yet! This time I was just lazy')

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, t = inputs
        log_y = cupy.log(x + 1e-5)
        self.y = x

	if(self.debug):
		ipdb.set_trace()

        if getattr(self, 'normalize', True):
            coeff = cupy.maximum(1, (t != self.ignore_label).sum())
        else:
            coeff = max(1, len(t))
        self._coeff = cupy.divide(1.0, coeff, dtype=x.dtype)

        log_y = cupy.rollaxis(log_y, 1, log_y.ndim)
        ret = cuda.reduce(
            'S t, raw T log_y, int32 n_channel, raw T coeff, raw T weights', 'T out',
            't == -1 ? 0 : log_y[_j * n_channel + t] * weights[t]',
            'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
        )(t, log_y.reduced_view(), log_y.shape[-1], self._coeff, self.weights.reduced_view())
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        raise Exception('Method not implemented yet! This time I was just lazy')

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs
        gloss = grad_outputs[0]
        n_unit = t.size // len(t)
        coeff = gloss * self._coeff
        gx = cuda.elementwise(
            'T y, S t, raw T coeff, S n_channel, S n_unit, raw T weights',
            'T gx',
            '''
               const int c = (i / n_unit % n_channel);
               gx = ((t == -1) || (c != t)) ? 0 : ((weights[t]*coeff[0]) / max(y, 1e-5));
            ''',
            'crossent_bwd')(self.y, cupy.expand_dims(t, 1), -coeff, x.shape[1], n_unit, self.weights.reduced_view())
        return gx, None


def weighted_cross_entropy(x, t, weights, normalize=True, debug=False):
    """Computes weighted cross entropy loss for softmax activations.

    Args:
        x (Variable): Variable holding a multidimensional array whose element
            indicates unnormalized log probability: the first axis of the
            variable represents the number of samples, and the second axis
            represents the number of classes. While this function computes
            a usual softmax cross entropy if the number of dimensions is equal
            to 2, it computes a cross entropy of the replicated softmax if the
            number of dimensions is greater than 2.
        t (Variable): Variable holding an int32 vector of groundtruth labels.
            If ``t[i] == -1``, correspondig ``x[i]`` is ignored.

    Returns:
        Variable: A variable holding a scalar array of the cross entropy loss.

    .. note::

       This function is differentiable only by ``x``.

    """
    return WeightedCrossEntropy(weights, normalize, debug)(x, t)
