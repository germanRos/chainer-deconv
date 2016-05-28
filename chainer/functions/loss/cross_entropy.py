import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import ipdb

class CrossEntropy(function.Function):

    """cross entropy loss."""

    ignore_label = -1

    def __init__(self, normalize=True, debug=False):
        self.use_cudnn = False
	self.normalize = normalize
	self.debug = debug

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
        x, t = inputs
	x += 1e-5
        log_y  = numpy.log(x)
        self.y = x

        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)

        log_p = log_yd[numpy.maximum(t.ravel(), 0), six.moves.range(t.size)]

        if getattr(self, 'normalize', True):
            count = (t != self.ignore_label).sum()
        else:
            count = len(x)
        self._coeff = 1.0 / max(count, 1)

        y = (log_p * (t.ravel() != self.ignore_label)).sum(keepdims=True) * (-self._coeff)
        return y.reshape(()),

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
            'S t, raw T log_y, int32 n_channel, raw T coeff', 'T out',
            't == -1 ? 0 : log_y[_j * n_channel + t]',
            'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
        )(t, log_y.reduced_view(), log_y.shape[-1], self._coeff)
        return ret,

    def backward_cpu(self, inputs, grad_outputs):
        # TO-DO!
        pass
        #return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs
        gloss = grad_outputs[0]
        n_unit = t.size // len(t)
        coeff = gloss * self._coeff
        gx = cuda.elementwise(
            'T y, S t, raw T coeff, S n_channel, S n_unit',
            'T gx',
            '''
               const int c = (i / n_unit % n_channel);
               gx = ((t == -1) || (c != t)) ? 0 : (coeff[0] / max(y, 1e-5));
            ''',
            'softmax_crossent_bwd')(
                self.y, cupy.expand_dims(t, 1), -coeff, x.shape[1], n_unit)
        return gx, None


def cross_entropy(x, t, normalize=True, debug=False):
    """Computes cross entropy loss for softmax activations.

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
    return CrossEntropy(normalize, debug)(x, t)
