import numpy

from chainer import cuda
from chainer import function
from chainer.utils import conv

#	cuda.elementwise(
#            'T in, int32 dim3_in, int32 dim4_in, int32 k_dim3, int32 k_dim4',
#            'raw T out',
#            '''
#	       int plane = (i / (dim3_in*dim4_in));
#	       int col_in = (i / dim4_in) % dim3_in;
#	       int row_in = i % dim4_in;

#	       int h_out = dim4_in * k_dim4;
#               int w_out = dim3_in * k_dim3;
#	       int col_out = col_in * k_dim3;
#	       int row_out = row_in * k_dim4;
#               int jj = plane*w_out*h_out + col_out*h_out + row_out;

#	       out[jj] = in;

#            ''', 'unpool')(x, DIM3_in, DIM4_in, k_DIM3, k_DIM4, y)

class Unpooling2D(function.Function):

    """Max pooling over a set of 2d planes."""
    def __init__(self, indices, shape):
	super(Unpooling2D, self)
	self.indices = indices.data
	self.shape = shape.data

    def forward_cpu(self, x):
    	raise Exception('Method not implemented yet! This time I was just lazy')

    def forward_gpu(self, x):
	shape = self.shape.tolist()
	y = cuda.cupy.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=x[0].dtype)

	cuda.elementwise(
            'T in, S indices',
            'raw T out',
            '''
	       out[indices] = in;

            ''', 'unpool')(x[0], self.indices, y.reduced_view())

	return y,

    def backward_cpu(self, x, gy):
        raise Exception('Method not implemented yet! This time I was just lazy')

    def backward_gpu(self, x, gy):
        gx = cuda.cupy.empty_like(x[0])
	cuda.elementwise(
            'raw T in, S indices',
            'T out',
            '''
	       out = in[indices];

            ''', 'unpool')(gy[0].reduced_view(), self.indices, gx)
        return gx,


def unpooling_2d(x, indices, shape):
    """Spatial unpooling based on pre-stored indices.

    This function does unpooling of a previous pooled blob.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Ouptut variable.

    """
    return Unpooling2D(indices, shape)(x)
