import numpy

from chainer import cuda
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
import ipdb

class MaxPooling2DIndices(pooling_2d.Pooling2D):

    """Max pooling over a set of 2d planes."""

    def forward_cpu(self, x):
          raise Exception('Method not implemented yet! It was sort of...complicated...')

    def forward_gpu(self, x):
        n, c, h, w = x[0].shape
        y_h = conv.get_conv_outsize(
            h, self.kh, self.sy, self.ph, self.cover_all)
        y_w = conv.get_conv_outsize(
            w, self.kw, self.sx, self.pw, self.cover_all)
        y = cuda.cupy.zeros((n, c, y_h, y_w), dtype=x[0].dtype)
        self.indexes = cuda.cupy.zeros((n, c, y_h, y_w), dtype=numpy.int32)

	indices_unpooling = cuda.cupy.zeros((n, c, y_h, y_w), dtype=numpy.int32)
        cuda.elementwise(
            'raw T in, int32 h, int32 w, int32 out_h, int32 out_w,'
            'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw',
            'T out, S indexes, S indices_unpooling',
            '''
               int c0    = (int)floor(double(i) / double(out_h * out_w));
               int out_y = int(floor(double(i) / double(out_w))) % out_h;
               int out_x = i % out_w;
               int in_y_0 = max(0, out_y * sy - ph);
               int in_y_1 = min(h, out_y * sy + kh - ph);
               int in_x_0 = max(0, out_x * sx - pw);
               int in_x_1 = min(w, out_x * sx + kw - pw);

               T maxval = in[in_x_0 + w * (in_y_0 + h * c0)];
               int argmax_y = in_y_0;
               int argmax_x = in_x_0;
	       int max_ind = in_x_0 + w * (in_y_0 + h * c0);
               for (int y = in_y_0; y < in_y_1; ++y) {
                 int offset_y = w * (y + h * c0);
                 for (int x = in_x_0; x < in_x_1; ++x) {
                   float v = in[x + offset_y];
                   if (maxval < v) {
                     maxval   = v;
                     argmax_y = y;
                     argmax_x = x;
                     max_ind = x + offset_y;
                   }
                 }
               }
               out = maxval;

               int argmax_ky = argmax_y + ph - out_y * sy;
               int argmax_kx = argmax_x + pw - out_x * sx;
               indexes = argmax_kx + kw * argmax_ky;
	       indices_unpooling = max_ind;
            ''', 'max_pool_fwd')(x[0].reduced_view(),
                                 h, w, y_h, y_w, self.kh, self.kw,
                                 self.sy, self.sx, self.ph, self.pw,
                                 y, self.indexes, indices_unpooling)

	
        return (y, indices_unpooling, cuda.cupy.array([n, c, h, w]))

    def backward_cpu(self, x, gy):
        raise Exception('I did not implemented the cpu forward, did you really expect me to implement the backward? please...')

    def backward_gpu(self, x, gy):
        n, c, h, w = x[0].shape
        y_h, y_w = gy[0].shape[2:]
        gx = cuda.cupy.empty_like(x[0])

        cuda.elementwise(
            'raw T gy, raw S indexes, int32 h, int32 w,'
            'int32 out_h, int32 out_w, int32 kh, int32 kw,'
            'int32 sy, int32 sx, int32 ph, int32 pw',
            'T gx',
            '''
               int c0 = i / (h * w);
               int y  = i / w % h + ph;
               int x  = i % w + pw;
               int out_y_0 = max(0,     (y - kh + sy) / sy);
               int out_y_1 = min(out_h, (y      + sy) / sy);
               int out_x_0 = max(0,     (x - kw + sx) / sx);
               int out_x_1 = min(out_w, (x      + sx) / sx);

               T val = 0;
               for (int out_y = out_y_0; out_y < out_y_1; ++out_y) {
                 int ky = y - out_y * sy;
                 for (int out_x = out_x_0; out_x < out_x_1; ++out_x) {
                   int kx = x - out_x * sx;
                   int offset = out_x + out_w * (out_y + out_h * c0);
                   if (indexes[offset] == kx + kw * ky) {
                     val += gy[offset];
                   }
                 }
               }
               gx = val;
            ''',
            'max_pool_bwd')(gy[0].reduced_view(), self.indexes.reduced_view(),
                            h, w, y_h, y_w, self.kh, self.kw,
                            self.sy, self.sx, self.ph, self.pw,
                            gx)
        return gx,


def max_pooling_2dIndices(x, ksize, stride=None, pad=0, cover_all=True):
    """Spatial max pooling function.

    This function acts similarly to :class:`~functions.Convolution2D`, but
    it computes the maximum of input spatial patch for each channel
    without any parameter instead of computing the inner products.

    Args:
        x (~chainer.Variable): Input variable.
        ksize (int or (int, int)): Size of pooling window. ``ksize=k`` and
            ``ksize=(k, k)`` are equivalent.
        stride (int or (int, int) or None): Stride of pooling applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent. If None is
            specified, then it uses same stride as the pooling window size.
        pad (int or (int, int)): Spatial padding width for the input array.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        cover_all (bool): If True, all spatial locations are pooled into some
            output pixels. It may make the output size larger.
        use_cudnn (bool): If True and CuDNN is enabled, then this function
            uses CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Ouptut variable.

    """
    return MaxPooling2DIndices(ksize, stride, pad, cover_all)(x)
