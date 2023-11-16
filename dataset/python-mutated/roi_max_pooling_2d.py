import numbers
import numpy
import six
import chainer
from chainer.backends import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer.functions.pooling.roi_pooling_2d import _roi_pooling_slice

def _pair(x):
    if False:
        i = 10
        return i + 15
    if isinstance(x, chainer.utils.collections_abc.Iterable):
        return x
    return (x, x)

class ROIMaxPooling2D(function.Function):
    """RoI max pooling over a set of 2d planes."""

    def __init__(self, outsize, spatial_scale):
        if False:
            for i in range(10):
                print('nop')
        (outh, outw) = _pair(outsize)
        if not (isinstance(outh, numbers.Integral) and outh > 0):
            raise TypeError('outsize[0] must be positive integer: {}, {}'.format(type(outh), outh))
        if not (isinstance(outw, numbers.Integral) and outw > 0):
            raise TypeError('outsize[1] must be positive integer: {}, {}'.format(type(outw), outw))
        if isinstance(spatial_scale, numbers.Integral):
            spatial_scale = float(spatial_scale)
        if not (isinstance(spatial_scale, numbers.Real) and spatial_scale > 0):
            raise TypeError('spatial_scale must be a positive float number: {}, {}'.format(type(spatial_scale), spatial_scale))
        (self.outh, self.outw) = (outh, outw)
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check.expect(in_types.size() == 3)
        (x_type, roi_type, roi_index_type) = in_types
        type_check.expect(x_type.dtype.kind == 'f', x_type.ndim == 4, x_type.dtype == roi_type.dtype, roi_type.ndim == 2, roi_type.shape[1] == 4, roi_index_type.dtype == numpy.int32, roi_index_type.ndim == 1, roi_type.shape[0] == roi_index_type.shape[0])

    def forward_cpu(self, inputs):
        if False:
            i = 10
            return i + 15
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape
        (bottom_data, bottom_rois, bottom_roi_indices) = inputs
        (channels, height, width) = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = numpy.full((n_rois, channels, self.outh, self.outw), -numpy.inf, dtype=bottom_data.dtype)
        self.argmax_data = -numpy.ones(top_data.shape, numpy.int32)
        for i_roi in six.moves.range(n_rois):
            idx = bottom_roi_indices[i_roi]
            (ymin, xmin, ymax, xmax) = bottom_rois[i_roi]
            ymin = int(round(ymin * self.spatial_scale))
            xmin = int(round(xmin * self.spatial_scale))
            ymax = int(round(ymax * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))
            roi_height = max(ymax - ymin, 1)
            roi_width = max(xmax - xmin, 1)
            strideh = 1.0 * roi_height / self.outh
            stridew = 1.0 * roi_width / self.outw
            for outh in six.moves.range(self.outh):
                (sliceh, lenh) = _roi_pooling_slice(outh, strideh, height, ymin)
                if sliceh.stop <= sliceh.start:
                    continue
                for outw in six.moves.range(self.outw):
                    (slicew, lenw) = _roi_pooling_slice(outw, stridew, width, xmin)
                    if slicew.stop <= slicew.start:
                        continue
                    roi_data = bottom_data[int(idx), :, sliceh, slicew].reshape(channels, -1)
                    top_data[i_roi, :, outh, outw] = numpy.max(roi_data, axis=1)
                    max_idx_slice = numpy.unravel_index(numpy.argmax(roi_data, axis=1), (lenh, lenw))
                    max_idx_slice_h = max_idx_slice[0] + sliceh.start
                    max_idx_slice_w = max_idx_slice[1] + slicew.start
                    max_idx_slice = max_idx_slice_h * width + max_idx_slice_w
                    self.argmax_data[i_roi, :, outh, outw] = max_idx_slice
        return (top_data,)

    def forward_gpu(self, inputs):
        if False:
            i = 10
            return i + 15
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape
        (bottom_data, bottom_rois, bottom_roi_indices) = inputs
        (channels, height, width) = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh, self.outw), dtype=bottom_data.dtype)
        self.argmax_data = cuda.cupy.empty(top_data.shape, numpy.int32)
        cuda.elementwise('\n            raw T bottom_data, raw T bottom_rois, raw int32 bottom_roi_indices,\n            T spatial_scale, int32 channels, int32 height, int32 width,\n            int32 pooled_height, int32 pooled_width\n            ', 'T top_data, int32 argmax_data', "\n            // pos in output filter\n            int pw = i % pooled_width;\n            int ph = (i / pooled_width) % pooled_height;\n            int c = (i / pooled_width / pooled_height) % channels;\n            int n = i / pooled_width / pooled_height / channels;\n\n            int roi_batch_ind = bottom_roi_indices[n];\n            int roi_start_h = round(bottom_rois[n * 4 + 0] * spatial_scale);\n            int roi_start_w = round(bottom_rois[n * 4 + 1] * spatial_scale);\n            int roi_end_h = round(bottom_rois[n * 4 + 2] * spatial_scale);\n            int roi_end_w = round(bottom_rois[n * 4 + 3] * spatial_scale);\n\n            // Force malformed ROIs to be 1x1\n            int roi_height = max(roi_end_h - roi_start_h , 1);\n            int roi_width = max(roi_end_w - roi_start_w, 1);\n            T bin_size_h = static_cast<T>(roi_height)\n                           / static_cast<T>(pooled_height);\n            T bin_size_w = static_cast<T>(roi_width)\n                           / static_cast<T>(pooled_width);\n\n            int hstart = static_cast<int>(floor(static_cast<T>(ph)\n                                          * bin_size_h));\n            int wstart = static_cast<int>(floor(static_cast<T>(pw)\n                                          * bin_size_w));\n            int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)\n                                        * bin_size_h));\n            int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)\n                                        * bin_size_w));\n\n            // Add roi offsets and clip to input boundaries\n            hstart = min(max(hstart + roi_start_h, 0), height);\n            hend = min(max(hend + roi_start_h, 0), height);\n            wstart = min(max(wstart + roi_start_w, 0), width);\n            wend = min(max(wend + roi_start_w, 0), width);\n\n            // Define an empty pooling region to be zero\n            T maxval = - (T) (1.0 / 0.0);\n            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd\n            int maxidx = -1;\n            int data_offset = (roi_batch_ind * channels + c) * height * width;\n            for (int h = hstart; h < hend; ++h) {\n                for (int w = wstart; w < wend; ++w) {\n                    int bottom_index = h * width + w;\n                    if (bottom_data[data_offset + bottom_index] > maxval) {\n                        maxval = bottom_data[data_offset + bottom_index];\n                        maxidx = bottom_index;\n                    }\n                }\n            }\n            top_data = maxval;\n            argmax_data = maxidx;\n            ", 'roi_max_pooling_2d_fwd')(bottom_data, bottom_rois, bottom_roi_indices, self.spatial_scale, channels, height, width, self.outh, self.outw, top_data, self.argmax_data)
        return (top_data,)

    def backward_cpu(self, inputs, gy):
        if False:
            for i in range(10):
                print('nop')
        (bottom_rois, bottom_roi_indices) = inputs[1:]
        (channels, height, width) = self._bottom_data_shape[1:]
        bottom_diff = numpy.zeros(self._bottom_data_shape, bottom_rois.dtype)
        pooled_height = self.outh
        pooled_width = self.outw
        top_diff = gy[0]
        for i in six.moves.range(top_diff.size):
            pw = i % pooled_width
            ph = int(i / pooled_width) % pooled_height
            c = int(i / pooled_width / pooled_height) % channels
            n = int(i / pooled_width / pooled_height / channels)
            roi_batch_ind = int(bottom_roi_indices[n])
            max_idx = self.argmax_data[n, c, ph, pw]
            h = int(max_idx / width)
            w = max_idx % width
            if max_idx != -1:
                bottom_diff[roi_batch_ind, c, h, w] += top_diff[n, c, ph, pw]
        return (bottom_diff, None, None)

    def backward_gpu(self, inputs, gy):
        if False:
            return 10
        utils.nondeterministic('atomicAdd')
        (bottom_rois, bottom_roi_indices) = inputs[1:]
        (channels, height, width) = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, bottom_rois.dtype)
        cuda.elementwise('\n            raw T top_diff, raw int32 argmax_data,\n            raw T bottom_rois, raw int32 bottom_roi_indices, int32 num_rois,\n            T spatial_scale, int32 channels, int32 height, int32 width,\n            int32 pooled_height, int32 pooled_width\n            ', 'raw T bottom_diff', '\n            int pw = i % pooled_width;\n            int ph = (i / pooled_width) % pooled_height;\n            int c = (i / pooled_width / pooled_height) % channels;\n            int n = i / pooled_width / pooled_height / channels;\n\n            int roi_batch_ind = bottom_roi_indices[n];\n            int bottom_diff_offset =\n                (roi_batch_ind * channels + c) * height * width;\n            int top_diff_offset =\n                (n * channels + c) * pooled_height * pooled_width;\n\n            int max_index =\n                argmax_data[top_diff_offset + ph * pooled_width + pw];\n            if (max_index != -1) {\n                atomicAdd(\n                    &bottom_diff[bottom_diff_offset + max_index],\n                    top_diff[top_diff_offset + ph * pooled_width + pw]);\n            }\n            ', 'roi_max_pooling_2d_bwd')(gy[0], self.argmax_data, bottom_rois, bottom_roi_indices, bottom_rois.shape[0], self.spatial_scale, channels, height, width, self.outh, self.outw, bottom_diff, size=gy[0].size)
        return (bottom_diff, None, None)

def roi_max_pooling_2d(x, rois, roi_indices, outsize, spatial_scale):
    if False:
        return 10
    'Spatial Region of Interest (ROI) max pooling function.\n\n    This function acts similarly to :func:`~chainer.functions.max_pooling_2d`,\n    but it computes the maximum of input spatial patch for each channel with\n    the region of interest.\n\n    Args:\n        x (~chainer.Variable): Input variable. The shape is expected to be\n            4 dimensional: (n: batch, c: channel, h, height, w: width).\n        rois (~chainer.Variable): Input roi variable. The shape is expected to\n            be (n: data size, 4), and each datum is set as below:\n            (y_min, x_min, y_max, x_max).\n        roi_indices (~chainer.Variable): Input roi variable. The shape is\n            expected to be (n: data size, ).\n        outsize ((int, int) or int): Expected output size after pooled\n            (height, width). ``outsize=o`` and ``outsize=(o, o)``\n            are equivalent.\n        spatial_scale (float): Scale of the roi is resized.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    See the original paper proposing ROIPooling:\n    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.\n\n    '
    return ROIMaxPooling2D(outsize, spatial_scale)(x, rois, roi_indices)