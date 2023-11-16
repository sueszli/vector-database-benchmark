import numpy
import six
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check

def _roi_pooling_slice(size, stride, max_size, roi_offset):
    if False:
        for i in range(10):
            print('nop')
    start = int(numpy.floor(size * stride))
    end = int(numpy.ceil((size + 1) * stride))
    start = min(max(start + roi_offset, 0), max_size)
    end = min(max(end + roi_offset, 0), max_size)
    return (slice(start, end), end - start)

class ROIPooling2D(function_node.FunctionNode):
    """RoI pooling over a set of 2d planes."""

    def __init__(self, outh, outw, spatial_scale):
        if False:
            for i in range(10):
                print('nop')
        (self.outh, self.outw) = (outh, outw)
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check.expect(in_types.size() == 2)
        (x_type, roi_type) = in_types
        type_check.expect(x_type.dtype.kind == 'f', x_type.ndim == 4, x_type.dtype == roi_type.dtype, roi_type.ndim == 2, roi_type.shape[1] == 5)

    def forward_cpu(self, inputs):
        if False:
            return 10
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape
        (bottom_data, bottom_rois) = inputs
        (channels, height, width) = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = numpy.zeros((n_rois, channels, self.outh, self.outw), dtype=bottom_data.dtype)
        self.argmax_data = numpy.zeros(top_data.shape, numpy.int32)
        for i_roi in six.moves.range(n_rois):
            (idx, xmin, ymin, xmax, ymax) = bottom_rois[i_roi]
            xmin = int(round(xmin * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))
            ymin = int(round(ymin * self.spatial_scale))
            ymax = int(round(ymax * self.spatial_scale))
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
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
            return 10
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape
        (bottom_data, bottom_rois) = inputs
        (channels, height, width) = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh, self.outw), dtype=bottom_data.dtype)
        self.argmax_data = cuda.cupy.empty(top_data.shape, numpy.int32)
        cuda.elementwise('\n            raw T bottom_data, T spatial_scale, int32 channels,\n            int32 height, int32 width, int32 pooled_height, int32 pooled_width,\n            raw T bottom_rois\n            ', 'T top_data, int32 argmax_data', "\n            // pos in output filter\n            int pw = i % pooled_width;\n            int ph = (i / pooled_width) % pooled_height;\n            int c = (i / pooled_width / pooled_height) % channels;\n            int num = i / pooled_width / pooled_height / channels;\n\n            int roi_batch_ind = bottom_rois[num * 5 + 0];\n            int roi_start_w = round(bottom_rois[num * 5 + 1] * spatial_scale);\n            int roi_start_h = round(bottom_rois[num * 5 + 2] * spatial_scale);\n            int roi_end_w = round(bottom_rois[num * 5 + 3] * spatial_scale);\n            int roi_end_h = round(bottom_rois[num * 5 + 4] * spatial_scale);\n\n            // Force malformed ROIs to be 1x1\n            int roi_width = max(roi_end_w - roi_start_w + 1, 1);\n            int roi_height = max(roi_end_h - roi_start_h + 1, 1);\n            float bin_size_h = static_cast<float>(roi_height)\n                           / static_cast<float>(pooled_height);\n            float bin_size_w = static_cast<float>(roi_width)\n                           / static_cast<float>(pooled_width);\n\n            int hstart = static_cast<int>(floor(static_cast<float>(ph)\n                                          * bin_size_h));\n            int wstart = static_cast<int>(floor(static_cast<float>(pw)\n                                          * bin_size_w));\n            int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)\n                                        * bin_size_h));\n            int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)\n                                        * bin_size_w));\n\n            // Add roi offsets and clip to input boundaries\n            hstart = min(max(hstart + roi_start_h, 0), height);\n            hend = min(max(hend + roi_start_h, 0), height);\n            wstart = min(max(wstart + roi_start_w, 0), width);\n            wend = min(max(wend + roi_start_w, 0), width);\n            bool is_empty = (hend <= hstart) || (wend <= wstart);\n\n            // Define an empty pooling region to be zero\n            float maxval = is_empty ? 0 : -1E+37;\n            // If nothing is pooled, argmax=-1 causes nothing to be backprop'd\n            int maxidx = -1;\n            int data_offset = (roi_batch_ind * channels + c) * height * width;\n            for (int h = hstart; h < hend; ++h) {\n                for (int w = wstart; w < wend; ++w) {\n                    int bottom_index = h * width + w;\n                    if (bottom_data[data_offset + bottom_index] > maxval) {\n                        maxval = bottom_data[data_offset + bottom_index];\n                        maxidx = bottom_index;\n                    }\n                }\n            }\n            top_data = maxval;\n            argmax_data = maxidx;\n            ", 'roi_pooling_2d_fwd')(bottom_data, self.spatial_scale, channels, height, width, self.outh, self.outw, bottom_rois, top_data, self.argmax_data)
        return (top_data,)

    def backward(self, indexes, grad_outputs):
        if False:
            print('Hello World!')
        (bottom_rois,) = self.get_retained_inputs()
        (gtop_data,) = grad_outputs
        f = ROIPooling2DGrad(self.outh, self.outw, self.spatial_scale, self._bottom_data_shape, self.argmax_data)
        return f.apply((bottom_rois, gtop_data))

class ROIPooling2DGrad(function_node.FunctionNode):

    def __init__(self, outh, outw, spatial_scale, bottom_data_shape, argmax_data):
        if False:
            return 10
        (self.outh, self.outw) = (outh, outw)
        self.spatial_scale = spatial_scale
        self._bottom_data_shape = bottom_data_shape
        self.argmax_data = argmax_data

    def forward_cpu(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (bottom_rois, gtop_data) = inputs
        (channels, height, width) = self._bottom_data_shape[1:]
        n_rois = bottom_rois.shape[0]
        bottom_delta = numpy.zeros(self._bottom_data_shape, bottom_rois.dtype)
        for i_roi in six.moves.range(n_rois):
            (idx, xmin, ymin, xmax, ymax) = bottom_rois[i_roi]
            idx = int(idx)
            xmin = int(round(xmin * self.spatial_scale))
            xmax = int(round(xmax * self.spatial_scale))
            ymin = int(round(ymin * self.spatial_scale))
            ymax = int(round(ymax * self.spatial_scale))
            roi_width = max(xmax - xmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
            strideh = float(roi_height) / float(self.outh)
            stridew = float(roi_width) / float(self.outw)
            for w in six.moves.range(xmin, xmax + 1):
                for h in six.moves.range(ymin, ymax + 1):
                    phstart = int(numpy.floor(float(h - ymin) / strideh))
                    phend = int(numpy.ceil(float(h - ymin + 1) / strideh))
                    pwstart = int(numpy.floor(float(w - xmin) / stridew))
                    pwend = int(numpy.ceil(float(w - xmin + 1) / stridew))
                    phstart = min(max(phstart, 0), self.outh)
                    phend = min(max(phend, 0), self.outh)
                    pwstart = min(max(pwstart, 0), self.outw)
                    pwend = min(max(pwend, 0), self.outw)
                    for ph in six.moves.range(phstart, phend):
                        for pw in six.moves.range(pwstart, pwend):
                            max_idx_tmp = self.argmax_data[i_roi, :, ph, pw]
                            for c in six.moves.range(channels):
                                if max_idx_tmp[c] == h * width + w:
                                    bottom_delta[idx, c, h, w] += gtop_data[i_roi, c, ph, pw]
        return (bottom_delta, None)

    def forward_gpu(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (bottom_rois, gtop_data) = inputs
        (channels, height, width) = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, bottom_rois.dtype)
        cuda.elementwise('\n            raw T top_diff, raw int32 argmax_data, int32 num_rois,\n            T spatial_scale, int32 channels, int32 height, int32 width,\n            int32 pooled_height, int32 pooled_width, raw T bottom_rois\n            ', 'T bottom_diff', "\n            int w = i % width;\n            int h = (i / width) % height;\n            int c = (i / (width * height)) % channels;\n            int num = i / (width * height * channels);\n\n            float gradient = 0;\n            // Accumulate gradient over all ROIs that pooled this element\n            for (int roi_n = 0; roi_n < num_rois; ++roi_n) {\n                // Skip if ROI's batch index doesn't match num\n                if (num != static_cast<int>(bottom_rois[roi_n * 5])) {\n                    continue;\n                }\n\n                int roi_start_w = round(bottom_rois[roi_n * 5 + 1]\n                                        * spatial_scale);\n                int roi_start_h = round(bottom_rois[roi_n * 5 + 2]\n                                        * spatial_scale);\n                int roi_end_w = round(bottom_rois[roi_n * 5 + 3]\n                                      * spatial_scale);\n                int roi_end_h = round(bottom_rois[roi_n * 5 + 4]\n                                      * spatial_scale);\n\n                // Skip if ROI doesn't include (h, w)\n                const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&\n                                     h >= roi_start_h && h <= roi_end_h);\n                if (!in_roi) {\n                    continue;\n                }\n\n                int offset = (roi_n * channels + c) * pooled_height\n                             * pooled_width;\n\n                // Compute feasible set of pooled units that could have pooled\n                // this bottom unit\n\n                // Force malformed ROIs to be 1x1\n                int roi_width = max(roi_end_w - roi_start_w + 1, 1);\n                int roi_height = max(roi_end_h - roi_start_h + 1, 1);\n\n                float bin_size_h = static_cast<float>(roi_height)\n                               / static_cast<float>(pooled_height);\n                float bin_size_w = static_cast<float>(roi_width)\n                               / static_cast<float>(pooled_width);\n\n                int phstart = floor(static_cast<float>(h - roi_start_h)\n                                    / bin_size_h);\n                int phend = ceil(static_cast<float>(h - roi_start_h + 1)\n                                 / bin_size_h);\n                int pwstart = floor(static_cast<float>(w - roi_start_w)\n                                    / bin_size_w);\n                int pwend = ceil(static_cast<float>(w - roi_start_w + 1)\n                                 / bin_size_w);\n\n                phstart = min(max(phstart, 0), pooled_height);\n                phend = min(max(phend, 0), pooled_height);\n                pwstart = min(max(pwstart, 0), pooled_width);\n                pwend = min(max(pwend, 0), pooled_width);\n\n                for (int ph = phstart; ph < phend; ++ph) {\n                    for (int pw = pwstart; pw < pwend; ++pw) {\n                        int index_ = ph * pooled_width + pw + offset;\n                        if (argmax_data[index_] == (h * width + w)) {\n                            gradient += top_diff[index_];\n                        }\n                    }\n                }\n            }\n            bottom_diff = gradient;\n            ", 'roi_pooling_2d_bwd')(gtop_data, self.argmax_data, bottom_rois.shape[0], self.spatial_scale, channels, height, width, self.outh, self.outw, bottom_rois, bottom_diff)
        return (bottom_diff, None)

    def backward(self, indexes, grad_outputs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

def roi_pooling_2d(x, rois, outh, outw, spatial_scale):
    if False:
        print('Hello World!')
    'Spatial Region of Interest (ROI) pooling function.\n\n    This function acts similarly to :func:`~chainer.functions.max_pooling_2d`,\n    but it computes the maximum of input spatial patch for each channel with\n    the region of interest.\n\n    Args:\n        x (~chainer.Variable): Input variable. The shape is expected to be\n            4 dimensional: (n: batch, c: channel, h, height, w: width).\n        rois (~chainer.Variable): Input roi variable. The shape is expected to\n            be (n: data size, 5), and each datum is set as below:\n            (batch_index, x_min, y_min, x_max, y_max).\n        outh (int): Height of output image after pooled.\n        outw (int): Width of output image after pooled.\n        spatial_scale (float): Scale of the roi is resized.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    See the original paper proposing ROIPooling:\n    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.\n\n    '
    return ROIPooling2D(outh, outw, spatial_scale).apply((x, rois))[0]