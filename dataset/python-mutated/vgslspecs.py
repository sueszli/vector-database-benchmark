"""String network description language mapping to TF-Slim calls where possible.

See vglspecs.md for detailed description.
"""
import re
from string import maketrans
import nn_ops
import shapes
from six.moves import xrange
import tensorflow as tf
import tensorflow.contrib.slim as slim

class VGSLSpecs(object):
    """Layers that can be built from a string definition."""

    def __init__(self, widths, heights, is_training):
        if False:
            while True:
                i = 10
        'Constructs a VGSLSpecs.\n\n    Args:\n      widths:  Tensor of size batch_size of the widths of the inputs.\n      heights: Tensor of size batch_size of the heights of the inputs.\n      is_training: True if the graph should be build for training.\n    '
        self.model_str = None
        self.is_training = is_training
        self.widths = widths
        self.heights = heights
        self.reduction_factors = [1.0, 1.0, 1.0, 1.0]
        self.valid_ops = [self.AddSeries, self.AddParallel, self.AddConvLayer, self.AddMaxPool, self.AddDropout, self.AddReShape, self.AddFCLayer, self.AddLSTMLayer]
        self.transtab = maketrans('(,)', '___')

    def Build(self, prev_layer, model_str):
        if False:
            while True:
                i = 10
        'Builds a network with input prev_layer from a VGSLSpecs description.\n\n    Args:\n      prev_layer: The input tensor.\n      model_str:  Model definition similar to Tesseract as follows:\n        ============ FUNCTIONAL OPS ============\n        C(s|t|r|l|m)[{name}]<y>,<x>,<d> Convolves using a y,x window, with no\n          shrinkage, SAME infill, d outputs, with s|t|r|l|m non-linear layer.\n          (s|t|r|l|m) specifies the type of non-linearity:\n          s = sigmoid\n          t = tanh\n          r = relu\n          l = linear (i.e., None)\n          m = softmax\n        F(s|t|r|l|m)[{name}]<d> Fully-connected with s|t|r|l|m non-linearity and\n          d outputs. Reduces height, width to 1. Input height and width must be\n          constant.\n        L(f|r|b)(x|y)[s][{name}]<n> LSTM cell with n outputs.\n          f runs the LSTM forward only.\n          r runs the LSTM reversed only.\n          b runs the LSTM bidirectionally.\n          x runs the LSTM in the x-dimension (on data with or without the\n             y-dimension).\n          y runs the LSTM in the y-dimension (data must have a y dimension).\n          s (optional) summarizes the output in the requested dimension,\n             outputting only the final step, collapsing the dimension to a\n             single element.\n          Examples:\n          Lfx128 runs a forward-only LSTM in the x-dimension with 128\n                 outputs, treating any y dimension independently.\n          Lfys64 runs a forward-only LSTM in the y-dimension with 64 outputs\n                 and collapses the y-dimension to 1 element.\n          NOTE that Lbxsn is implemented as (LfxsnLrxsn) since the summaries\n          need to be taken from opposite ends of the output\n        Do[{name}] Insert a dropout layer.\n        ============ PLUMBING OPS ============\n        [...] Execute ... networks in series (layers).\n        (...) Execute ... networks in parallel, with their output concatenated\n          in depth.\n        S[{name}]<d>(<a>x<b>)<e>,<f> Splits one dimension, moves one part to\n          another dimension.\n          Splits input dimension d into a x b, sending the high part (a) to the\n          high side of dimension e, and the low part (b) to the high side of\n          dimension f. Exception: if d=e=f, then then dimension d is internally\n          transposed to bxa.\n          Either a or b can be zero, meaning whatever is left after taking out\n          the other, allowing dimensions to be of variable size.\n          Eg. S3(3x50)2,3 will split the 150-element depth into 3x50, with the 3\n          going to the most significant part of the width, and the 50 part\n          staying in depth.\n          This will rearrange a 3x50 output parallel operation to spread the 3\n          output sets over width.\n        Mp[{name}]<y>,<x> Maxpool the input, reducing the (y,x) rectangle to a\n          single vector value.\n\n    Returns:\n      Output tensor\n    '
        self.model_str = model_str
        (final_layer, _) = self.BuildFromString(prev_layer, 0)
        return final_layer

    def GetLengths(self, dim=2, factor=1):
        if False:
            print('Hello World!')
        "Returns the lengths of the batch of elements in the given dimension.\n\n    WARNING: The returned sizes may not exactly match TF's calculation.\n    Args:\n      dim: dimension to get the sizes of, in [1,2]. batch, depth not allowed.\n      factor: A scalar value to multiply by.\n\n    Returns:\n      The original heights/widths scaled by the current scaling of the model and\n      the given factor.\n\n    Raises:\n      ValueError: If the args are invalid.\n    "
        if dim == 1:
            lengths = self.heights
        elif dim == 2:
            lengths = self.widths
        else:
            raise ValueError('Invalid dimension given to GetLengths')
        lengths = tf.cast(lengths, tf.float32)
        if self.reduction_factors[dim] is not None:
            lengths = tf.div(lengths, self.reduction_factors[dim])
        else:
            lengths = tf.ones_like(lengths)
        if factor != 1:
            lengths = tf.multiply(lengths, tf.cast(factor, tf.float32))
        return tf.cast(lengths, tf.int32)

    def BuildFromString(self, prev_layer, index):
        if False:
            i = 10
            return i + 15
        'Adds the layers defined by model_str[index:] to the model.\n\n    Args:\n      prev_layer: Input tensor.\n      index:      Position in model_str to start parsing\n\n    Returns:\n      Output tensor, next model_str index.\n\n    Raises:\n      ValueError: If the model string is unrecognized.\n    '
        index = self._SkipWhitespace(index)
        for op in self.valid_ops:
            (output_layer, next_index) = op(prev_layer, index)
            if output_layer is not None:
                return (output_layer, next_index)
        if output_layer is not None:
            return (output_layer, next_index)
        raise ValueError('Unrecognized model string:' + self.model_str[index:])

    def AddSeries(self, prev_layer, index):
        if False:
            return 10
        'Builds a sequence of layers for a VGSLSpecs model.\n\n    Args:\n      prev_layer: Input tensor.\n      index:      Position in model_str to start parsing\n\n    Returns:\n      Output tensor of the series, end index in model_str.\n\n    Raises:\n      ValueError: If [] are unbalanced.\n    '
        if self.model_str[index] != '[':
            return (None, None)
        index += 1
        while index < len(self.model_str) and self.model_str[index] != ']':
            (prev_layer, index) = self.BuildFromString(prev_layer, index)
        if index == len(self.model_str):
            raise ValueError('Missing ] at end of series!' + self.model_str)
        return (prev_layer, index + 1)

    def AddParallel(self, prev_layer, index):
        if False:
            return 10
        "tf.concats outputs of layers that run on the same inputs.\n\n    Args:\n      prev_layer: Input tensor.\n      index:      Position in model_str to start parsing\n\n    Returns:\n      Output tensor of the parallel,  end index in model_str.\n\n    Raises:\n      ValueError: If () are unbalanced or the elements don't match.\n    "
        if self.model_str[index] != '(':
            return (None, None)
        index += 1
        layers = []
        num_dims = 0
        original_factors = self.reduction_factors
        final_factors = None
        while index < len(self.model_str) and self.model_str[index] != ')':
            self.reduction_factors = original_factors
            (layer, index) = self.BuildFromString(prev_layer, index)
            if num_dims == 0:
                num_dims = len(layer.get_shape())
            elif num_dims != len(layer.get_shape()):
                raise ValueError('All elements of parallel must return same num dims')
            layers.append(layer)
            if final_factors:
                if final_factors != self.reduction_factors:
                    raise ValueError('All elements of parallel must scale the same')
            else:
                final_factors = self.reduction_factors
        if index == len(self.model_str):
            raise ValueError('Missing ) at end of parallel!' + self.model_str)
        return (tf.concat(axis=num_dims - 1, values=layers), index + 1)

    def AddConvLayer(self, prev_layer, index):
        if False:
            print('Hello World!')
        'Add a single standard convolutional layer.\n\n    Args:\n      prev_layer: Input tensor.\n      index:      Position in model_str to start parsing\n\n    Returns:\n      Output tensor, end index in model_str.\n    '
        pattern = re.compile('(C)(s|t|r|l|m)({\\w+})?(\\d+),(\\d+),(\\d+)')
        m = pattern.match(self.model_str, index)
        if m is None:
            return (None, None)
        name = self._GetLayerName(m.group(0), index, m.group(3))
        width = int(m.group(4))
        height = int(m.group(5))
        depth = int(m.group(6))
        fn = self._NonLinearity(m.group(2))
        return (slim.conv2d(prev_layer, depth, [height, width], activation_fn=fn, scope=name), m.end())

    def AddMaxPool(self, prev_layer, index):
        if False:
            return 10
        'Add a maxpool layer.\n\n    Args:\n      prev_layer: Input tensor.\n      index:      Position in model_str to start parsing\n\n    Returns:\n      Output tensor, end index in model_str.\n    '
        pattern = re.compile('(Mp)({\\w+})?(\\d+),(\\d+)(?:,(\\d+),(\\d+))?')
        m = pattern.match(self.model_str, index)
        if m is None:
            return (None, None)
        name = self._GetLayerName(m.group(0), index, m.group(2))
        height = int(m.group(3))
        width = int(m.group(4))
        y_stride = height if m.group(5) is None else m.group(5)
        x_stride = width if m.group(6) is None else m.group(6)
        self.reduction_factors[1] *= y_stride
        self.reduction_factors[2] *= x_stride
        return (slim.max_pool2d(prev_layer, [height, width], [y_stride, x_stride], padding='SAME', scope=name), m.end())

    def AddDropout(self, prev_layer, index):
        if False:
            i = 10
            return i + 15
        'Adds a dropout layer.\n\n    Args:\n      prev_layer: Input tensor.\n      index:      Position in model_str to start parsing\n\n    Returns:\n      Output tensor, end index in model_str.\n    '
        pattern = re.compile('(Do)({\\w+})?')
        m = pattern.match(self.model_str, index)
        if m is None:
            return (None, None)
        name = self._GetLayerName(m.group(0), index, m.group(2))
        layer = slim.dropout(prev_layer, 0.5, is_training=self.is_training, scope=name)
        return (layer, m.end())

    def AddReShape(self, prev_layer, index):
        if False:
            while True:
                i = 10
        'Reshapes the input tensor by moving each (x_scale,y_scale) rectangle to.\n\n       the depth dimension. NOTE that the TF convention is that inputs are\n       [batch, y, x, depth].\n\n    Args:\n      prev_layer: Input tensor.\n      index:      Position in model_str to start parsing\n\n    Returns:\n      Output tensor, end index in model_str.\n    '
        pattern = re.compile('(S)(?:{(\\w)})?(\\d+)\\((\\d+)x(\\d+)\\)(\\d+),(\\d+)')
        m = pattern.match(self.model_str, index)
        if m is None:
            return (None, None)
        name = self._GetLayerName(m.group(0), index, m.group(2))
        src_dim = int(m.group(3))
        part_a = int(m.group(4))
        part_b = int(m.group(5))
        dest_dim_a = int(m.group(6))
        dest_dim_b = int(m.group(7))
        if part_a == 0:
            part_a = -1
        if part_b == 0:
            part_b = -1
        prev_shape = tf.shape(prev_layer)
        layer = shapes.transposing_reshape(prev_layer, src_dim, part_a, part_b, dest_dim_a, dest_dim_b, name=name)
        result_shape = tf.shape(layer)
        for i in xrange(len(self.reduction_factors)):
            if self.reduction_factors[i] is not None:
                factor1 = tf.cast(self.reduction_factors[i], tf.float32)
                factor2 = tf.cast(prev_shape[i], tf.float32)
                divisor = tf.cast(result_shape[i], tf.float32)
                self.reduction_factors[i] = tf.div(tf.multiply(factor1, factor2), divisor)
        return (layer, m.end())

    def AddFCLayer(self, prev_layer, index):
        if False:
            for i in range(10):
                print('nop')
        'Parse expression and add Fully Connected Layer.\n\n    Args:\n      prev_layer: Input tensor.\n      index:      Position in model_str to start parsing\n\n    Returns:\n      Output tensor, end index in model_str.\n    '
        pattern = re.compile('(F)(s|t|r|l|m)({\\w+})?(\\d+)')
        m = pattern.match(self.model_str, index)
        if m is None:
            return (None, None)
        fn = self._NonLinearity(m.group(2))
        name = self._GetLayerName(m.group(0), index, m.group(3))
        depth = int(m.group(4))
        input_depth = shapes.tensor_dim(prev_layer, 1) * shapes.tensor_dim(prev_layer, 2) * shapes.tensor_dim(prev_layer, 3)
        shaped = tf.reshape(prev_layer, [-1, input_depth], name=name + '_reshape_in')
        output = slim.fully_connected(shaped, depth, activation_fn=fn, scope=name)
        self.reduction_factors[1] = None
        self.reduction_factors[2] = None
        return (tf.reshape(output, [shapes.tensor_dim(prev_layer, 0), 1, 1, depth], name=name + '_reshape_out'), m.end())

    def AddLSTMLayer(self, prev_layer, index):
        if False:
            print('Hello World!')
        'Parse expression and add LSTM Layer.\n\n    Args:\n      prev_layer: Input tensor.\n      index:      Position in model_str to start parsing\n\n    Returns:\n      Output tensor, end index in model_str.\n    '
        pattern = re.compile('(L)(f|r|b)(x|y)(s)?({\\w+})?(\\d+)')
        m = pattern.match(self.model_str, index)
        if m is None:
            return (None, None)
        direction = m.group(2)
        dim = m.group(3)
        summarize = m.group(4) == 's'
        name = self._GetLayerName(m.group(0), index, m.group(5))
        depth = int(m.group(6))
        if direction == 'b' and summarize:
            fwd = self._LSTMLayer(prev_layer, 'forward', dim, True, depth, name + '_forward')
            back = self._LSTMLayer(prev_layer, 'backward', dim, True, depth, name + '_reverse')
            return (tf.concat(axis=3, values=[fwd, back], name=name + '_concat'), m.end())
        if direction == 'f':
            direction = 'forward'
        elif direction == 'r':
            direction = 'backward'
        else:
            direction = 'bidirectional'
        outputs = self._LSTMLayer(prev_layer, direction, dim, summarize, depth, name)
        if summarize:
            if dim == 'x':
                self.reduction_factors[2] = None
            else:
                self.reduction_factors[1] = None
        return (outputs, m.end())

    def _LSTMLayer(self, prev_layer, direction, dim, summarize, depth, name):
        if False:
            while True:
                i = 10
        "Adds an LSTM layer with the given pre-parsed attributes.\n\n    Always maps 4-D to 4-D regardless of summarize.\n    Args:\n      prev_layer: Input tensor.\n      direction:  'forward' 'backward' or 'bidirectional'\n      dim:        'x' or 'y', dimension to consider as time.\n      summarize:  True if we are to return only the last timestep.\n      depth:      Output depth.\n      name:       Some string naming the op.\n\n    Returns:\n      Output tensor.\n    "
        if dim == 'x':
            lengths = self.GetLengths(2, 1)
            inputs = prev_layer
        else:
            lengths = self.GetLengths(1, 1)
            inputs = tf.transpose(prev_layer, [0, 2, 1, 3], name=name + '_ytrans_in')
        input_batch = shapes.tensor_dim(inputs, 0)
        num_slices = shapes.tensor_dim(inputs, 1)
        num_steps = shapes.tensor_dim(inputs, 2)
        input_depth = shapes.tensor_dim(inputs, 3)
        inputs = tf.reshape(inputs, [-1, num_steps, input_depth], name=name + '_reshape_in')
        tile_factor = tf.to_float(input_batch * num_slices) / tf.to_float(tf.shape(lengths)[0])
        lengths = tf.tile(lengths, [tf.cast(tile_factor, tf.int32)])
        lengths = tf.cast(lengths, tf.int64)
        outputs = nn_ops.rnn_helper(inputs, lengths, cell_type='lstm', num_nodes=depth, direction=direction, name=name, stddev=0.1)
        if direction == 'bidirectional':
            output_depth = depth * 2
        else:
            output_depth = depth
        if summarize:
            outputs = tf.slice(outputs, [0, num_steps - 1, 0], [-1, 1, -1], name=name + '_sum_slice')
            outputs = tf.reshape(outputs, [input_batch, num_slices, 1, output_depth], name=name + '_reshape_out')
        else:
            outputs = tf.reshape(outputs, [input_batch, num_slices, num_steps, output_depth], name=name + '_reshape_out')
        if dim == 'y':
            outputs = tf.transpose(outputs, [0, 2, 1, 3], name=name + '_ytrans_out')
        return outputs

    def _NonLinearity(self, code):
        if False:
            print('Hello World!')
        'Returns the non-linearity function pointer for the given string code.\n\n    For forwards compatibility, allows the full names for stand-alone\n    non-linearities, as well as the single-letter names used in ops like C,F.\n    Args:\n      code: String code representing a non-linearity function.\n    Returns:\n      non-linearity function represented by the code.\n    '
        if code in ['s', 'Sig']:
            return tf.sigmoid
        elif code in ['t', 'Tanh']:
            return tf.tanh
        elif code in ['r', 'Relu']:
            return tf.nn.relu
        elif code in ['m', 'Smax']:
            return tf.nn.softmax
        return None

    def _GetLayerName(self, op_str, index, name_str):
        if False:
            print('Hello World!')
        'Generates a name for the op, using a user-supplied name if possible.\n\n    Args:\n      op_str:     String representing the parsed op.\n      index:      Position in model_str of the start of the op.\n      name_str:   User-supplied {name} with {} that need removing or None.\n\n    Returns:\n      Selected name.\n    '
        if name_str:
            return name_str[1:-1]
        else:
            return op_str.translate(self.transtab) + '_' + str(index)

    def _SkipWhitespace(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Skips any leading whitespace in the model description.\n\n    Args:\n      index:      Position in model_str to start parsing\n\n    Returns:\n      end index in model_str of whitespace.\n    '
        pattern = re.compile('([ \\t\\n]+)')
        m = pattern.match(self.model_str, index)
        if m is None:
            return index
        return m.end()