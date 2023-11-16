from paddle import _legacy_C_ops
from paddle.framework import dygraph_only

@dygraph_only
def _append_bias_in_dygraph(input, bias=None, axis=1, use_mkldnn=False):
    if False:
        i = 10
        return i + 15
    'Append bias operation in dygraph mode.\n\n        Args:\n            input: the input variable.\n            bias:  the bias to be appended\n            axis:  the axis to perform operation\n            use_mkldnn: whether to use mkldnn\n\n    Return the Variable after bias operation\n    '
    if bias is None:
        return input
    return _legacy_C_ops.elementwise_add(input, bias, 'axis', axis, 'use_mkldnn', use_mkldnn)