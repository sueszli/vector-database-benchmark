import numpy as np
import paddle
from paddle import base
from paddle.base.layer_helper import LayerHelper
from paddle.base.param_attr import ParamAttr
from paddle.nn import Layer, initializer as I

def resnet_unit(x, filter_x, scale_x, bias_x, mean_x, var_x, z, filter_z, scale_z, bias_z, mean_z, var_z, stride, stride_z, padding, dilation, groups, momentum, eps, data_format, fuse_add, has_shortcut, use_global_stats, is_test, act):
    if False:
        for i in range(10):
            print('nop')
    helper = LayerHelper('resnet_unit', **locals())
    bn_param_dtype = base.core.VarDesc.VarType.FP32
    bit_mask_dtype = base.core.VarDesc.VarType.INT32
    out = helper.create_variable_for_type_inference(x.dtype)
    bit_mask = helper.create_variable_for_type_inference(dtype=bit_mask_dtype, stop_gradient=True)
    conv_x = helper.create_variable_for_type_inference(dtype=x.dtype, stop_gradient=True)
    saved_mean_x = helper.create_variable_for_type_inference(dtype=bn_param_dtype, stop_gradient=True)
    saved_invstd_x = helper.create_variable_for_type_inference(dtype=bn_param_dtype, stop_gradient=True)
    running_mean_x = mean_x
    running_var_x = var_x
    conv_z = helper.create_variable_for_type_inference(dtype=x.dtype, stop_gradient=True)
    saved_mean_z = helper.create_variable_for_type_inference(dtype=bn_param_dtype, stop_gradient=True)
    saved_invstd_z = helper.create_variable_for_type_inference(dtype=bn_param_dtype, stop_gradient=True)
    running_mean_z = helper.create_variable_for_type_inference(dtype=bn_param_dtype, stop_gradient=True) if mean_z is None else mean_z
    running_var_z = helper.create_variable_for_type_inference(dtype=bn_param_dtype, stop_gradient=True) if var_z is None else var_z
    inputs = {'X': x, 'FilterX': filter_x, 'ScaleX': scale_x, 'BiasX': bias_x, 'MeanX': mean_x, 'VarX': var_x, 'Z': z, 'FilterZ': filter_z, 'ScaleZ': scale_z, 'BiasZ': bias_z, 'MeanZ': mean_z, 'VarZ': var_z}
    attrs = {'stride': stride, 'stride_z': stride_z, 'padding': padding, 'dilation': dilation, 'group': groups, 'momentum': momentum, 'epsilon': eps, 'data_format': data_format, 'fuse_add': fuse_add, 'has_shortcut': has_shortcut, 'use_global_stats': use_global_stats, 'is_test': is_test, 'act_type': act}
    outputs = {'Y': out, 'BitMask': bit_mask, 'ConvX': conv_x, 'SavedMeanX': saved_mean_x, 'SavedInvstdX': saved_invstd_x, 'RunningMeanX': running_mean_x, 'RunningVarX': running_var_x, 'ConvZ': conv_z, 'SavedMeanZ': saved_mean_z, 'SavedInvstdZ': saved_invstd_z, 'RunningMeanZ': running_mean_z, 'RunningVarZ': running_var_z}
    helper.append_op(type='resnet_unit', inputs=inputs, outputs=outputs, attrs=attrs)
    return out

class ResNetUnit(Layer):
    """
    ******Temporary version******.
    ResNetUnit is designed for optimize the performence by using cudnnv8 API.
    """

    def __init__(self, num_channels_x, num_filters, filter_size, stride=1, momentum=0.9, eps=1e-05, data_format='NHWC', act='relu', fuse_add=False, has_shortcut=False, use_global_stats=False, is_test=False, filter_x_attr=None, scale_x_attr=None, bias_x_attr=None, moving_mean_x_name=None, moving_var_x_name=None, num_channels_z=1, stride_z=1, filter_z_attr=None, scale_z_attr=None, bias_z_attr=None, moving_mean_z_name=None, moving_var_z_name=None):
        if False:
            return 10
        super().__init__()
        self._stride = stride
        self._stride_z = stride_z
        self._dilation = 1
        self._kernel_size = paddle.utils.convert_to_list(filter_size, 2, 'kernel_size')
        self._padding = (filter_size - 1) // 2
        self._groups = 1
        self._momentum = momentum
        self._eps = eps
        self._data_format = data_format
        self._act = act
        self._fuse_add = fuse_add
        self._has_shortcut = has_shortcut
        self._use_global_stats = use_global_stats
        self._is_test = is_test
        valid_format = {'NHWC', 'NCHW'}
        if data_format not in valid_format:
            raise ValueError("conv_format must be one of {}, but got conv_format='{}'".format(valid_format, data_format))

        def _get_default_param_initializer(channels):
            if False:
                for i in range(10):
                    print('nop')
            filter_elem_num = np.prod(self._kernel_size) * channels
            std = (2.0 / filter_elem_num) ** 0.5
            return I.Normal(0.0, std)
        is_nchw = data_format == 'NCHW'
        bn_param_dtype = base.core.VarDesc.VarType.FP32
        if not is_nchw:
            bn_param_shape = [1, 1, 1, num_filters]
            filter_x_shape = [num_filters, filter_size, filter_size, num_channels_x]
            filter_z_shape = [num_filters, filter_size, filter_size, num_channels_z]
        else:
            bn_param_shape = [1, num_filters, 1, 1]
            filter_x_shape = [num_filters, num_channels_x, filter_size, filter_size]
            filter_z_shape = [num_filters, num_channels_z, filter_size, filter_size]
        self.filter_x = self.create_parameter(shape=filter_x_shape, attr=filter_x_attr, default_initializer=_get_default_param_initializer(num_channels_x))
        self.scale_x = self.create_parameter(shape=bn_param_shape, attr=scale_x_attr, dtype=bn_param_dtype, default_initializer=I.Constant(1.0))
        self.bias_x = self.create_parameter(shape=bn_param_shape, attr=bias_x_attr, dtype=bn_param_dtype, is_bias=True)
        self.mean_x = self.create_parameter(attr=ParamAttr(name=moving_mean_x_name, initializer=I.Constant(0.0), trainable=False), shape=bn_param_shape, dtype=bn_param_dtype)
        self.mean_x.stop_gradient = True
        self.var_x = self.create_parameter(attr=ParamAttr(name=moving_var_x_name, initializer=I.Constant(1.0), trainable=False), shape=bn_param_shape, dtype=bn_param_dtype)
        self.var_x.stop_gradient = True
        if has_shortcut:
            self.filter_z = self.create_parameter(shape=filter_z_shape, attr=filter_z_attr, default_initializer=_get_default_param_initializer(num_channels_z))
            self.scale_z = self.create_parameter(shape=bn_param_shape, attr=scale_z_attr, dtype=bn_param_dtype, default_initializer=I.Constant(1.0))
            self.bias_z = self.create_parameter(shape=bn_param_shape, attr=bias_z_attr, dtype=bn_param_dtype, is_bias=True)
            self.mean_z = self.create_parameter(attr=ParamAttr(name=moving_mean_z_name, initializer=I.Constant(0.0), trainable=False), shape=bn_param_shape, dtype=bn_param_dtype)
            self.mean_z.stop_gradient = True
            self.var_z = self.create_parameter(attr=ParamAttr(name=moving_var_z_name, initializer=I.Constant(1.0), trainable=False), shape=bn_param_shape, dtype=bn_param_dtype)
            self.var_z.stop_gradient = True
        else:
            self.filter_z = None
            self.scale_z = None
            self.bias_z = None
            self.mean_z = None
            self.var_z = None

    def forward(self, x, z=None):
        if False:
            i = 10
            return i + 15
        if self._fuse_add and z is None:
            raise ValueError('z can not be None')
        out = resnet_unit(x, self.filter_x, self.scale_x, self.bias_x, self.mean_x, self.var_x, z, self.filter_z, self.scale_z, self.bias_z, self.mean_z, self.var_z, self._stride, self._stride_z, self._padding, self._dilation, self._groups, self._momentum, self._eps, self._data_format, self._fuse_add, self._has_shortcut, self._use_global_stats, self._is_test, self._act)
        return out