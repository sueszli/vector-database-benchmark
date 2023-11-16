import copy
import json
import logging
import os
import paddle
from ...base.framework import IrGraph, core
from ..log_helper import get_logger
from .quantization_pass import AddQuantDequantPass, ConvertToInt8Pass, OutScaleForInferencePass, OutScaleForTrainingPass, QuantizationFreezePass, QuantizationTransformPass
_logger = get_logger(__name__, level=logging.INFO)
from . import quant_config
from .post_training_quantization import PostTrainingQuantizationProgram
from .quantization_pass import AddQuantDequantForInferencePass, AddQuantDequantPassV2, QuantizationTransformPassV2, QuantWeightPass
WEIGHT_QUANTIZATION_TYPES = ['abs_max', 'channel_wise_abs_max', 'range_abs_max', 'moving_average_abs_max']
WEIGHT_QUANTIZATION_TYPES_TENSORRT = ['channel_wise_abs_max']
ACTIVATION_QUANTIZATION_TYPES = ['abs_max', 'range_abs_max', 'moving_average_abs_max']
ACTIVATION_QUANTIZATION_TYPES_TENSORRT = ['range_abs_max', 'moving_average_abs_max']
VALID_DTYPES = ['int8']
TRANSFORM_PASS_OP_TYPES = list(quant_config.SUPPORT_WEIGHT_QUANTIZATION_OP_DICT.keys())
QUANT_DEQUANT_PASS_OP_TYPES = list(quant_config.SUPPORT_ACT_QUANTIZATION_OP_DICT.keys())
TENSORRT_OP_TYPES = ['mul', 'conv2d', 'pool2d', 'depthwise_conv2d', 'elementwise_add', 'leaky_relu']
VARS_MAPPING_TABLE = './mapping_table_for_saving_inference_model'
_quant_config_default = {'weight_quantize_type': 'channel_wise_abs_max', 'activation_quantize_type': 'moving_average_abs_max', 'weight_bits': 8, 'activation_bits': 8, 'not_quant_pattern': ['skip_quant'], 'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'], 'dtype': 'int8', 'window_size': 10000, 'moving_rate': 0.9, 'for_tensorrt': False, 'is_full_quantize': False, 'onnx_format': True, 'quant_post_first': False, 'scale_trainable': True}

def load_dict():
    if False:
        i = 10
        return i + 15
    with open(VARS_MAPPING_TABLE, 'r') as file:
        data = file.read()
        data = json.loads(data)
        return data

def save_dict(table):
    if False:
        return 10
    with open(VARS_MAPPING_TABLE, 'w') as file:
        file.write(json.dumps(table))

def _parse_configs(user_config):
    if False:
        while True:
            i = 10
    "\n    check if user's configs are valid.\n    Args:\n        user_config(dict): user's config.\n    Return:\n        configs(dict): final configs will be used.\n    "
    configs = copy.deepcopy(_quant_config_default)
    configs.update(user_config)
    assert isinstance(configs['for_tensorrt'], bool) and isinstance(configs['is_full_quantize'], bool), "'for_tensorrt' and 'is_full_quantize' must both be bool'"
    if configs['for_tensorrt']:
        weight_types = WEIGHT_QUANTIZATION_TYPES_TENSORRT
        activation_types = ACTIVATION_QUANTIZATION_TYPES_TENSORRT
        platform = 'TensorRT'
    else:
        weight_types = WEIGHT_QUANTIZATION_TYPES
        activation_types = WEIGHT_QUANTIZATION_TYPES
        platform = 'PaddleLite'
    assert configs['weight_quantize_type'] in weight_types, 'Unknown weight_quantize_type: {}. {} only supports {} '.format(configs['weight_quantize_type'], platform, weight_types)
    assert configs['activation_quantize_type'] in activation_types, 'Unknown activation_quantize_type: {}. {} only supports {}'.format(configs['activation_quantize_type'], platform, activation_types)
    assert isinstance(configs['weight_bits'], int), 'weight_bits must be int value.'
    assert configs['weight_bits'] >= 1 and configs['weight_bits'] <= 16, 'weight_bits should be between 1 and 16.'
    assert isinstance(configs['activation_bits'], int), 'activation_bits must be int value.'
    assert configs['activation_bits'] >= 1 and configs['activation_bits'] <= 16, 'activation_bits should be between 1 and 16.'
    assert isinstance(configs['not_quant_pattern'], (list, str)), 'not_quant_pattern must be list or str'
    assert isinstance(configs['quantize_op_types'], list), 'quantize_op_types must be a list'
    if configs['for_tensorrt']:
        configs['quantize_op_types'] = TENSORRT_OP_TYPES
    elif configs['is_full_quantize']:
        configs['quantize_op_types'] = TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES
    else:
        for op_type in configs['quantize_op_types']:
            assert op_type in QUANT_DEQUANT_PASS_OP_TYPES or op_type in TRANSFORM_PASS_OP_TYPES, '{} is not support,                         now support op types are {}'.format(op_type, TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES)
    assert isinstance(configs['dtype'], str), 'dtype must be a str.'
    assert configs['dtype'] in VALID_DTYPES, 'dtype can only be ' + ' '.join(VALID_DTYPES)
    assert isinstance(configs['window_size'], int), "window_size must be int value, window size for 'range_abs_max' quantization, default is 10000."
    assert isinstance(configs['moving_rate'], float), 'moving_rate must be float value, The decay coefficient of moving average, default is 0.9.'
    return configs

def quant_aware(program, place, config=None, scope=None, for_test=False, weight_quantize_func=None, act_quantize_func=None, weight_preprocess_func=None, act_preprocess_func=None, optimizer_func=None, executor=None, return_program=False, calib_config={}, draw_graph=False, return_scale_dict=False, scale_dict=None, model_type=None, pattern_ops=None):
    if False:
        for i in range(10):
            print('nop')
    'Add quantization  and dequantization operators to "program"\n    for quantization training or testing.\n    Args:\n        program(paddle.static.Program): training or testing ``program``.\n        place(paddle.CPUPlace or paddle.CUDAPlace): This parameter represents\n            the executor run on which device.\n        config(dict, optional): configs for quantization. if None, will use default config.\n            Default: None.\n        scope(paddle.static.Scope): Scope records the mapping between variable names and variables,\n            similar to brackets in programming languages. Usually users can use\n            `paddle.static.global_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_.\n            When ``None`` will use `paddle.static.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_ .\n            Default: ``None``.\n        for_test(bool): If the \'program\' parameter is a test program, this parameter should be set to ``True``.\n            Otherwise, set to ``False``.Default: False\n        weight_quantize_func(function): Function that defines how to quantize weight. Using this\n                can quickly test if user\'s quantization method works or not. In this function, user should\n                both define quantization function and dequantization function, that is, the function\'s input\n                is non-quantized weight and function returns dequantized weight. If None, will use\n                quantization op defined by \'weight_quantize_type\'.\n                Default is None.\n        act_quantize_func(function): Function that defines how to quantize activation. Using this\n                can quickly test if user\'s quantization method works or not. In this function, user should\n                both define quantization and dequantization process, that is, the function\'s input\n                is non-quantized activation and function returns dequantized activation. If None, will use\n                quantization op defined by \'activation_quantize_type\'.\n                Default is None.\n        weight_preprocess_func(function): Function that defines how to preprocess weight before quantization. Using this\n                can quickly test if user\'s preprocess method works or not. The function\'s input\n                is non-quantized weight and function returns processed weight to be quantized. If None, the weight will\n                be quantized directly.\n                Default is None.\n        act_preprocess_func(function): Function that defines how to preprocess activation before quantization. Using this\n                can quickly test if user\'s preprocess method works or not. The function\'s input\n                is non-quantized activation and function returns processed activation to be quantized. If None, the activation will\n                be quantized directly.\n                Default is None.\n        optimizer_func(function): Fuction return a optimizer. When \'is_test\' is False and user want to use self-defined\n            quantization function and preprocess function, this function must be set. Default is None.\n        exe(paddle.static.Executor): If user want to use self-defined quantization function and preprocess function, exe must be set for\n                initialization. Default is None.\n        return_program(bool): If user want return value is a Program rather than Compiled Program, This argument should be set True.\n                Default is False.\n        draw_graph(bool): whether to draw graph when quantization is initialized. In order to prevent cycle,\n                the ERNIE model needs to be set to True. Default is False.\n        return_scale_dict(bool): If user want to return scale dict, model_type and pattern_ops, this argument should be set True.\n                Default is False.\n        scale_dict(dict): Use scale dict to initialize scales in program. Default is None.\n        model_type(str): Model type can be \'transformer\' or \'non-transformer\'. If model type is transformer, patterns will be analyzed.\n                Default is None.\n        pattern_ops(dict): Pattern_ops contain pattern name and corresponding ops. Default is None.\n    Returns:\n        paddle.static.CompiledProgram | paddle.static.Program: Program with quantization and dequantization ``operators``\n    '
    scope = paddle.static.global_scope() if not scope else scope
    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), 'config must be dict'
        config = _parse_configs(config)
    _logger.info(f'quant_aware config {config}')
    skip_tensor_list = []
    same_scale_tensor_list = []
    is_test = True if for_test else not config['scale_trainable']
    if config['quant_post_first'] and for_test:
        if 'quantizable_op_type' not in calib_config:
            calib_config['quantizable_op_type'] = config['quantize_op_types']
        exe = paddle.static.Executor() if executor is None else executor
        post_training_quantization = PostTrainingQuantizationProgram(exe, program, freeze_model=False, skip_tensor_list=skip_tensor_list, same_scale_tensor_list=same_scale_tensor_list, batch_nums=10, scale_dict=scale_dict, return_graph=True, **calib_config)
        main_graph = post_training_quantization.quantize()
        scale_dict = post_training_quantization._scale_dict
        sub_graphs = list(main_graph.all_sub_graphs())
    else:
        main_graph = IrGraph(core.Graph(program.desc), for_test=for_test)
        sub_graphs = list(main_graph.all_sub_graphs())
        transform_pass_ops = []
        quant_dequant_ops = []
        if 'quant_config' in config and config['quant_config']:
            transform_pass_ops = config['quant_config'].weight_quant_operation_types
            quant_dequant_ops = config['quant_config'].activation_quant_operation_types
        else:
            for op_type in config['quantize_op_types']:
                if op_type in TRANSFORM_PASS_OP_TYPES:
                    transform_pass_ops.append(op_type)
                elif op_type in QUANT_DEQUANT_PASS_OP_TYPES:
                    quant_dequant_ops.append(op_type)
        if len(transform_pass_ops) > 0:
            transform_func = QuantizationTransformPassV2 if config['onnx_format'] else QuantizationTransformPass
            transform_pass = transform_func(scope=scope, place=place, weight_bits=config['weight_bits'], activation_bits=config['activation_bits'], activation_quantize_type=config['activation_quantize_type'], weight_quantize_type=config['weight_quantize_type'], window_size=config['window_size'], moving_rate=config['moving_rate'], quantizable_op_type=transform_pass_ops, skip_pattern=config['not_quant_pattern'], weight_quantize_func=weight_quantize_func, act_quantize_func=act_quantize_func, weight_preprocess_func=weight_preprocess_func, act_preprocess_func=act_preprocess_func, optimizer_func=optimizer_func, executor=executor, is_test=is_test)
            for sub_graph in sub_graphs:
                transform_pass.apply(sub_graph)
        if len(quant_dequant_ops) > 0:
            qdq_func = AddQuantDequantPassV2 if config['onnx_format'] else AddQuantDequantPass
            quant_dequant_pass = qdq_func(scope=scope, place=place, moving_rate=config['moving_rate'], quant_bits=config['activation_bits'], skip_pattern=config['not_quant_pattern'], quantizable_op_type=quant_dequant_ops, is_test=is_test)
            for sub_graph in sub_graphs:
                quant_dequant_pass.apply(sub_graph)
    out_scale_training_pass = OutScaleForTrainingPass(scope=scope, place=place, moving_rate=config['moving_rate'], is_test=is_test, scale_dict=scale_dict)
    for sub_graph in sub_graphs:
        out_scale_training_pass.apply(sub_graph)
    if (weight_preprocess_func is not None or act_preprocess_func is not None) and (not for_test) and (not config['onnx_format']):
        _logger.info('When a preprocess_func is used in quant_aware, Need to save a mapping table to match variable names in the convert phase.')
        _logger.info(f"The mapping table is saved as '{VARS_MAPPING_TABLE}'.")
        for sub_graph in sub_graphs:
            save_dict(sub_graph.out_node_mapping_table)
    if draw_graph:
        main_graph.draw('./', 'graph.pdf')
    if for_test or return_program:
        quant_program = main_graph.to_program()
    else:
        quant_program = paddle.static.CompiledProgram(main_graph.graph)
    if return_scale_dict:
        return (quant_program, scale_dict, model_type, pattern_ops)
    else:
        return quant_program

def convert(program, place, config=None, scope=None, save_int8=False):
    if False:
        return 10
    "\n    convert quantized and well-trained ``program`` to final  quantized\n    ``program``that can be used to  save ``inference model``.\n\n    Args:\n        program(paddle.static.Program): quantized and well-trained ``test program``.\n        place(paddle.CPUPlace or paddle.CUDAPlace): This parameter represents\n                the executor run on which device.\n        config(dict, optional): configs for convert. if set None, will use\n                default config. It must be same with config that used in\n                'quant_aware'. Default is None.\n        scope(paddle.static.Scope, optional):  Scope records the mapping between\n                variable names and variables, similar to brackets in\n                programming languages. Usually users can use\n                `paddle.static.global_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_.\n                When ``None`` will use\n                `paddle.static.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_\n                . Default: ``None``.\n        save_int8: Whether to return ``program`` which model parameters'\n                dtype is ``int8``. This parameter can only be used to\n                get model size. Default: ``False``.\n    Returns:\n        Tuple : freezed program which can be used for inference.\n                when ``save_int8`` is False, return ``freezed_program(paddle.static.Program)``.\n                when ``save_int8`` is True, return ``freezed_program(paddle.static.Program)``\n                and ``freezed_program_int8(paddle.static.Program)``\n    "
    scope = paddle.static.global_scope() if not scope else scope
    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), 'config must be dict'
        config = _parse_configs(config)
    _logger.info(f'convert config {config}')
    test_graph = IrGraph(core.Graph(program.desc), for_test=True)
    if config['onnx_format']:
        quant_weight_pass = QuantWeightPass(scope, place)
        for sub_graph in test_graph.all_sub_graphs():
            quant_weight_pass.apply(sub_graph)
        out_scale_infer_pass = AddQuantDequantForInferencePass(scope=scope, place=place, quant_bits=config['activation_bits'])
        for sub_graph in test_graph.all_sub_graphs():
            out_scale_infer_pass.apply(sub_graph)
    else:
        out_scale_infer_pass = OutScaleForInferencePass(scope=scope)
        for sub_graph in test_graph.all_sub_graphs():
            out_scale_infer_pass.apply(sub_graph)
        freeze_pass = QuantizationFreezePass(scope=scope, place=place, weight_bits=config['weight_bits'], activation_bits=config['activation_bits'], weight_quantize_type=config['weight_quantize_type'])
        if os.path.exists(VARS_MAPPING_TABLE):
            test_graph.out_node_mapping_table = load_dict()
        for sub_graph in test_graph.all_sub_graphs():
            freeze_pass.apply(sub_graph)
    freezed_program = test_graph.to_program()
    global_block = freezed_program.global_block()
    for _op in global_block.ops:
        if _op.type == 'while':
            _block_id = _op.attr('sub_block').id
            _block = freezed_program.block(_block_id)
            persistables = []
            for (_name, _var) in _block.vars.items():
                if _var.persistable:
                    global_block._clone_variable(_var)
                    persistables.append(_name)
            for _name in persistables:
                _block._remove_var(_name)
            persistables.extend(_op.input('X'))
            _op.desc.set_input('X', persistables)
    assert not (save_int8 and config['onnx_format']), "When onnx_format=True, already saved int8 weight,so you can't set save_int8=True."
    if save_int8:
        convert_int8_pass = ConvertToInt8Pass(scope=scope, place=place)
        for sub_graph in test_graph.all_sub_graphs():
            convert_int8_pass.apply(sub_graph)
        freezed_program_int8 = test_graph.to_program()
        return (freezed_program, freezed_program_int8)
    else:
        return freezed_program