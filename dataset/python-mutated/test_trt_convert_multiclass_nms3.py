import unittest
from functools import partial
from typing import Dict, List
import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest
import paddle.inference as paddle_infer

class TrtConvertMulticlassNMS3Test(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def create_inference_config(self, use_trt=True) -> paddle_infer.Config:
        if False:
            return 10
        if use_trt:
            config = paddle_infer.Config()
            config.disable_glog_info()
            config.enable_use_gpu(100, 0)
            config.set_optim_cache_dir(self.cache_dir)
            config.switch_ir_debug()
            config.enable_tensorrt_engine(max_batch_size=self.trt_param.max_batch_size, workspace_size=self.trt_param.workspace_size, min_subgraph_size=self.trt_param.min_subgraph_size, precision_mode=self.trt_param.precision, use_static=self.trt_param.use_static, use_calib_mode=self.trt_param.use_calib_mode)
            if len(self.dynamic_shape.min_input_shape) != 0 and self.dynamic_shape.min_input_shape.keys() == self.dynamic_shape.max_input_shape.keys() and (self.dynamic_shape.min_input_shape.keys() == self.dynamic_shape.opt_input_shape.keys()):
                config.set_trt_dynamic_shape_info(self.dynamic_shape.min_input_shape, self.dynamic_shape.max_input_shape, self.dynamic_shape.opt_input_shape, self.dynamic_shape.disable_trt_plugin_fp16)
            return config
        else:
            config = paddle_infer.Config()
            config.switch_ir_debug(True)
            config.set_optim_cache_dir(self.cache_dir)
            config.disable_glog_info()
            return config

    def sample_program_configs(self):
        if False:
            i = 10
            return i + 15

        def generate_boxes(batch, num_boxes):
            if False:
                return 10
            return np.arange(batch * num_boxes * 4, dtype=np.float32).reshape([batch, num_boxes, 4])

        def generate_scores(batch, num_boxes, num_classes):
            if False:
                while True:
                    i = 10
            return np.arange(batch * num_classes * num_boxes, dtype=np.float32).reshape([batch, num_classes, num_boxes])
        for batch in [1, 2]:
            self.batch = batch
            for nms_eta in [0.8, 1.1]:
                for (num_boxes, num_classes) in [[80, 100], [40, 200], [20, 400]]:
                    (self.num_boxes, self.num_classes) = (num_boxes, num_classes)
                    for score_threshold in [0.01]:
                        ops_config = [{'op_type': 'multiclass_nms3', 'op_inputs': {'BBoxes': ['input_bboxes'], 'Scores': ['input_scores']}, 'op_outputs': {'Out': ['nms_output_boxes'], 'Index': ['nms_output_index'], 'NmsRoisNum': ['nms_output_num']}, 'op_attrs': {'background_label': -1, 'score_threshold': score_threshold, 'nms_top_k': num_boxes, 'keep_top_k': num_boxes, 'nms_threshold': 0.3, 'normalized': False, 'nms_eta': nms_eta}, 'outputs_dtype': {'nms_output_index': np.int32}}]
                        ops = self.generate_op_config(ops_config)
                        program_config = ProgramConfig(ops=ops, weights={}, inputs={'input_bboxes': TensorConfig(data_gen=partial(generate_boxes, batch, num_boxes)), 'input_scores': TensorConfig(data_gen=partial(generate_scores, batch, num_boxes, num_classes))}, outputs=['nms_output_boxes', 'nms_output_num', 'nms_output_index'])
                        yield program_config

    def sample_predictor_configs(self, program_config) -> (paddle_infer.Config, List[int], float):
        if False:
            i = 10
            return i + 15

        def generate_dynamic_shape(attrs):
            if False:
                print('Hello World!')
            self.dynamic_shape.min_input_shape = {'input_bboxes': [1, self.num_boxes, 4], 'input_scores': [1, self.num_classes, self.num_boxes]}
            self.dynamic_shape.max_input_shape = {'input_bboxes': [8, self.num_boxes, 4], 'input_scores': [8, self.num_classes, self.num_boxes]}
            self.dynamic_shape.opt_input_shape = {'input_bboxes': [self.batch, self.num_boxes, 4], 'input_scores': [self.batch, self.num_classes, self.num_boxes]}

        def clear_dynamic_shape():
            if False:
                return 10
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if False:
                while True:
                    i = 10
            return (1, 2)
        attrs = [program_config.ops[i].attrs for i in range(len(program_config.ops))]
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 1e-05)
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        program_config.set_input_type(np.float16)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, False), 0.01)
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield (self.create_inference_config(), generate_trt_nodes_num(attrs, True), 1e-05)

    def assert_tensors_near(self, atol: float, rtol: float, tensor: Dict[str, np.array], baseline: Dict[str, np.array]):
        if False:
            for i in range(10):
                print('nop')
        for (key, arr) in tensor.items():
            if key == 'nms_output_index':
                continue
            if key == 'nms_output_boxes':
                basline_arr = np.array(sorted(baseline[key].reshape((-1, 6)), key=lambda i: [i[0], i[1]]))
                arr = np.array(sorted(arr.reshape((-1, 6)), key=lambda i: [i[0], i[1]]))
            else:
                basline_arr = np.array(baseline[key].reshape((-1, 1)))
                arr = np.array(arr.reshape((-1, 1)))
            self.assertTrue(basline_arr.shape == arr.shape, 'The output shapes are not equal, the baseline shape is ' + str(basline_arr.shape) + ', but got ' + str(arr.shape))
            diff = abs(basline_arr - arr)
            np.testing.assert_allclose(basline_arr, arr, rtol=rtol, atol=atol, err_msg=f'Output has diff, Maximum absolute error: {np.amax(diff)}')

    def assert_op_size(self, trt_engine_num, paddle_op_num):
        if False:
            print('Hello World!')
        return True

    def test(self):
        if False:
            while True:
                i = 10
        self.trt_param.workspace_size = 1 << 25
        self.run_test()
if __name__ == '__main__':
    unittest.main()