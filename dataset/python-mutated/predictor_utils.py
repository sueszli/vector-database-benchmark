import os
import numpy as np
from paddle import base
from paddle.base.core import AnalysisConfig, create_paddle_predictor

class PredictorTools:
    """
    Paddle-Inference predictor
    """

    def __init__(self, model_path, model_file, params_file, feeds_var):
        if False:
            print('Hello World!')
        '\n        __init__\n        '
        self.model_path = model_path
        self.model_file = model_file
        self.params_file = params_file
        self.feeds_var = feeds_var

    def _load_model_and_set_config(self):
        if False:
            print('Hello World!')
        '\n        load model from file and set analysis config\n        '
        if os.path.exists(os.path.join(self.model_path, self.params_file)):
            config = AnalysisConfig(os.path.join(self.model_path, self.model_file), os.path.join(self.model_path, self.params_file))
        else:
            config = AnalysisConfig(os.path.join(self.model_path))
        if base.is_compiled_with_cuda():
            config.enable_use_gpu(100, 0)
        else:
            config.disable_gpu()
        config.switch_specify_input_names(True)
        config.switch_use_feed_fetch_ops(False)
        config.enable_memory_optim()
        config.disable_glog_info()
        config.switch_ir_optim(False)
        return config

    def _get_analysis_outputs(self, config):
        if False:
            print('Hello World!')
        '\n        Return outputs of paddle inference\n        Args:\n            config (AnalysisConfig): predictor configs\n        Returns:\n            outs (numpy array): forward netwrok prediction outputs\n        '
        predictor = create_paddle_predictor(config)
        tensor_shapes = predictor.get_input_tensor_shape()
        names = predictor.get_input_names()
        for (i, name) in enumerate(names):
            shape = tensor_shapes[name]
            tensor = predictor.get_input_tensor(name)
            feed_data = self.feeds_var[i]
            tensor.copy_from_cpu(np.array(feed_data))
            if type(feed_data) == base.LoDTensor:
                tensor.set_lod(feed_data.lod())
        repeat_time = 2
        for i in range(repeat_time):
            predictor.zero_copy_run()
        output_names = predictor.get_output_names()
        outs = [predictor.get_output_tensor(out_name).copy_to_cpu() for out_name in output_names]
        return outs

    def __call__(self):
        if False:
            print('Hello World!')
        '\n        __call__\n        '
        config = self._load_model_and_set_config()
        outputs = self._get_analysis_outputs(config)
        return outputs