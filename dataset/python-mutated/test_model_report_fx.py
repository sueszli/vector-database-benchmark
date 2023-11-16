import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn.functional as F
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.fx._model_report.detector import DynamicStaticDetector, InputWeightEqualizationDetector, PerChannelDetector, OutlierDetector
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx._model_report.model_report import ModelReport
from torch.ao.quantization.observer import HistogramObserver, default_per_channel_weight_observer, default_observer
from torch.ao.nn.intrinsic.modules.fused import ConvReLU2d, LinearReLU
from torch.testing._internal.common_quantization import ConvModel, QuantizationTestCase, SingleLayerLinearModel, TwoLayerLinearModel, skipIfNoFBGEMM, skipIfNoQNNPACK, override_quantized_engine
'\nPartition of input domain:\n\nModel contains: conv or linear, both conv and linear\n    Model contains: ConvTransposeNd (not supported for per_channel)\n\nModel is: post training quantization model, quantization aware training model\nModel is: composed with nn.Sequential, composed in class structure\n\nQConfig utilizes per_channel weight observer, backend uses non per_channel weight observer\nQConfig_dict uses only one default qconfig, Qconfig dict uses > 1 unique qconfigs\n\nPartition on output domain:\n\nThere are possible changes / suggestions, there are no changes / suggestions\n'
DEFAULT_NO_OPTIMS_ANSWER_STRING = 'Further Optimizations for backend {}: \nNo further per_channel optimizations possible.'
NESTED_CONV_LINEAR_EXAMPLE = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 2, 1), torch.nn.Sequential(torch.nn.Linear(9, 27), torch.nn.ReLU()), torch.nn.Linear(27, 27), torch.nn.ReLU(), torch.nn.Conv2d(3, 3, 2, 1))
LAZY_CONV_LINEAR_EXAMPLE = torch.nn.Sequential(torch.nn.LazyConv2d(3, 3, 2, 1), torch.nn.Sequential(torch.nn.Linear(5, 27), torch.nn.ReLU()), torch.nn.ReLU(), torch.nn.Linear(27, 27), torch.nn.ReLU(), torch.nn.LazyConv2d(3, 3, 2, 1))
FUSION_CONV_LINEAR_EXAMPLE = torch.nn.Sequential(ConvReLU2d(torch.nn.Conv2d(3, 3, 2, 1), torch.nn.ReLU()), torch.nn.Sequential(LinearReLU(torch.nn.Linear(9, 27), torch.nn.ReLU())), LinearReLU(torch.nn.Linear(27, 27), torch.nn.ReLU()), torch.nn.Conv2d(3, 3, 2, 1))

class ThreeOps(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, x):
        if False:
            print('Hello World!')
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_example_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        return (torch.randn(1, 3, 3, 3),)

class TwoThreeOps(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.block1 = ThreeOps()
        self.block2 = ThreeOps()

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.block1(x)
        y = self.block2(x)
        z = x + y
        z = F.relu(z)
        return z

    def get_example_inputs(self):
        if False:
            print('Hello World!')
        return (torch.randn(1, 3, 3, 3),)

class TestFxModelReportDetector(QuantizationTestCase):
    """Prepares and calibrate the model"""

    def _prepare_model_and_run_input(self, model, q_config_mapping, input):
        if False:
            i = 10
            return i + 15
        model_prep = torch.ao.quantization.quantize_fx.prepare_fx(model, q_config_mapping, input)
        model_prep(input).sum()
        return model_prep
    'Case includes:\n        one conv or linear\n        post training quantization\n        composed as module\n        qconfig uses per_channel weight observer\n        Only 1 qconfig in qconfig dict\n        Output has no changes / suggestions\n    '

    @skipIfNoFBGEMM
    def test_simple_conv(self):
        if False:
            for i in range(10):
                print('nop')
        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = 'fbgemm'
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            input = torch.randn(1, 3, 10, 10)
            prepared_model = self._prepare_model_and_run_input(ConvModel(), q_config_mapping, input)
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            (optims_str, per_channel_info) = per_channel_detector.generate_detector_report(prepared_model)
            self.assertEqual(optims_str, DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine))
            self.assertEqual(per_channel_info['conv']['backend'], torch.backends.quantized.engine)
            self.assertEqual(len(per_channel_info), 1)
            self.assertEqual(list(per_channel_info)[0], 'conv')
            self.assertEqual(per_channel_info['conv']['per_channel_quantization_supported'], True)
            self.assertEqual(per_channel_info['conv']['per_channel_quantization_used'], True)
    "Case includes:\n        Multiple conv or linear\n        post training quantization\n        composed as module\n        qconfig doesn't use per_channel weight observer\n        Only 1 qconfig in qconfig dict\n        Output has possible changes / suggestions\n    "

    @skipIfNoQNNPACK
    def test_multi_linear_model_without_per_channel(self):
        if False:
            print('Hello World!')
        with override_quantized_engine('qnnpack'):
            torch.backends.quantized.engine = 'qnnpack'
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            prepared_model = self._prepare_model_and_run_input(TwoLayerLinearModel(), q_config_mapping, TwoLayerLinearModel().get_example_inputs()[0])
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            (optims_str, per_channel_info) = per_channel_detector.generate_detector_report(prepared_model)
            self.assertNotEqual(optims_str, DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine))
            rand_key: str = list(per_channel_info.keys())[0]
            self.assertEqual(per_channel_info[rand_key]['backend'], torch.backends.quantized.engine)
            self.assertEqual(len(per_channel_info), 2)
            for linear_key in per_channel_info.keys():
                module_entry = per_channel_info[linear_key]
                self.assertEqual(module_entry['per_channel_quantization_supported'], True)
                self.assertEqual(module_entry['per_channel_quantization_used'], False)
    "Case includes:\n        Multiple conv or linear\n        post training quantization\n        composed as Module\n        qconfig doesn't use per_channel weight observer\n        More than 1 qconfig in qconfig dict\n        Output has possible changes / suggestions\n    "

    @skipIfNoQNNPACK
    def test_multiple_q_config_options(self):
        if False:
            i = 10
            return i + 15
        with override_quantized_engine('qnnpack'):
            torch.backends.quantized.engine = 'qnnpack'
            per_channel_qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True), weight=default_per_channel_weight_observer)

            class ConvLinearModel(torch.nn.Module):

                def __init__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    super().__init__()
                    self.conv1 = torch.nn.Conv2d(3, 3, 2, 1)
                    self.fc1 = torch.nn.Linear(9, 27)
                    self.relu = torch.nn.ReLU()
                    self.fc2 = torch.nn.Linear(27, 27)
                    self.conv2 = torch.nn.Conv2d(3, 3, 2, 1)

                def forward(self, x):
                    if False:
                        while True:
                            i = 10
                    x = self.conv1(x)
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    x = self.conv2(x)
                    return x
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine)).set_object_type(torch.nn.Conv2d, per_channel_qconfig)
            prepared_model = self._prepare_model_and_run_input(ConvLinearModel(), q_config_mapping, torch.randn(1, 3, 10, 10))
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            (optims_str, per_channel_info) = per_channel_detector.generate_detector_report(prepared_model)
            self.assertNotEqual(optims_str, DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine))
            self.assertEqual(len(per_channel_info), 4)
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry['per_channel_quantization_supported'], True)
                if 'fc' in key:
                    self.assertEqual(module_entry['per_channel_quantization_used'], False)
                elif 'conv' in key:
                    self.assertEqual(module_entry['per_channel_quantization_used'], True)
                else:
                    raise ValueError('Should only contain conv and linear layers as key values')
    "Case includes:\n        Multiple conv or linear\n        post training quantization\n        composed as sequential\n        qconfig doesn't use per_channel weight observer\n        Only 1 qconfig in qconfig dict\n        Output has possible changes / suggestions\n    "

    @skipIfNoQNNPACK
    def test_sequential_model_format(self):
        if False:
            i = 10
            return i + 15
        with override_quantized_engine('qnnpack'):
            torch.backends.quantized.engine = 'qnnpack'
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            prepared_model = self._prepare_model_and_run_input(NESTED_CONV_LINEAR_EXAMPLE, q_config_mapping, torch.randn(1, 3, 10, 10))
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            (optims_str, per_channel_info) = per_channel_detector.generate_detector_report(prepared_model)
            self.assertNotEqual(optims_str, DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine))
            self.assertEqual(len(per_channel_info), 4)
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry['per_channel_quantization_supported'], True)
                self.assertEqual(module_entry['per_channel_quantization_used'], False)
    "Case includes:\n        Multiple conv or linear\n        post training quantization\n        composed as sequential\n        qconfig doesn't use per_channel weight observer\n        Only 1 qconfig in qconfig dict\n        Output has possible changes / suggestions\n    "

    @skipIfNoQNNPACK
    def test_conv_sub_class_considered(self):
        if False:
            return 10
        with override_quantized_engine('qnnpack'):
            torch.backends.quantized.engine = 'qnnpack'
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            prepared_model = self._prepare_model_and_run_input(LAZY_CONV_LINEAR_EXAMPLE, q_config_mapping, torch.randn(1, 3, 10, 10))
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            (optims_str, per_channel_info) = per_channel_detector.generate_detector_report(prepared_model)
            self.assertNotEqual(optims_str, DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine))
            self.assertEqual(len(per_channel_info), 4)
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry['per_channel_quantization_supported'], True)
                self.assertEqual(module_entry['per_channel_quantization_used'], False)
    'Case includes:\n        Multiple conv or linear\n        post training quantization\n        composed as sequential\n        qconfig uses per_channel weight observer\n        Only 1 qconfig in qconfig dict\n        Output has no possible changes / suggestions\n    '

    @skipIfNoFBGEMM
    def test_fusion_layer_in_sequential(self):
        if False:
            for i in range(10):
                print('nop')
        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = 'fbgemm'
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            prepared_model = self._prepare_model_and_run_input(FUSION_CONV_LINEAR_EXAMPLE, q_config_mapping, torch.randn(1, 3, 10, 10))
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            (optims_str, per_channel_info) = per_channel_detector.generate_detector_report(prepared_model)
            self.assertEqual(optims_str, DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine))
            self.assertEqual(len(per_channel_info), 4)
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry['per_channel_quantization_supported'], True)
                self.assertEqual(module_entry['per_channel_quantization_used'], True)
    'Case includes:\n        Multiple conv or linear\n        quantitative aware training\n        composed as model\n        qconfig does not use per_channel weight observer\n        Only 1 qconfig in qconfig dict\n        Output has possible changes / suggestions\n    '

    @skipIfNoQNNPACK
    def test_qat_aware_model_example(self):
        if False:
            print('Hello World!')

        class QATConvLinearReluModel(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = torch.nn.BatchNorm2d(1)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    return 10
                x = self.quant(x)
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x
        with override_quantized_engine('qnnpack'):
            model_fp32 = QATConvLinearReluModel()
            model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
            model_fp32.eval()
            model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'bn', 'relu']])
            model_fp32_fused.train()
            model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused)
            per_channel_detector = PerChannelDetector(torch.backends.quantized.engine)
            (optims_str, per_channel_info) = per_channel_detector.generate_detector_report(model_fp32_prepared)
            self.assertNotEqual(optims_str, DEFAULT_NO_OPTIMS_ANSWER_STRING.format(torch.backends.quantized.engine))
            self.assertEqual(len(per_channel_info), 1)
            for key in per_channel_info.keys():
                module_entry = per_channel_info[key]
                self.assertEqual(module_entry['per_channel_quantization_supported'], True)
                self.assertEqual(module_entry['per_channel_quantization_used'], False)
'\nPartition on Domain / Things to Test\n\n- All zero tensor\n- Multiple tensor dimensions\n- All of the outward facing functions\n- Epoch min max are correctly updating\n- Batch range is correctly averaging as expected\n- Reset for each epoch is correctly resetting the values\n\nPartition on Output\n- the calcuation of the ratio is occurring correctly\n\n'

class TestFxModelReportObserver(QuantizationTestCase):

    class NestedModifiedSingleLayerLinear(torch.nn.Module):

        def __init__(self):
            if False:
                return 10
            super().__init__()
            self.obs1 = ModelReportObserver()
            self.mod1 = SingleLayerLinearModel()
            self.obs2 = ModelReportObserver()
            self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            if False:
                print('Hello World!')
            x = self.obs1(x)
            x = self.mod1(x)
            x = self.obs2(x)
            x = self.fc1(x)
            x = self.relu(x)
            return x

    def run_model_and_common_checks(self, model, ex_input, num_epochs, batch_size):
        if False:
            for i in range(10):
                print('nop')
        split_up_data = torch.split(ex_input, batch_size)
        for epoch in range(num_epochs):
            model.apply(lambda module: module.reset_batch_and_epoch_values() if isinstance(module, ModelReportObserver) else None)
            self.assertEqual(model.obs1.average_batch_activation_range, torch.tensor(float(0)))
            self.assertEqual(model.obs1.epoch_activation_min, torch.tensor(float('inf')))
            self.assertEqual(model.obs1.epoch_activation_max, torch.tensor(float('-inf')))
            for (index, batch) in enumerate(split_up_data):
                num_tracked_so_far = model.obs1.num_batches_tracked
                self.assertEqual(num_tracked_so_far, index)
                (batch_min, batch_max) = torch.aminmax(batch)
                current_average_range = model.obs1.average_batch_activation_range
                current_epoch_min = model.obs1.epoch_activation_min
                current_epoch_max = model.obs1.epoch_activation_max
                model(ex_input)
                correct_updated_value = (current_average_range * num_tracked_so_far + (batch_max - batch_min)) / (num_tracked_so_far + 1)
                self.assertEqual(model.obs1.average_batch_activation_range, correct_updated_value)
                if current_epoch_max - current_epoch_min > 0:
                    self.assertEqual(model.obs1.get_batch_to_epoch_ratio(), correct_updated_value / (current_epoch_max - current_epoch_min))
    'Case includes:\n        all zero tensor\n        dim size = 2\n        run for 1 epoch\n        run for 10 batch\n        tests input data observer\n    '

    def test_zero_tensor_errors(self):
        if False:
            print('Hello World!')
        model = self.NestedModifiedSingleLayerLinear()
        ex_input = torch.zeros((10, 1, 5))
        self.run_model_and_common_checks(model, ex_input, 1, 1)
        self.assertEqual(model.obs1.epoch_activation_min, 0)
        self.assertEqual(model.obs1.epoch_activation_max, 0)
        self.assertEqual(model.obs1.average_batch_activation_range, 0)
        with self.assertRaises(ValueError):
            ratio_val = model.obs1.get_batch_to_epoch_ratio()
    'Case includes:\n    non-zero tensor\n    dim size = 2\n    run for 1 epoch\n    run for 1 batch\n    tests input data observer\n    '

    def test_single_batch_of_ones(self):
        if False:
            while True:
                i = 10
        model = self.NestedModifiedSingleLayerLinear()
        ex_input = torch.ones((1, 1, 5))
        self.run_model_and_common_checks(model, ex_input, 1, 1)
        self.assertEqual(model.obs1.epoch_activation_min, 1)
        self.assertEqual(model.obs1.epoch_activation_max, 1)
        self.assertEqual(model.obs1.average_batch_activation_range, 0)
        with self.assertRaises(ValueError):
            ratio_val = model.obs1.get_batch_to_epoch_ratio()
    'Case includes:\n    non-zero tensor\n    dim size = 2\n    run for 10 epoch\n    run for 15 batch\n    tests non input data observer\n    '

    def test_observer_after_relu(self):
        if False:
            i = 10
            return i + 15

        class NestedModifiedObserverAfterRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.mod1 = SingleLayerLinearModel()
                self.obs2 = ModelReportObserver()
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.obs1(x)
                x = self.mod1(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.obs2(x)
                return x
        model = NestedModifiedObserverAfterRelu()
        ex_input = torch.randn((15, 1, 5))
        self.run_model_and_common_checks(model, ex_input, 10, 15)
    'Case includes:\n        non-zero tensor\n        dim size = 2\n        run for multiple epoch\n        run for multiple batch\n        tests input data observer\n    '

    def test_random_epochs_and_batches(self):
        if False:
            i = 10
            return i + 15

        class TinyNestModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
                self.relu = torch.nn.ReLU()
                self.obs2 = ModelReportObserver()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.obs1(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.obs2(x)
                return x

        class LargerIncludeNestModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.nested = TinyNestModule()
                self.fc1 = SingleLayerLinearModel()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.obs1(x)
                x = self.nested(x)
                x = self.fc1(x)
                x = self.relu(x)
                return x

        class ModifiedThreeOps(torch.nn.Module):

            def __init__(self, batch_norm_dim):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.linear = torch.nn.Linear(7, 3, 2)
                self.obs2 = ModelReportObserver()
                if batch_norm_dim == 2:
                    self.bn = torch.nn.BatchNorm2d(2)
                elif batch_norm_dim == 3:
                    self.bn = torch.nn.BatchNorm3d(4)
                else:
                    raise ValueError('Dim should only be 2 or 3')
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.obs1(x)
                x = self.linear(x)
                x = self.obs2(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        class HighDimensionNet(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.obs1 = ModelReportObserver()
                self.fc1 = torch.nn.Linear(3, 7)
                self.block1 = ModifiedThreeOps(3)
                self.fc2 = torch.nn.Linear(3, 7)
                self.block2 = ModifiedThreeOps(3)
                self.fc3 = torch.nn.Linear(3, 7)

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.obs1(x)
                x = self.fc1(x)
                x = self.block1(x)
                x = self.fc2(x)
                y = self.block2(x)
                y = self.fc3(y)
                z = x + y
                z = F.relu(z)
                return z
        models = [self.NestedModifiedSingleLayerLinear(), LargerIncludeNestModel(), ModifiedThreeOps(2), HighDimensionNet()]
        num_epochs = 10
        num_batches = 15
        input_shapes = [(1, 5), (1, 5), (2, 3, 7), (4, 1, 8, 3)]
        inputs = []
        for shape in input_shapes:
            ex_input = torch.randn((num_batches, *shape))
            inputs.append(ex_input)
        for (index, model) in enumerate(models):
            self.run_model_and_common_checks(model, inputs[index], num_epochs, num_batches)
'\nPartition on domain / things to test\n\nThere is only a single test case for now.\n\nThis will be more thoroughly tested with the implementation of the full end to end tool coming soon.\n'

class TestFxModelReportDetectDynamicStatic(QuantizationTestCase):

    @skipIfNoFBGEMM
    def test_nested_detection_case(self):
        if False:
            for i in range(10):
                print('nop')

        class SingleLinear(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                if False:
                    return 10
                x = self.linear(x)
                return x

        class TwoBlockNet(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.block1 = SingleLinear()
                self.block2 = SingleLinear()

            def forward(self, x):
                if False:
                    return 10
                x = self.block1(x)
                y = self.block2(x)
                z = x + y
                z = F.relu(z)
                return z
        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = 'fbgemm'
            model = TwoBlockNet()
            example_input = torch.randint(-10, 0, (1, 3, 3, 3))
            example_input = example_input.to(torch.float)
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig('fbgemm'))
            model_prep = quantize_fx.prepare_fx(model, q_config_mapping, example_input)
            obs_ctr = ModelReportObserver
            linear_fqn = 'block2.linear'
            target_linear = None
            for node in model_prep.graph.nodes:
                if node.target == linear_fqn:
                    target_linear = node
                    break
            with model_prep.graph.inserting_before(target_linear):
                obs_to_insert = obs_ctr()
                pre_obs_fqn = linear_fqn + '.model_report_pre_observer'
                model_prep.add_submodule(pre_obs_fqn, obs_to_insert)
                model_prep.graph.create_node(op='call_module', target=pre_obs_fqn, args=target_linear.args)
            with model_prep.graph.inserting_after(target_linear):
                obs_to_insert = obs_ctr()
                post_obs_fqn = linear_fqn + '.model_report_post_observer'
                model_prep.add_submodule(post_obs_fqn, obs_to_insert)
                model_prep.graph.create_node(op='call_module', target=post_obs_fqn, args=(target_linear,))
            model_prep.recompile()
            num_iterations = 10
            for i in range(num_iterations):
                if i % 2 == 0:
                    example_input = torch.randint(-10, 0, (1, 3, 3, 3)).to(torch.float)
                else:
                    example_input = torch.randint(0, 10, (1, 3, 3, 3)).to(torch.float)
                model_prep(example_input)
            dynamic_vs_static_detector = DynamicStaticDetector()
            (dynam_vs_stat_str, dynam_vs_stat_dict) = dynamic_vs_static_detector.generate_detector_report(model_prep)
            data_dist_info = [dynam_vs_stat_dict[linear_fqn][DynamicStaticDetector.PRE_OBS_DATA_DIST_KEY], dynam_vs_stat_dict[linear_fqn][DynamicStaticDetector.POST_OBS_DATA_DIST_KEY]]
            self.assertTrue('stationary' in data_dist_info)
            self.assertTrue('non-stationary' in data_dist_info)
            self.assertTrue(dynam_vs_stat_dict[linear_fqn]['dynamic_recommended'])

class TestFxModelReportClass(QuantizationTestCase):

    @skipIfNoFBGEMM
    def test_constructor(self):
        if False:
            return 10
        '\n        Tests the constructor of the ModelReport class.\n        Specifically looks at:\n        - The desired reports\n        - Ensures that the observers of interest are properly initialized\n        '
        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = 'fbgemm'
            backend = torch.backends.quantized.engine
            model = ThreeOps()
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            model_prep = quantize_fx.prepare_fx(model, q_config_mapping, model.get_example_inputs()[0])
            test_detector_set = {DynamicStaticDetector(), PerChannelDetector(backend)}
            model_report = ModelReport(model_prep, test_detector_set)
            detector_name_set = {detector.get_detector_name() for detector in test_detector_set}
            self.assertEqual(model_report.get_desired_reports_names(), detector_name_set)
            with self.assertRaises(ValueError):
                model_report = ModelReport(model, set())
            num_expected_entries = len(test_detector_set)
            self.assertEqual(len(model_report.get_observers_of_interest()), num_expected_entries)
            for value in model_report.get_observers_of_interest().values():
                self.assertEqual(len(value), 0)

    @skipIfNoFBGEMM
    def test_prepare_model_callibration(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests model_report.prepare_detailed_calibration that prepares the model for callibration\n        Specifically looks at:\n        - Whether observers are properly inserted into regular nn.Module\n        - Whether the target and the arguments of the observers are proper\n        - Whether the internal representation of observers of interest is updated\n        '
        with override_quantized_engine('fbgemm'):
            model = TwoThreeOps()
            torch.backends.quantized.engine = 'fbgemm'
            backend = torch.backends.quantized.engine
            test_detector_set = {DynamicStaticDetector(), PerChannelDetector(backend)}
            example_input = model.get_example_inputs()[0]
            current_backend = torch.backends.quantized.engine
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            model_prep = quantize_fx.prepare_fx(model, q_config_mapping, example_input)
            model_report = ModelReport(model_prep, test_detector_set)
            prepared_for_callibrate_model = model_report.prepare_detailed_calibration()
            modules_observer_cnt = 0
            for (fqn, module) in prepared_for_callibrate_model.named_modules():
                if isinstance(module, ModelReportObserver):
                    modules_observer_cnt += 1
            self.assertEqual(modules_observer_cnt, 4)
            model_report_str_check = 'model_report'
            for node in prepared_for_callibrate_model.graph.nodes:
                if isinstance(node.target, str) and model_report_str_check in node.target:
                    if 'pre_observer' in node.target:
                        self.assertEqual(node.args, node.next.args)
                    if 'post_observer' in node.target:
                        self.assertEqual(node.args, (node.prev,))
            self.assertEqual(len(model_report.get_observers_of_interest()), 2)
            for detector in test_detector_set:
                self.assertTrue(detector.get_detector_name() in model_report.get_observers_of_interest().keys())
                detector_obs_of_interest_fqns = model_report.get_observers_of_interest()[detector.get_detector_name()]
                if isinstance(detector, PerChannelDetector):
                    self.assertEqual(len(detector_obs_of_interest_fqns), 0)
                elif isinstance(detector, DynamicStaticDetector):
                    self.assertEqual(len(detector_obs_of_interest_fqns), 4)
            with self.assertRaises(ValueError):
                prepared_for_callibrate_model = model_report.prepare_detailed_calibration()

    def get_module_and_graph_cnts(self, callibrated_fx_module):
        if False:
            while True:
                i = 10
        '\n        Calculates number of ModelReportObserver modules in the model as well as the graph structure.\n        Returns a tuple of two elements:\n        int: The number of ModelReportObservers found in the model\n        int: The number of model_report nodes found in the graph\n        '
        modules_observer_cnt = 0
        for (fqn, module) in callibrated_fx_module.named_modules():
            if isinstance(module, ModelReportObserver):
                modules_observer_cnt += 1
        model_report_str_check = 'model_report'
        graph_observer_cnt = 0
        for node in callibrated_fx_module.graph.nodes:
            if isinstance(node.target, str) and model_report_str_check in node.target:
                graph_observer_cnt += 1
        return (modules_observer_cnt, graph_observer_cnt)

    @skipIfNoFBGEMM
    def test_generate_report(self):
        if False:
            while True:
                i = 10
        '\n            Tests model_report.generate_model_report to ensure report generation\n            Specifically looks at:\n            - Whether correct number of reports are being generated\n            - Whether observers are being properly removed if specified\n            - Whether correct blocking from generating report twice if obs removed\n        '
        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = 'fbgemm'
            filled_detector_set = {DynamicStaticDetector(), PerChannelDetector(torch.backends.quantized.engine)}
            single_detector_set = {DynamicStaticDetector()}
            model_full = TwoThreeOps()
            model_single = TwoThreeOps()
            example_input = model_full.get_example_inputs()[0]
            current_backend = torch.backends.quantized.engine
            q_config_mapping = QConfigMapping()
            q_config_mapping.set_global(torch.ao.quantization.get_default_qconfig(torch.backends.quantized.engine))
            model_prep_full = quantize_fx.prepare_fx(model_full, q_config_mapping, example_input)
            model_prep_single = quantize_fx.prepare_fx(model_single, q_config_mapping, example_input)
            model_report_full = ModelReport(model_prep_full, filled_detector_set)
            model_report_single = ModelReport(model_prep_single, single_detector_set)
            prepared_for_callibrate_model_full = model_report_full.prepare_detailed_calibration()
            prepared_for_callibrate_model_single = model_report_single.prepare_detailed_calibration()
            num_iterations = 10
            for i in range(num_iterations):
                example_input = torch.tensor(torch.randint(100, (1, 3, 3, 3)), dtype=torch.float)
                prepared_for_callibrate_model_full(example_input)
                prepared_for_callibrate_model_single(example_input)
            model_full_report = model_report_full.generate_model_report(True)
            model_single_report = model_report_single.generate_model_report(False)
            self.assertEqual(len(model_full_report), len(filled_detector_set))
            self.assertEqual(len(model_single_report), len(single_detector_set))
            (modules_observer_cnt, graph_observer_cnt) = self.get_module_and_graph_cnts(prepared_for_callibrate_model_full)
            self.assertEqual(modules_observer_cnt, 0)
            self.assertEqual(graph_observer_cnt, 0)
            (modules_observer_cnt, graph_observer_cnt) = self.get_module_and_graph_cnts(prepared_for_callibrate_model_single)
            self.assertNotEqual(modules_observer_cnt, 0)
            self.assertNotEqual(graph_observer_cnt, 0)
            with self.assertRaises(Exception):
                model_full_report = model_report_full.generate_model_report(prepared_for_callibrate_model_full, False)
            model_single_report = model_report_single.generate_model_report(False)

    @skipIfNoFBGEMM
    def test_generate_visualizer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that the ModelReport class can properly create the ModelReportVisualizer instance\n        Checks that:\n            - Correct number of modules are represented\n            - Modules are sorted\n            - Correct number of features for each module\n        '
        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = 'fbgemm'
            detector_set = set()
            detector_set.add(OutlierDetector(reference_percentile=0.95))
            detector_set.add(InputWeightEqualizationDetector(0.5))
            model = TwoThreeOps()
            (prepared_for_callibrate_model, mod_report) = _get_prepped_for_calibration_model_helper(model, detector_set, model.get_example_inputs()[0])
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)
            prepared_for_callibrate_model(example_input)
            with self.assertRaises(Exception):
                mod_rep_visualizaiton = mod_report.generate_visualizer()
            generated_report = mod_report.generate_model_report(remove_inserted_observers=False)
            mod_rep_visualizer: ModelReportVisualizer = mod_report.generate_visualizer()
            mod_fqns_to_features = mod_rep_visualizer.generated_reports
            self.assertEqual(len(mod_fqns_to_features), 6)
            for module_fqn in mod_fqns_to_features:
                if '.linear' in module_fqn:
                    linear_info = mod_fqns_to_features[module_fqn]
                    self.assertEqual(len(linear_info), 20)

    @skipIfNoFBGEMM
    def test_qconfig_mapping_generation(self):
        if False:
            print('Hello World!')
        '\n        Tests for generation of qconfigs by ModelReport API\n        - Tests that qconfigmapping is generated\n        - Tests that mappings include information for for relavent modules\n        '
        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = 'fbgemm'
            detector_set = set()
            detector_set.add(PerChannelDetector())
            detector_set.add(DynamicStaticDetector())
            model = TwoThreeOps()
            (prepared_for_callibrate_model, mod_report) = _get_prepped_for_calibration_model_helper(model, detector_set, model.get_example_inputs()[0])
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)
            prepared_for_callibrate_model(example_input)
            qconfig_mapping = mod_report.generate_qconfig_mapping()
            generated_report = mod_report.generate_model_report(remove_inserted_observers=False)
            mod_reports_by_fqn = mod_report.generate_visualizer().generated_reports
            self.assertEqual(len(qconfig_mapping.module_name_qconfigs), len(mod_reports_by_fqn))
            self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 2)
            for key in qconfig_mapping.module_name_qconfigs:
                config = qconfig_mapping.module_name_qconfigs[key]
                self.assertEqual(config.weight, default_per_channel_weight_observer)
                self.assertEqual(config.activation, default_observer)
            prepared = quantize_fx.prepare_fx(TwoThreeOps(), qconfig_mapping, example_input)
            converted = quantize_fx.convert_fx(prepared)

    @skipIfNoFBGEMM
    def test_equalization_mapping_generation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests for generation of qconfigs by ModelReport API\n        - Tests that equalization config generated when input-weight equalization detector used\n        - Tests that mappings include information for for relavent modules\n        '
        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = 'fbgemm'
            detector_set = set()
            detector_set.add(InputWeightEqualizationDetector(0.6))
            model = TwoThreeOps()
            (prepared_for_callibrate_model, mod_report) = _get_prepped_for_calibration_model_helper(model, detector_set, model.get_example_inputs()[0])
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)
            prepared_for_callibrate_model(example_input)
            qconfig_mapping = mod_report.generate_qconfig_mapping()
            equalization_mapping = mod_report.generate_equalization_mapping()
            self.assertEqual(len(qconfig_mapping.module_name_qconfigs), 2)
            prepared = quantize_fx.prepare_fx(TwoThreeOps(), qconfig_mapping, example_input, _equalization_config=equalization_mapping)
            converted = quantize_fx.convert_fx(prepared)

class TestFxDetectInputWeightEqualization(QuantizationTestCase):

    class SimpleConv(torch.nn.Module):

        def __init__(self, con_dims):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.relu = torch.nn.ReLU()
            self.conv = torch.nn.Conv2d(con_dims[0], con_dims[1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        def forward(self, x):
            if False:
                while True:
                    i = 10
            x = self.conv(x)
            x = self.relu(x)
            return x

    class TwoBlockComplexNet(torch.nn.Module):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.block1 = TestFxDetectInputWeightEqualization.SimpleConv((3, 32))
            self.block2 = TestFxDetectInputWeightEqualization.SimpleConv((3, 3))
            self.conv = torch.nn.Conv2d(32, 3, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
            self.linear = torch.nn.Linear(768, 10)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            if False:
                while True:
                    i = 10
            x = self.block1(x)
            x = self.conv(x)
            y = self.block2(x)
            y = y.repeat(1, 1, 2, 2)
            z = x + y
            z = z.flatten(start_dim=1)
            z = self.linear(z)
            z = self.relu(z)
            return z

        def get_fusion_modules(self):
            if False:
                print('Hello World!')
            return [['conv', 'relu']]

        def get_example_inputs(self):
            if False:
                while True:
                    i = 10
            return (torch.randn((1, 3, 28, 28)),)

    class ReluOnly(torch.nn.Module):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            if False:
                print('Hello World!')
            x = self.relu(x)
            return x

        def get_example_inputs(self):
            if False:
                while True:
                    i = 10
            return (torch.arange(27).reshape((1, 3, 3, 3)),)

    def _get_prepped_for_calibration_model(self, model, detector_set, fused=False):
        if False:
            for i in range(10):
                print('nop')
        'Returns a model that has been prepared for callibration and corresponding model_report'
        example_input = model.get_example_inputs()[0]
        return _get_prepped_for_calibration_model_helper(model, detector_set, example_input, fused)

    @skipIfNoFBGEMM
    def test_input_weight_equalization_determine_points(self):
        if False:
            print('Hello World!')
        with override_quantized_engine('fbgemm'):
            detector_set = {InputWeightEqualizationDetector(0.5)}
            non_fused = self._get_prepped_for_calibration_model(self.TwoBlockComplexNet(), detector_set)
            fused = self._get_prepped_for_calibration_model(self.TwoBlockComplexNet(), detector_set, fused=True)
            for (prepared_for_callibrate_model, mod_report) in [non_fused, fused]:
                mods_to_check = {nn.Linear, nn.Conv2d}
                node_fqns = {node.target for node in prepared_for_callibrate_model.graph.nodes}
                correct_number_of_obs_inserted = 4
                number_of_obs_found = 0
                obs_name_to_find = InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME
                for node in prepared_for_callibrate_model.graph.nodes:
                    if obs_name_to_find in str(node.target):
                        number_of_obs_found += 1
                self.assertEqual(number_of_obs_found, correct_number_of_obs_inserted)
                for (fqn, module) in prepared_for_callibrate_model.named_modules():
                    is_in_include_list = sum([isinstance(module, x) for x in mods_to_check]) > 0
                    if is_in_include_list:
                        self.assertTrue(hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))
                    else:
                        self.assertTrue(not hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))

    @skipIfNoFBGEMM
    def test_input_weight_equalization_report_gen(self):
        if False:
            while True:
                i = 10
        with override_quantized_engine('fbgemm'):
            test_input_weight_detector = InputWeightEqualizationDetector(0.4)
            detector_set = {test_input_weight_detector}
            model = self.TwoBlockComplexNet()
            (prepared_for_callibrate_model, model_report) = self._get_prepped_for_calibration_model(model, detector_set)
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)
            prepared_for_callibrate_model(example_input)
            generated_report = model_report.generate_model_report(True)
            self.assertEqual(len(generated_report), 1)
            (input_weight_str, input_weight_dict) = generated_report[test_input_weight_detector.get_detector_name()]
            self.assertEqual(len(input_weight_dict), 4)
            example_input = example_input.reshape((3, 28, 28))
            for module_fqn in input_weight_dict:
                if 'block1.linear' in module_fqn:
                    block_1_lin_recs = input_weight_dict[module_fqn]
                    ch_axis = block_1_lin_recs[InputWeightEqualizationDetector.CHANNEL_KEY]
                    (example_min, example_max) = torch.aminmax(example_input, dim=ch_axis)
                    dimension_min = torch.amin(example_min, dim=ch_axis)
                    dimension_max = torch.amax(example_max, dim=ch_axis)
                    min_per_key = InputWeightEqualizationDetector.ACTIVATION_PREFIX
                    min_per_key += InputWeightEqualizationDetector.PER_CHANNEL_MIN_KEY
                    max_per_key = InputWeightEqualizationDetector.ACTIVATION_PREFIX
                    max_per_key += InputWeightEqualizationDetector.PER_CHANNEL_MAX_KEY
                    per_channel_min = block_1_lin_recs[min_per_key]
                    per_channel_max = block_1_lin_recs[max_per_key]
                    self.assertEqual(per_channel_min, dimension_min)
                    self.assertEqual(per_channel_max, dimension_max)
                    min_key = InputWeightEqualizationDetector.ACTIVATION_PREFIX
                    min_key += InputWeightEqualizationDetector.GLOBAL_MIN_KEY
                    max_key = InputWeightEqualizationDetector.ACTIVATION_PREFIX
                    max_key += InputWeightEqualizationDetector.GLOBAL_MAX_KEY
                    global_min = block_1_lin_recs[min_key]
                    global_max = block_1_lin_recs[max_key]
                    self.assertEqual(global_min, min(dimension_min))
                    self.assertEqual(global_max, max(dimension_max))
                    input_ratio = torch.sqrt((per_channel_max - per_channel_min) / (global_max - global_min))
                    min_per_key = InputWeightEqualizationDetector.WEIGHT_PREFIX
                    min_per_key += InputWeightEqualizationDetector.PER_CHANNEL_MIN_KEY
                    max_per_key = InputWeightEqualizationDetector.WEIGHT_PREFIX
                    max_per_key += InputWeightEqualizationDetector.PER_CHANNEL_MAX_KEY
                    per_channel_min = block_1_lin_recs[min_per_key]
                    per_channel_max = block_1_lin_recs[max_per_key]
                    min_key = InputWeightEqualizationDetector.WEIGHT_PREFIX
                    min_key += InputWeightEqualizationDetector.GLOBAL_MIN_KEY
                    max_key = InputWeightEqualizationDetector.WEIGHT_PREFIX
                    max_key += InputWeightEqualizationDetector.GLOBAL_MAX_KEY
                    global_min = block_1_lin_recs[min_key]
                    global_max = block_1_lin_recs[max_key]
                    weight_ratio = torch.sqrt((per_channel_max - per_channel_min) / (global_max - global_min))
                    comp_stat = block_1_lin_recs[InputWeightEqualizationDetector.COMP_METRIC_KEY]
                    weight_to_input_ratio = weight_ratio / input_ratio
                    self.assertEqual(comp_stat, weight_to_input_ratio)
                    break

    @skipIfNoFBGEMM
    def test_input_weight_equalization_report_gen_empty(self):
        if False:
            return 10
        with override_quantized_engine('fbgemm'):
            test_input_weight_detector = InputWeightEqualizationDetector(0.4)
            detector_set = {test_input_weight_detector}
            model = self.ReluOnly()
            (prepared_for_callibrate_model, model_report) = self._get_prepped_for_calibration_model(model, detector_set)
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)
            prepared_for_callibrate_model(example_input)
            generated_report = model_report.generate_model_report(True)
            self.assertEqual(len(generated_report), 1)
            (input_weight_str, input_weight_dict) = generated_report[test_input_weight_detector.get_detector_name()]
            self.assertEqual(len(input_weight_dict), 0)
            self.assertEqual(input_weight_str.count('\n'), 2)

class TestFxDetectOutliers(QuantizationTestCase):

    class LargeBatchModel(torch.nn.Module):

        def __init__(self, param_size):
            if False:
                i = 10
                return i + 15
            super().__init__()
            self.param_size = param_size
            self.linear = torch.nn.Linear(param_size, param_size)
            self.relu_1 = torch.nn.ReLU()
            self.conv = torch.nn.Conv2d(param_size, param_size, 1)
            self.relu_2 = torch.nn.ReLU()

        def forward(self, x):
            if False:
                print('Hello World!')
            x = self.linear(x)
            x = self.relu_1(x)
            x = self.conv(x)
            x = self.relu_2(x)
            return x

        def get_example_inputs(self):
            if False:
                return 10
            param_size = self.param_size
            return (torch.randn((1, param_size, param_size, param_size)),)

        def get_outlier_inputs(self):
            if False:
                for i in range(10):
                    print('nop')
            param_size = self.param_size
            random_vals = torch.randn((1, param_size, param_size, param_size))
            random_vals[:, 0:param_size:2, 0, 3] = torch.tensor([328000000.0])
            return (random_vals,)

    def _get_prepped_for_calibration_model(self, model, detector_set, use_outlier_data=False):
        if False:
            return 10
        'Returns a model that has been prepared for callibration and corresponding model_report'
        example_input = model.get_example_inputs()[0]
        if use_outlier_data:
            example_input = model.get_outlier_inputs()[0]
        return _get_prepped_for_calibration_model_helper(model, detector_set, example_input)

    @skipIfNoFBGEMM
    def test_outlier_detection_determine_points(self):
        if False:
            while True:
                i = 10
        with override_quantized_engine('fbgemm'):
            detector_set = {OutlierDetector(reference_percentile=0.95)}
            (prepared_for_callibrate_model, mod_report) = self._get_prepped_for_calibration_model(self.LargeBatchModel(param_size=128), detector_set)
            mods_to_check = {nn.Linear, nn.Conv2d, nn.ReLU}
            correct_number_of_obs_inserted = 4
            number_of_obs_found = 0
            obs_name_to_find = InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME
            number_of_obs_found = sum([1 if obs_name_to_find in str(node.target) else 0 for node in prepared_for_callibrate_model.graph.nodes])
            self.assertEqual(number_of_obs_found, correct_number_of_obs_inserted)
            for (fqn, module) in prepared_for_callibrate_model.named_modules():
                is_in_include_list = isinstance(module, tuple(mods_to_check))
                if is_in_include_list:
                    self.assertTrue(hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))
                else:
                    self.assertTrue(not hasattr(module, InputWeightEqualizationDetector.DEFAULT_PRE_OBSERVER_NAME))

    @skipIfNoFBGEMM
    def test_no_outlier_report_gen(self):
        if False:
            for i in range(10):
                print('nop')
        with override_quantized_engine('fbgemm'):
            outlier_detector = OutlierDetector(reference_percentile=0.95)
            dynamic_static_detector = DynamicStaticDetector(tolerance=0.5)
            param_size: int = 4
            detector_set = {outlier_detector, dynamic_static_detector}
            model = self.LargeBatchModel(param_size=param_size)
            (prepared_for_callibrate_model, mod_report) = self._get_prepped_for_calibration_model(model, detector_set)
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)
            prepared_for_callibrate_model(example_input)
            generated_report = mod_report.generate_model_report(True)
            self.assertEqual(len(generated_report), 2)
            (outlier_str, outlier_dict) = generated_report[outlier_detector.get_detector_name()]
            self.assertEqual(len(outlier_dict), 4)
            for module_fqn in outlier_dict:
                module_dict = outlier_dict[module_fqn]
                outlier_info = module_dict[OutlierDetector.OUTLIER_KEY]
                self.assertEqual(sum(outlier_info), 0)
                self.assertEqual(len(module_dict[OutlierDetector.COMP_METRIC_KEY]), param_size)
                self.assertEqual(len(module_dict[OutlierDetector.NUM_BATCHES_KEY]), param_size)

    @skipIfNoFBGEMM
    def test_all_outlier_report_gen(self):
        if False:
            return 10
        with override_quantized_engine('fbgemm'):
            outlier_detector = OutlierDetector(ratio_threshold=1, reference_percentile=0)
            param_size: int = 16
            detector_set = {outlier_detector}
            model = self.LargeBatchModel(param_size=param_size)
            (prepared_for_callibrate_model, mod_report) = self._get_prepped_for_calibration_model(model, detector_set)
            example_input = model.get_example_inputs()[0]
            example_input = example_input.to(torch.float)
            prepared_for_callibrate_model(example_input)
            generated_report = mod_report.generate_model_report(True)
            self.assertEqual(len(generated_report), 1)
            (outlier_str, outlier_dict) = generated_report[outlier_detector.get_detector_name()]
            self.assertEqual(len(outlier_dict), 4)
            for module_fqn in outlier_dict:
                module_dict = outlier_dict[module_fqn]
                outlier_info = module_dict[OutlierDetector.OUTLIER_KEY]
                assert sum(outlier_info) >= len(outlier_info) / 2
                self.assertEqual(len(module_dict[OutlierDetector.COMP_METRIC_KEY]), param_size)
                self.assertEqual(len(module_dict[OutlierDetector.NUM_BATCHES_KEY]), param_size)

    @skipIfNoFBGEMM
    def test_multiple_run_consistent_spike_outlier_report_gen(self):
        if False:
            print('Hello World!')
        with override_quantized_engine('fbgemm'):
            outlier_detector = OutlierDetector(reference_percentile=0.95)
            param_size: int = 8
            detector_set = {outlier_detector}
            model = self.LargeBatchModel(param_size=param_size)
            (prepared_for_callibrate_model, mod_report) = self._get_prepped_for_calibration_model(model, detector_set, use_outlier_data=True)
            example_input = model.get_outlier_inputs()[0]
            example_input = example_input.to(torch.float)
            for i in range(30):
                example_input = model.get_outlier_inputs()[0]
                example_input = example_input.to(torch.float)
                if i % 14 == 0:
                    example_input[0][1] = torch.zeros_like(example_input[0][1])
                prepared_for_callibrate_model(example_input)
            generated_report = mod_report.generate_model_report(True)
            self.assertEqual(len(generated_report), 1)
            (outlier_str, outlier_dict) = generated_report[outlier_detector.get_detector_name()]
            self.assertEqual(len(outlier_dict), 4)
            for module_fqn in outlier_dict:
                module_dict = outlier_dict[module_fqn]
                sufficient_batches_info = module_dict[OutlierDetector.IS_SUFFICIENT_BATCHES_KEY]
                assert sum(sufficient_batches_info) >= len(sufficient_batches_info) / 2
                outlier_info = module_dict[OutlierDetector.OUTLIER_KEY]
                self.assertEqual(sum(outlier_info), len(outlier_info) / 2)
                self.assertEqual(len(module_dict[OutlierDetector.COMP_METRIC_KEY]), param_size)
                self.assertEqual(len(module_dict[OutlierDetector.NUM_BATCHES_KEY]), param_size)
                if module_fqn == 'linear.0':
                    counts_info = module_dict[OutlierDetector.CONSTANT_COUNTS_KEY]
                    assert sum(counts_info) >= 2
                    matched_max = sum([val == 328000000.0 for val in module_dict[OutlierDetector.MAX_VALS_KEY]])
                    self.assertEqual(matched_max, param_size / 2)

class TestFxModelReportVisualizer(QuantizationTestCase):

    def _callibrate_and_generate_visualizer(self, model, prepared_for_callibrate_model, mod_report):
        if False:
            print('Hello World!')
        '\n        Callibrates the passed in model, generates report, and returns the visualizer\n        '
        example_input = model.get_example_inputs()[0]
        example_input = example_input.to(torch.float)
        prepared_for_callibrate_model(example_input)
        generated_report = mod_report.generate_model_report(remove_inserted_observers=False)
        mod_rep_visualizer: ModelReportVisualizer = mod_report.generate_visualizer()
        return mod_rep_visualizer

    @skipIfNoFBGEMM
    def test_get_modules_and_features(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests the get_all_unique_module_fqns and get_all_unique_feature_names methods of\n        ModelReportVisualizer\n\n        Checks whether returned sets are of proper size and filtered properly\n        '
        with override_quantized_engine('fbgemm'):
            torch.backends.quantized.engine = 'fbgemm'
            detector_set = set()
            detector_set.add(OutlierDetector(reference_percentile=0.95))
            detector_set.add(InputWeightEqualizationDetector(0.5))
            model = TwoThreeOps()
            (prepared_for_callibrate_model, mod_report) = _get_prepped_for_calibration_model_helper(model, detector_set, model.get_example_inputs()[0])
            mod_rep_visualizer: ModelReportVisualizer = self._callibrate_and_generate_visualizer(model, prepared_for_callibrate_model, mod_report)
            actual_model_fqns = set(mod_rep_visualizer.generated_reports.keys())
            returned_model_fqns = mod_rep_visualizer.get_all_unique_module_fqns()
            self.assertEqual(returned_model_fqns, actual_model_fqns)
            b_1_linear_features = mod_rep_visualizer.generated_reports['block1.linear']
            returned_all_feats = mod_rep_visualizer.get_all_unique_feature_names(False)
            self.assertEqual(returned_all_feats, set(b_1_linear_features.keys()))
            plottable_set = set()
            for feature_name in b_1_linear_features:
                if type(b_1_linear_features[feature_name]) == torch.Tensor:
                    plottable_set.add(feature_name)
            returned_plottable_feats = mod_rep_visualizer.get_all_unique_feature_names()
            self.assertEqual(returned_plottable_feats, plottable_set)

    def _prep_visualizer_helper(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a mod rep visualizer that we test in various ways\n        '
        torch.backends.quantized.engine = 'fbgemm'
        detector_set = set()
        detector_set.add(OutlierDetector(reference_percentile=0.95))
        detector_set.add(InputWeightEqualizationDetector(0.5))
        model = TwoThreeOps()
        (prepared_for_callibrate_model, mod_report) = _get_prepped_for_calibration_model_helper(model, detector_set, model.get_example_inputs()[0])
        mod_rep_visualizer: ModelReportVisualizer = self._callibrate_and_generate_visualizer(model, prepared_for_callibrate_model, mod_report)
        return mod_rep_visualizer

    @skipIfNoFBGEMM
    def test_generate_tables_match_with_report(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests the generate_table_view()\n        ModelReportVisualizer\n\n        Checks whether the generated dict has proper information\n            Visual check that the tables look correct performed during testing\n        '
        with override_quantized_engine('fbgemm'):
            mod_rep_visualizer = self._prep_visualizer_helper()
            table_dict = mod_rep_visualizer.generate_filtered_tables()
            (tensor_headers, tensor_table) = table_dict[ModelReportVisualizer.TABLE_TENSOR_KEY]
            (channel_headers, channel_table) = table_dict[ModelReportVisualizer.TABLE_CHANNEL_KEY]
            tensor_info_modules = {row[1] for row in tensor_table}
            channel_info_modules = {row[1] for row in channel_table}
            combined_modules: Set = tensor_info_modules.union(channel_info_modules)
            generated_report_keys: Set = set(mod_rep_visualizer.generated_reports.keys())
            self.assertEqual(combined_modules, generated_report_keys)

    @skipIfNoFBGEMM
    def test_generate_tables_no_match(self):
        if False:
            print('Hello World!')
        '\n        Tests the generate_table_view()\n        ModelReportVisualizer\n\n        Checks whether the generated dict has proper information\n            Visual check that the tables look correct performed during testing\n        '
        with override_quantized_engine('fbgemm'):
            mod_rep_visualizer = self._prep_visualizer_helper()
            empty_tables_dict = mod_rep_visualizer.generate_filtered_tables(module_fqn_filter='random not there module')
            (tensor_headers, tensor_table) = empty_tables_dict[ModelReportVisualizer.TABLE_TENSOR_KEY]
            (channel_headers, channel_table) = empty_tables_dict[ModelReportVisualizer.TABLE_CHANNEL_KEY]
            tensor_info_modules = {row[1] for row in tensor_table}
            channel_info_modules = {row[1] for row in channel_table}
            combined_modules: Set = tensor_info_modules.union(channel_info_modules)
            self.assertEqual(len(combined_modules), 0)

    @skipIfNoFBGEMM
    def test_generate_tables_single_feat_match(self):
        if False:
            print('Hello World!')
        '\n        Tests the generate_table_view()\n        ModelReportVisualizer\n\n        Checks whether the generated dict has proper information\n            Visual check that the tables look correct performed during testing\n        '
        with override_quantized_engine('fbgemm'):
            mod_rep_visualizer = self._prep_visualizer_helper()
            single_feat_dict = mod_rep_visualizer.generate_filtered_tables(feature_filter=OutlierDetector.MAX_VALS_KEY)
            (tensor_headers, tensor_table) = single_feat_dict[ModelReportVisualizer.TABLE_TENSOR_KEY]
            (channel_headers, channel_table) = single_feat_dict[ModelReportVisualizer.TABLE_CHANNEL_KEY]
            tensor_info_features = len(tensor_headers)
            channel_info_features = len(channel_headers) - ModelReportVisualizer.NUM_NON_FEATURE_CHANNEL_HEADERS
            self.assertEqual(tensor_info_features, 0)
            self.assertEqual(channel_info_features, 1)

def _get_prepped_for_calibration_model_helper(model, detector_set, example_input, fused: bool=False):
    if False:
        for i in range(10):
            print('nop')
    'Returns a model that has been prepared for callibration and corresponding model_report'
    torch.backends.quantized.engine = 'fbgemm'
    example_input = example_input.to(torch.float)
    q_config_mapping = torch.ao.quantization.get_default_qconfig_mapping()
    if fused:
        model = torch.ao.quantization.fuse_modules(model, model.get_fusion_modules())
    model_prep = quantize_fx.prepare_fx(model, q_config_mapping, example_input)
    model_report = ModelReport(model_prep, detector_set)
    prepared_for_callibrate_model = model_report.prepare_detailed_calibration()
    return (prepared_for_callibrate_model, model_report)