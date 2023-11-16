import copy
import math
import torch
import torch.nn as nn
import torch.backends.mkldnn
from torch.nn import Conv2d, BatchNorm2d, ReLU, init
from torch.ao.nn.intrinsic.qat import ConvBn2d, ConvBnReLU2d
from torch.nn.modules.utils import _pair
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.qat as nnqat
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat.dynamic as nnqatd
from torch.ao.quantization import prepare, convert, prepare_qat, quantize_qat, QuantStub, DeQuantStub, default_qconfig, default_qat_qconfig, default_embedding_qat_qconfig, default_symmetric_qnnpack_qat_qconfig, get_default_qat_qconfig, FixedQParamsFakeQuantize, FusedMovingAvgObsFakeQuantize, get_embedding_qat_module_mappings, get_embedding_static_quant_module_mappings, NoopObserver
from torch.ao.quantization.qconfig import qconfig_equals
from torch.testing._internal.common_quantization import DeFusedEmbeddingBagLinear, QuantizationTestCase, QuantStubModel, ManualLinearQATModel, ManualDropoutQATModel, ManualLinearDynamicQATModel, ManualConvLinearQATModel, ManualConvLinearSymmQATModel, ManualEmbeddingBagLinear, TwoLayerLinearModel, test_only_eval_fn, test_only_train_fn
from torch.testing._internal.common_quantized import override_quantized_engine, supported_qengines, override_qengines
from torch.testing._internal.common_utils import skipIfNoXNNPACK
from hypothesis import given
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()
from functools import reduce

class _ReferenceConvBnNd(torch.nn.Conv2d, torch.nn.modules.conv._ConvNd):
    """
    Conv-BN fusion implemented with explicit folding. Useful
    to verify numerical equivalency with non-folded version.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode, eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None):
        if False:
            i = 10
            return i + 15
        nn.modules.conv._ConvNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, False, padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.eps = eps
        self.momentum = momentum
        self.freeze_bn = freeze_bn if self.training else True
        self.num_features = out_channels
        self.gamma = nn.Parameter(torch.empty(out_channels))
        self.beta = nn.Parameter(torch.empty(out_channels))
        self.affine = True
        self.track_running_stats = True
        self.register_buffer('running_mean', torch.zeros(out_channels))
        self.register_buffer('running_var', torch.ones(out_channels))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.activation_post_process = self.qconfig.activation()
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_bn_parameters()

    def reset_running_stats(self):
        if False:
            return 10
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_bn_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        self.reset_running_stats()
        init.uniform_(self.gamma)
        init.zeros_(self.beta)
        if self.bias is not None:
            (fan_in, _) = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def reset_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        super().reset_parameters()
        if hasattr(self, 'gamma'):
            self.reset_bn_parameters()

    def update_bn_stats(self):
        if False:
            i = 10
            return i + 15
        self.freeze_bn = False
        return self

    def freeze_bn_stats(self):
        if False:
            i = 10
            return i + 15
        self.freeze_bn = True
        return self

    def _forward(self, input):
        if False:
            while True:
                i = 10
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and (not self.freeze_bn) and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        running_std = torch.sqrt(self.running_var + self.eps)
        scale_factor = self.gamma / running_std
        scaled_weight = self.weight * scale_factor.reshape([-1, 1, 1, 1])
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias, dtype=input.dtype)
        else:
            zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device, dtype=input.dtype)
        conv = self._conv_forward(input, self.weight_fake_quant(scaled_weight), zero_bias)
        if self.training and (not self.freeze_bn):
            if self.bias is not None:
                conv_orig = conv / scale_factor.reshape([1, -1, 1, 1]) + self.bias.reshape([1, -1, 1, 1])
            else:
                conv_orig = conv / scale_factor.reshape([1, -1, 1, 1])
            batch_mean = torch.mean(conv_orig, dim=[0, 2, 3])
            batch_var = torch.var(conv_orig, dim=[0, 2, 3], unbiased=False)
            n = float(conv_orig.numel() / conv_orig.size()[1])
            unbiased_batch_var = batch_var * (n / (n - 1))
            batch_rstd = torch.ones_like(batch_var, memory_format=torch.contiguous_format) / torch.sqrt(batch_var + self.eps)
            conv = (self.gamma * batch_rstd).reshape([1, -1, 1, 1]) * conv_orig + (self.beta - self.gamma * batch_rstd * batch_mean).reshape([1, -1, 1, 1])
            self.running_mean = exponential_average_factor * batch_mean.detach() + (1 - exponential_average_factor) * self.running_mean
            self.running_var = exponential_average_factor * unbiased_batch_var.detach() + (1 - exponential_average_factor) * self.running_var
        elif self.bias is None:
            conv = conv + (self.beta - self.gamma * self.running_mean / running_std).reshape([1, -1, 1, 1])
        else:
            conv = conv + (self.gamma * (self.bias - self.running_mean) / running_std + self.beta).reshape([1, -1, 1, 1])
        return conv

    def extra_repr(self):
        if False:
            while True:
                i = 10
        return super().extra_repr()

    def forward(self, input):
        if False:
            print('Hello World!')
        return self.activation_post_process(self._forward(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        if False:
            return 10
        'Create a qat module from a float module or qparams_dict\n            Args: `mod` a float module, either produced by torch.ao.quantization utilities\n            or directly from user\n        '
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig
        (conv, bn) = (mod[0], mod[1])
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None, conv.padding_mode, bn.eps, bn.momentum, False, qconfig)
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.gamma = bn.weight
        qat_convbn.beta = bn.bias
        qat_convbn.running_mean = bn.running_mean
        qat_convbn.running_var = bn.running_var
        qat_convbn.num_batches_tracked = bn.num_batches_tracked
        return qat_convbn

class _ReferenceConvBn2d(_ReferenceConvBnNd, nn.Conv2d):
    _FLOAT_MODULE = torch.ao.nn.intrinsic.ConvBn2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None, padding_mode='zeros', eps=1e-05, momentum=0.1, freeze_bn=False, qconfig=None):
        if False:
            while True:
                i = 10
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        _ReferenceConvBnNd.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode, eps, momentum, freeze_bn, qconfig)

class TestQuantizeEagerQAT(QuantizationTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.embed_linear_data_train = [[torch.randint(0, 10, (12, 12), dtype=torch.long), torch.randn((12, 1), dtype=torch.float)] for _ in range(2)]
        self.embed_data = [[torch.randint(0, 10, (12, 1))]]

    def test_manual(self):
        if False:
            return 10
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualLinearQATModel(qengine)
                model = prepare_qat(model)
                self.checkObservers(model)
                test_only_train_fn(model, self.train_data)
                model = convert(model)

                def checkQuantized(model):
                    if False:
                        i = 10
                        return i + 15
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    self.assertEqual(type(model.fc2), nnq.Linear)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)
                    self.checkNoQconfig(model)
                checkQuantized(model)
                model = quantize_qat(ManualLinearQATModel(qengine), test_only_train_fn, [self.train_data])
                checkQuantized(model)

    def test_dropout(self):
        if False:
            while True:
                i = 10
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualDropoutQATModel(qengine)
                model = prepare_qat(model)
                self.checkObservers(model)
                test_only_train_fn(model, self.train_data)
                model = convert(model)

                def checkQuantized(model):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    self.assertEqual(type(model.dropout), nnq.Dropout)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)
                    self.checkNoQconfig(model)
                checkQuantized(model)
                model = quantize_qat(ManualDropoutQATModel(qengine), test_only_train_fn, [self.train_data])
                checkQuantized(model)

    def test_eval_only_fake_quant(self):
        if False:
            i = 10
            return i + 15
        'Using FakeQuant in evaluation only mode,\n        this is useful for estimating accuracy loss when we quantize the\n        network\n        '
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualLinearQATModel(qengine)
                model = prepare_qat(model)
                self.checkObservers(model)
                model.eval()
                test_only_eval_fn(model, self.calib_data)

    def test_conv_linear(self):
        if False:
            i = 10
            return i + 15
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualConvLinearQATModel()
                model = prepare_qat(model)
                self.checkObservers(model)
                test_only_train_fn(model, self.img_data_2d_train)
                model = convert(model)

                def checkQuantized(model):
                    if False:
                        while True:
                            i = 10
                    self.assertEqual(type(model.conv), nnq.Conv2d)
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    self.assertEqual(type(model.fc2), nnq.Linear)
                    test_only_eval_fn(model, self.img_data_2d)
                    self.checkScriptable(model, self.img_data_2d)
                    self.checkNoQconfig(model)
                checkQuantized(model)
                model = ManualConvLinearQATModel()
                model = quantize_qat(model, test_only_train_fn, [self.img_data_2d_train])
                checkQuantized(model)

    @skipIfNoXNNPACK
    def test_conv_linear_symm(self):
        if False:
            while True:
                i = 10
        'Same as test_conv_linear but with Symmetric quantization.\n        Supported only with qengine=qnnpack, which uses symmetric\n        kernels from xnnpack library.'
        for qengine in supported_qengines:
            if qengine != 'qnnpack':
                continue
            with override_quantized_engine(qengine):
                model = ManualConvLinearSymmQATModel()
                model = prepare_qat(model)
                self.checkObservers(model)
                test_only_train_fn(model, self.img_data_2d_train)
                model = convert(model)

                def checkQuantized(model):
                    if False:
                        print('Hello World!')
                    self.assertEqual(type(model.conv), nnq.Conv2d)
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    self.assertEqual(type(model.fc2), nnq.Linear)
                    test_only_eval_fn(model, self.img_data_2d)
                    self.checkScriptable(model, self.img_data_2d)
                    self.checkNoQconfig(model)
                checkQuantized(model)
                model = ManualConvLinearSymmQATModel()
                model = quantize_qat(model, test_only_train_fn, [self.img_data_2d_train])
                checkQuantized(model)

    def test_dynamic_qat_linear(self):
        if False:
            return 10
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                with self.assertRaisesRegex(ValueError, 'Dynamic QAT requires a memoryless observer.' + 'This means a MovingAverage observer with averaging constant equal to 1'):
                    model = ManualLinearDynamicQATModel(default_qat_qconfig)
                    model = prepare_qat(model, mapping={torch.nn.Linear: nnqatd.Linear})
                model = ManualLinearDynamicQATModel()
                model = prepare_qat(model, mapping={torch.nn.Linear: nnqatd.Linear})
                self.assertEqual(type(model.fc1), nnqatd.Linear)
                self.assertEqual(type(model.fc2), nnqatd.Linear)
                self.checkObservers(model)
                test_only_train_fn(model, self.train_data)
                model = convert(model, mapping={nnqatd.Linear: nnqd.Linear})
                self.assertEqual(type(model.fc1), nnqd.Linear)
                self.assertEqual(type(model.fc2), nnqd.Linear)
                test_only_eval_fn(model, self.calib_data)
                self.checkScriptable(model, self.calib_data)
                self.checkNoQconfig(model)

    def test_defused_embedding_bag_linear(self):
        if False:
            while True:
                i = 10
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = DeFusedEmbeddingBagLinear().train()
                model = prepare_qat(model, mapping=get_embedding_qat_module_mappings())
                self.checkObservers(model)
                test_only_train_fn(model, self.embed_linear_data_train)
                self.assertEqual(type(model.linear.activation_post_process), FusedMovingAvgObsFakeQuantize)
                self.assertEqual(type(model.emb.activation_post_process), NoopObserver)
                self.assertEqual(model.emb.weight_fake_quant.zero_point.dtype, torch.float32)
                self.assertEqual(model.linear.weight_fake_quant.zero_point.dtype, torch.int32)
                model = convert(model, mapping=get_embedding_static_quant_module_mappings())

                def checkQuantized(model):
                    if False:
                        return 10
                    self.assertEqual(type(model.emb), nn.quantized.Embedding)
                    self.assertEqual(type(model.linear), nn.quantized.Linear)
                    test_only_eval_fn(model, self.embed_data)
                    self.checkScriptable(model, self.embed_data)
                    self.checkNoQconfig(model)
                checkQuantized(model)

    def test_embedding_bag_linear(self):
        if False:
            for i in range(10):
                print('nop')
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualEmbeddingBagLinear().train()
                model = prepare_qat(model, mapping=get_embedding_qat_module_mappings())
                self.checkObservers(model)
                test_only_train_fn(model, self.embed_linear_data_train)
                self.assertFalse(hasattr(model, 'activation_post_process'))
                self.assertEqual(model.emb.weight_fake_quant.zero_point.dtype, torch.float32)
                self.assertEqual(model.linear.weight_fake_quant.zero_point.dtype, torch.int32)
                model = convert(model, mapping=get_embedding_static_quant_module_mappings())

                def checkQuantized(model):
                    if False:
                        print('Hello World!')
                    self.assertTrue(type(model.emb), nn.quantized.EmbeddingBag)
                    self.assertTrue(type(model.linear), nnq.Linear)
                    test_only_eval_fn(model, self.embed_data)
                    self.checkScriptable(model, self.embed_data)
                    self.checkNoQconfig(model)
                checkQuantized(model)
                model = ManualEmbeddingBagLinear()

    def test_train_save_load_eval(self):
        if False:
            return 10
        'Test QAT flow of creating a model, doing QAT and saving the quantized state_dict\n        During eval, we first call prepare_qat and conver on the model and then load the state_dict\n        and compare results against original model\n        '
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = TwoLayerLinearModel()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
                model = prepare_qat(model)
                fq_state_dict = model.state_dict()
                test_only_train_fn(model, self.train_data)
                model = convert(model)
                quant_state_dict = model.state_dict()
                x = torch.rand(2, 5, dtype=torch.float)
                ref = model(x)
                model = TwoLayerLinearModel()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
                torch.ao.quantization.prepare_qat(model, inplace=True)
                new_state_dict = model.state_dict()
                self.assertEqual(set(fq_state_dict.keys()), set(new_state_dict.keys()))
                torch.ao.quantization.convert(model, inplace=True)
                model.eval()
                model.load_state_dict(quant_state_dict)
                out = model(x)
                self.assertEqual(ref, out)
                model = TwoLayerLinearModel()
                model.eval()
                model = torch.ao.quantization.QuantWrapper(model)
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                torch.ao.quantization.prepare(model, inplace=True)
                torch.ao.quantization.convert(model, inplace=True)
                self.assertEqual(set(model.state_dict().keys()), set(quant_state_dict.keys()))
                model.eval()
                model.load_state_dict(quant_state_dict)
                out = model(x)
                self.assertEqual(ref, out)

    @override_qengines
    def test_forward_hooks_preserved(self):
        if False:
            while True:
                i = 10
        'Test QAT on preserving pre forward and post forward hooks of original model\n        '
        qengine = torch.backends.quantized.engine
        model = QuantStubModel()
        counter = {'pre_forwards': 0, 'forwards': 0}

        def fw_pre_hook(h_module, input):
            if False:
                while True:
                    i = 10
            counter['pre_forwards'] += 1

        def fw_hook(h_module, input, output):
            if False:
                return 10
            counter['forwards'] += 1
        model.fc.register_forward_pre_hook(fw_pre_hook)
        model.fc.register_forward_hook(fw_hook)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        model = prepare_qat(model)

        def checkHooksIsPresent(model, before_convert=True):
            if False:
                print('Hello World!')
            forward_hooks = 1
            if before_convert:
                self.assertEqual(len(model.quant._forward_hooks.values()), 1, 'Quantization observer hook has disappeared')
                forward_hooks = 2
            self.assertObjectIn(fw_pre_hook, model.fc._forward_pre_hooks.values())
            self.assertObjectIn(fw_hook, model.fc._forward_hooks.values())
            self.assertEqual(len(model.fc._forward_pre_hooks.values()), 1, 'Extra pre forward hooks have appeared on a layer')
            self.assertEqual(len(model.fc._forward_hooks.values()), forward_hooks, 'Extra post forward hooks have appeared on a layer')
        checkHooksIsPresent(model, True)
        x = torch.rand(2, 5, dtype=torch.float)
        model(x)
        torch.ao.quantization.convert(model, inplace=True)
        checkHooksIsPresent(model, False)

    def test_add_scalar_uses_input_qparams(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.ff = torch.ao.nn.quantized.FloatFunctional()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.quant(x)
                x = self.ff.add_scalar(x, 1.0)
                return x
        m = M()
        m.qconfig = torch.ao.quantization.default_qconfig
        mp = torch.ao.quantization.prepare_qat(m)
        mp(torch.randn(4, 4))
        mq = torch.ao.quantization.convert(mp)
        res = mq(torch.randn(4, 4))
        eps = 1e-05
        self.assertTrue(torch.abs(mq.quant.scale - res.q_scale()) < eps)

    def test_mul_scalar_uses_input_qparams(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.ff = torch.ao.nn.quantized.FloatFunctional()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.quant(x)
                x = self.ff.mul_scalar(x, 2.0)
                return x
        m = M()
        m.qconfig = torch.ao.quantization.default_qconfig
        mp = torch.ao.quantization.prepare_qat(m)
        mp(torch.randn(4, 4))
        mq = torch.ao.quantization.convert(mp)
        res = mq(torch.randn(4, 4))
        eps = 1e-05
        self.assertTrue(torch.abs(mq.quant.scale * 2 - res.q_scale()) < eps)

    @override_qengines
    def test_qat_embedding_bag_errors(self):
        if False:
            i = 10
            return i + 15
        default_qat_qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)
        with self.assertRaisesRegex(AssertionError, 'qconfig must be provided for QAT module'):
            nnqat.EmbeddingBag(10, 5, qconfig=None)
        with self.assertRaisesRegex(AssertionError, 'Embedding Bag weights requires a qscheme of ' + 'torch.per_channel_affine_float_qparams'):
            nnqat.EmbeddingBag(10, 5, qconfig=default_qat_qconfig)
        embed = nn.Embedding(10, 5)
        with self.assertRaisesRegex(AssertionError, 'qat.EmbeddingBag.from_float only works for EmbeddingBag'):
            nnqat.EmbeddingBag.from_float(embed)
        embed_bag = nn.EmbeddingBag(10, 5)
        with self.assertRaisesRegex(AssertionError, 'Input float module must have qconfig defined'):
            nnqat.EmbeddingBag.from_float(embed_bag)
        embed_bag.qconfig = None
        with self.assertRaisesRegex(AssertionError, 'Input float module must have a valid qconfig'):
            nnqat.EmbeddingBag.from_float(embed_bag)
        embed_bag.qconfig = default_qat_qconfig
        with self.assertRaisesRegex(AssertionError, 'Embedding Bag weights requires a qscheme of ' + 'torch.per_channel_affine_float_qparams'):
            nnqat.EmbeddingBag.from_float(embed_bag)

    def test_embedding_qat_qconfig_equal(self):
        if False:
            print('Hello World!')
        model = ManualEmbeddingBagLinear().train()
        model = prepare_qat(model)
        self.assertTrue(qconfig_equals(model.emb.qconfig, default_embedding_qat_qconfig))

class TestQuantizeEagerQATNumerics(QuantizationTestCase):

    def _test_activation_convert_numerics_impl(self, Act, data):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.act = Act()
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.quant(x)
                x = self.act(x)
                x = self.dequant(x)
                return x
        m = M().train()
        m.qconfig = default_qat_qconfig
        m = prepare_qat(m)
        before_convert = m(data)
        m = convert(m)
        after_convert = m(data)
        self.assertEqual(before_convert, after_convert)

    def test_fixed_qparam_ops(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.tanh = torch.nn.Tanh()
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.quant(x)
                x = self.sigmoid(x)
                x = self.hardsigmoid(x)
                x = self.tanh(x)
                x = self.dequant(x)
                return x
        m = M().train()
        m.qconfig = default_qat_qconfig
        m = prepare_qat(m)
        for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
            self.assertEqual(type(getattr(m, attr).activation_post_process), FixedQParamsFakeQuantize)
        data = torch.randn(1, 3, 2, 4)
        before_convert = m(data)
        m = convert(m)
        after_convert = m(data)
        self.assertEqual(before_convert, after_convert)
        for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
            self.assertFalse(hasattr(getattr(m, attr), 'activation_post_process'))
            self.assertTrue(len(getattr(m, attr)._forward_hooks.items()) == 0)

        def checkNoFQModule(m):
            if False:
                return 10
            for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
                self.assertFalse(hasattr(getattr(m, attr), 'activation_post_process'))
                self.assertTrue(len(getattr(m, attr)._forward_hooks.items()) == 0)
        m = M().eval()
        m.qconfig = default_qconfig
        m = prepare(m)
        checkNoFQModule(m)
        m = convert(m)
        checkNoFQModule(m)

    def test_leaky_relu(self):
        if False:
            print('Hello World!')
        data = torch.randn(1, 3, 2, 4)
        self._test_activation_convert_numerics_impl(nn.LeakyReLU, data)

    def test_relu(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                if False:
                    return 10
                x = self.relu(x)
                return x
        m = M().train()
        m.qconfig = default_qconfig
        m = prepare_qat(m)
        self.assertFalse(hasattr(m, 'activation_post_process'))
        m = convert(m)
        self.assertTrue(type(m.relu), nn.ReLU)

    @given(batch_size=st.integers(2, 4), input_channels_per_group=st.sampled_from([2, 3, 4]), height=st.integers(5, 10), width=st.integers(5, 10), output_channels_per_group=st.sampled_from([2, 3]), groups=st.integers(1, 3), kernel_h=st.integers(1, 3), kernel_w=st.integers(1, 3), stride_h=st.integers(1, 2), stride_w=st.integers(1, 2), pad_h=st.integers(0, 2), pad_w=st.integers(0, 2), dilation=st.integers(1, 1), padding_mode=st.sampled_from(['zeros', 'circular']), use_relu=st.booleans(), eps=st.sampled_from([1e-05, 0.0001, 0.001]), momentum=st.sampled_from([0.1, 0.2, 0.3]), freeze_bn=st.booleans(), zero_gamma=st.booleans(), has_bias=st.booleans(), use_slow_fusion=st.booleans())
    def test_conv_bn_relu(self, batch_size, input_channels_per_group, height, width, output_channels_per_group, groups, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation, padding_mode, use_relu, eps, momentum, freeze_bn, zero_gamma, has_bias, use_slow_fusion):
        if False:
            while True:
                i = 10
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        dilation_h = dilation_w = dilation
        conv_op = Conv2d(input_channels, output_channels, (kernel_h, kernel_w), (stride_h, stride_w), (pad_h, pad_w), (dilation_h, dilation_w), groups, has_bias, padding_mode).to(dtype=torch.double)
        bn_op = BatchNorm2d(output_channels, eps, momentum).to(dtype=torch.double)
        relu_op = ReLU()
        cls = ConvBnReLU2d if use_relu else ConvBn2d
        qat_op = cls(input_channels, output_channels, (kernel_h, kernel_w), (stride_h, stride_w), (pad_h, pad_w), (dilation_h, dilation_w), groups, has_bias, padding_mode, eps, momentum, freeze_bn=True, qconfig=default_qat_qconfig).to(dtype=torch.double)
        qat_op._enable_slow_path_for_better_numerical_stability = use_slow_fusion
        if zero_gamma and use_slow_fusion:
            torch.nn.init.zeros_(qat_op.bn.weight)
        qat_op.apply(torch.ao.quantization.disable_fake_quant)
        if freeze_bn:
            qat_op.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
        else:
            qat_op.apply(torch.ao.nn.intrinsic.qat.update_bn_stats)
        input = torch.randn(batch_size, input_channels, height, width, dtype=torch.double, requires_grad=True)
        conv_op.weight = torch.nn.Parameter(qat_op.weight.detach())
        if has_bias:
            conv_op.bias = torch.nn.Parameter(qat_op.bias.detach())
        bn_op.running_mean = qat_op.bn.running_mean.clone()
        bn_op.running_var = qat_op.bn.running_var.clone()
        bn_op.weight = torch.nn.Parameter(qat_op.bn.weight.detach())
        bn_op.bias = torch.nn.Parameter(qat_op.bn.bias.detach())

        def compose(functions):
            if False:
                return 10
            return reduce(lambda f, g: lambda x: f(g(x)), functions[::-1], lambda x: x)
        if not use_relu:

            def relu_op(x):
                if False:
                    while True:
                        i = 10
                return x
        if freeze_bn:

            def ref_op(x):
                if False:
                    for i in range(10):
                        print('nop')
                x = conv_op(x)
                x = (x - bn_op.running_mean.reshape([1, -1, 1, 1])) * (bn_op.weight / torch.sqrt(bn_op.running_var + bn_op.eps)).reshape([1, -1, 1, 1]) + bn_op.bias.reshape([1, -1, 1, 1])
                x = relu_op(x)
                return x
        else:
            ref_op = compose([conv_op, bn_op, relu_op])
        input_clone = input.clone().detach().requires_grad_()
        for i in range(2):
            result_ref = ref_op(input)
            result_actual = qat_op(input_clone)
            self.assertEqual(result_ref, result_actual)
            dout = torch.randn(result_ref.size(), dtype=torch.double)
            loss = (result_ref - dout).sum()
            loss.backward()
            input_grad_ref = input.grad.cpu()
            weight_grad_ref = conv_op.weight.grad.cpu()
            gamma_grad_ref = bn_op.weight.grad.cpu()
            beta_grad_ref = bn_op.bias.grad.cpu()
            running_mean_ref = bn_op.running_mean
            running_var_ref = bn_op.running_var
            num_batches_tracked_ref = bn_op.num_batches_tracked
            loss = (result_actual - dout).sum()
            loss.backward()
            input_grad_actual = input_clone.grad.cpu()
            weight_grad_actual = qat_op.weight.grad.cpu()
            gamma_grad_actual = qat_op.bn.weight.grad.cpu()
            beta_grad_actual = qat_op.bn.bias.grad.cpu()
            running_mean_actual = qat_op.bn.running_mean
            running_var_actual = qat_op.bn.running_var
            num_batches_tracked_actual = qat_op.bn.num_batches_tracked
            precision = 1e-10
            self.assertEqual(input_grad_ref, input_grad_actual, atol=precision, rtol=0)
            self.assertEqual(weight_grad_ref, weight_grad_actual, atol=precision, rtol=0)
            self.assertEqual(gamma_grad_ref, gamma_grad_actual, atol=precision, rtol=0)
            self.assertEqual(beta_grad_ref, beta_grad_actual, atol=precision, rtol=0)
            self.assertEqual(num_batches_tracked_ref, num_batches_tracked_actual, atol=precision, rtol=0)
            self.assertEqual(running_mean_ref, running_mean_actual, atol=precision, rtol=0)
            self.assertEqual(running_var_ref, running_var_actual, atol=precision, rtol=0)

    @given(batch_size=st.integers(2, 4), input_channels_per_group=st.sampled_from([2, 3, 4]), height=st.integers(5, 10), width=st.integers(5, 10), output_channels_per_group=st.sampled_from([2, 3]), groups=st.integers(1, 3), kernel_h=st.integers(1, 3), kernel_w=st.integers(1, 3), stride_h=st.integers(1, 2), stride_w=st.integers(1, 2), pad_h=st.integers(0, 2), pad_w=st.integers(0, 2), dilation=st.integers(1, 1), padding_mode=st.sampled_from(['zeros', 'circular']), eps=st.sampled_from([1e-05, 0.0001, 0.001]), momentum=st.sampled_from([0.1, 0.2, 0.3]), freeze_bn=st.booleans(), bias=st.booleans())
    def test_conv_bn_folded_vs_unfolded(self, batch_size, input_channels_per_group, height, width, output_channels_per_group, groups, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation, padding_mode, eps, momentum, freeze_bn, bias):
        if False:
            while True:
                i = 10
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        dilation_h = dilation_w = dilation
        qat_op = ConvBn2d(input_channels, output_channels, (kernel_h, kernel_w), (stride_h, stride_w), (pad_h, pad_w), (dilation_h, dilation_w), groups, bias, padding_mode, eps, momentum, freeze_bn=freeze_bn, qconfig=default_qat_qconfig).to(dtype=torch.double)
        qat_ref_op = _ReferenceConvBn2d(input_channels, output_channels, (kernel_h, kernel_w), (stride_h, stride_w), (pad_h, pad_w), (dilation_h, dilation_w), groups, bias, padding_mode, eps, momentum, freeze_bn=freeze_bn, qconfig=default_qat_qconfig).to(dtype=torch.double)
        qat_op.apply(torch.ao.quantization.disable_fake_quant)
        qat_ref_op.apply(torch.ao.quantization.disable_fake_quant)
        qat_ref_op.weight = torch.nn.Parameter(qat_op.weight.detach().clone())
        qat_ref_op.running_mean = qat_op.bn.running_mean.clone()
        qat_ref_op.running_var = qat_op.bn.running_var.clone()
        qat_ref_op.gamma = torch.nn.Parameter(qat_op.bn.weight.detach().clone())
        qat_ref_op.beta = torch.nn.Parameter(qat_op.bn.bias.detach().clone())
        if qat_op.bias is not None:
            qat_ref_op.bias = torch.nn.Parameter(qat_op.bias.detach().clone())
        lr = 0.01
        qat_op_optim = torch.optim.SGD(qat_op.parameters(), lr=lr)
        qat_ref_op_optim = torch.optim.SGD(qat_ref_op.parameters(), lr=lr)
        for i in range(5):
            qat_op.train()
            qat_ref_op.train()
            qat_op_optim.zero_grad()
            qat_ref_op_optim.zero_grad()
            input = torch.randn(batch_size, input_channels, height, width, dtype=torch.double, requires_grad=True)
            input_clone = input.clone().detach().requires_grad_()
            if i > 2:
                qat_op.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
                qat_ref_op.freeze_bn_stats()
            if i > 3:
                qat_op.apply(torch.ao.quantization.disable_observer)
                qat_ref_op.apply(torch.ao.quantization.disable_observer)
            result_ref = qat_ref_op(input)
            result_actual = qat_op(input_clone)
            self.assertEqual(result_ref, result_actual)
            dout = torch.randn(result_ref.size(), dtype=torch.double) + 10.0
            loss = (result_ref - dout).sum()
            loss.backward()
            input_grad_ref = input.grad.cpu()
            weight_grad_ref = qat_ref_op.weight.grad.cpu()
            gamma_grad_ref = qat_ref_op.gamma.grad.cpu()
            beta_grad_ref = qat_ref_op.beta.grad.cpu()
            running_mean_ref = qat_ref_op.running_mean
            running_var_ref = qat_ref_op.running_var
            num_batches_tracked_ref = qat_ref_op.num_batches_tracked
            loss = (result_actual - dout).sum()
            loss.backward()
            input_grad_actual = input_clone.grad.cpu()
            weight_grad_actual = qat_op.weight.grad.cpu()
            gamma_grad_actual = qat_op.bn.weight.grad.cpu()
            beta_grad_actual = qat_op.bn.bias.grad.cpu()
            running_mean_actual = qat_op.bn.running_mean
            running_var_actual = qat_op.bn.running_var
            num_batches_tracked_actual = qat_op.bn.num_batches_tracked
            precision = 1e-05
            self.assertEqual(input_grad_ref, input_grad_actual, atol=precision, rtol=0)
            self.assertEqual(weight_grad_ref, weight_grad_actual, atol=precision, rtol=0)
            self.assertEqual(gamma_grad_ref, gamma_grad_actual, atol=precision, rtol=0)
            self.assertEqual(beta_grad_ref, beta_grad_actual, atol=precision, rtol=0)
            self.assertEqual(num_batches_tracked_ref, num_batches_tracked_actual, atol=precision, rtol=0)
            self.assertEqual(running_mean_ref, running_mean_actual, atol=precision, rtol=0)
            self.assertEqual(running_var_ref, running_var_actual, atol=precision, rtol=0)
            qat_op_optim.step()
            qat_ref_op_optim.step()

    @override_qengines
    def test_linear_bn_numerics(self):
        if False:
            i = 10
            return i + 15
        qengine = torch.backends.quantized.engine
        m_ref = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        m_ref_copy = copy.deepcopy(m_ref)
        m_ref_copy = torch.ao.quantization.fuse_modules_qat(m_ref_copy, [['0', '1']])
        qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        m_ref_copy[0].qconfig = qconfig
        m = nniqat.LinearBn1d.from_float(m_ref_copy[0])
        m.apply(torch.ao.quantization.disable_fake_quant)
        data = torch.randn(4, 4)
        r1 = m_ref(data)
        r2 = m(data)
        self.assertTrue(torch.allclose(r1, r2))

    @skipIfNoXNNPACK
    @override_qengines
    def test_linear_bn_symm_numerics(self):
        if False:
            while True:
                i = 10
        qengine = torch.backends.quantized.engine
        if qengine != 'qnnpack':
            return
        m_ref = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        m_ref_copy = copy.deepcopy(m_ref)
        m_ref_copy = torch.ao.quantization.fuse_modules_qat(m_ref_copy, [['0', '1']])
        qconfig = default_symmetric_qnnpack_qat_qconfig
        m_ref_copy[0].qconfig = qconfig
        m = nniqat.LinearBn1d.from_float(m_ref_copy[0])
        m.apply(torch.ao.quantization.disable_fake_quant)
        data = torch.randn(4, 4)
        r1 = m_ref(data)
        r2 = m(data)
        self.assertTrue(torch.allclose(r1, r2))

    @override_qengines
    def test_linear_bn_workflow(self):
        if False:
            i = 10
            return i + 15
        qengine = torch.backends.quantized.engine
        m = nn.Sequential(QuantStub(), nn.Linear(4, 4), nn.BatchNorm1d(4))
        data = torch.randn(4, 4)
        m.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        m = torch.ao.quantization.fuse_modules_qat(m, [['1', '2']])
        mp = prepare_qat(m)
        mp(data)
        mq = convert(mp)
        self.assertTrue(type(mq[1]) == nnq.Linear)
        self.assertTrue(type(mq[2]) == nn.Identity)
if __name__ == '__main__':
    raise RuntimeError('This test file is not meant to be run directly, use:\n\n\tpython test/test_quantization.py TESTNAME\n\ninstead.')