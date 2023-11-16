import logging
import torch
import torch.ao.quantization as tq
from torch import nn
from torch.ao import pruning
from torch.testing._internal.common_utils import TestCase
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, convert_to_reference_fx, prepare_qat_fx
from torch.ao.pruning import fqn_to_module
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
sparse_defaults = {'sparsity_level': 0.8, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}

def _get_model_and_sparsifier_and_sparse_config(qconfig=None):
    if False:
        i = 10
        return i + 15
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU(), tq.QuantStub(), nn.Linear(4, 4), nn.ReLU(), tq.DeQuantStub())
    if qconfig:
        model[4].qconfig = qconfig
        model[5].qconfig = qconfig
    sparsifier = pruning.WeightNormSparsifier(**sparse_defaults)
    sparse_config = [{'tensor_fqn': '5.weight', 'sparsity_level': 0.7, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}, {'tensor_fqn': '0.weight'}]
    return (model, sparsifier, sparse_config)

def _squash_mask_calibrate_and_convert(model, sparsifier, input):
    if False:
        i = 10
        return i + 15
    sparsifier.step()
    sparsifier.squash_mask()
    model(input)
    tq.convert(model, inplace=True)

def _calculate_sparsity(tensor):
    if False:
        i = 10
        return i + 15
    return ((tensor == 0).sum() / tensor.numel()).item()

class TestComposability(TestCase):

    def test_q_prep_before_s_prep(self):
        if False:
            i = 10
            return i + 15
        (mod, sparsifier, sparse_config) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig('fbgemm'))
        tq.prepare(mod, inplace=True)
        sparsifier.prepare(mod, config=sparse_config)
        self.assertTrue(hasattr(mod[0], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'activation_post_process'))
        _squash_mask_calibrate_and_convert(mod, sparsifier, torch.randn(1, 4, 4, 4))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    def test_s_prep_before_q_prep(self):
        if False:
            return 10
        (mod, sparsifier, sparse_config) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig('fbgemm'))
        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'activation_post_process'))
        _squash_mask_calibrate_and_convert(mod, sparsifier, torch.randn(1, 4, 4, 4))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    def test_convert_without_squash_mask(self):
        if False:
            while True:
                i = 10
        (mod, sparsifier, sparse_config) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig('fbgemm'))
        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'activation_post_process'))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(mod[5].weight)
        mod(torch.randn(1, 4, 4, 4))
        tq.convert(mod, inplace=True)
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(sparsity_level, sparse_config[0]['sparsity_level'])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]['sparsity_level'])

    def test_s_prep_before_fusion(self):
        if False:
            return 10
        (mod, sparsifier, sparse_config) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig('fbgemm'))
        sparsifier.prepare(mod, config=sparse_config)
        tq.fuse_modules(mod, [['5', '6']], inplace=True)
        mod[5].qconfig = tq.get_default_qconfig('fbgemm')
        tq.prepare(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], 'parametrizations'))
        self.assertTrue(hasattr(mod[5][0], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'activation_post_process'))
        _squash_mask_calibrate_and_convert(mod, sparsifier, torch.randn(1, 4, 4, 4))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    def test_fusion_before_s_prep(self):
        if False:
            i = 10
            return i + 15
        (mod, sparsifier, _) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig('fbgemm'))
        tq.fuse_modules(mod, [['5', '6']], inplace=True)
        sparse_config = [{'tensor_fqn': '5.0.weight', 'sparsity_level': 0.7, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}, {'tensor_fqn': '0.weight'}]
        sparsifier.prepare(mod, config=sparse_config)
        mod[5].qconfig = tq.get_default_qconfig('fbgemm')
        tq.prepare(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], 'parametrizations'))
        self.assertTrue(hasattr(mod[5][0], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'activation_post_process'))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(mod[5][0].weight)
        mod(torch.randn(1, 4, 4, 4))
        tq.convert(mod, inplace=True)
        self.assertTrue(isinstance(mod[5], torch.ao.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(sparsity_level, sparse_config[0]['sparsity_level'])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]['sparsity_level'])

    def test_s_prep_before_qat_prep(self):
        if False:
            i = 10
            return i + 15
        (mod, sparsifier, sparse_config) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qat_qconfig('fbgemm'))
        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare_qat(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'activation_post_process'))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.qat.Linear))
        _squash_mask_calibrate_and_convert(mod, sparsifier, torch.randn(1, 4, 4, 4))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]['sparsity_level'])

    def test_qat_prep_before_s_prep(self):
        if False:
            while True:
                i = 10
        (mod, sparsifier, _) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qat_qconfig('fbgemm'))
        tq.prepare_qat(mod, inplace=True)
        sparse_config = [{'tensor_fqn': '5.weight', 'sparsity_level': 0.7, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}, {'tensor_fqn': '0.weight'}]
        sparsifier.prepare(mod, config=sparse_config)
        self.assertTrue(hasattr(mod[0], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'parametrizations'))
        self.assertTrue(hasattr(mod[5], 'activation_post_process'))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.qat.Linear))
        _squash_mask_calibrate_and_convert(mod, sparsifier, torch.randn(1, 4, 4, 4))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]['sparsity_level'])

def _module_has_activation_post_process(model, fqn_of_module):
    if False:
        return 10
    for node in model.graph.nodes:
        if 'activation_post_process' in node.name:
            if node.args[0].target == fqn_of_module:
                return True
    return False

class TestFxComposability(TestCase):
    """This series of tests checks that various steps of the quantization and sparsity flow
    compose cleanly despite variation in sequencing.
    """

    def test_q_prep_fx_before_s_prep(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This test checks that the ordering of prepare_fx -> sparse prepare -> convert_fx\n        compose cleanly without issue and that the final result is sparsified without\n        having to call squash mask between sparse prepare and convert_fx. This also tests the\n        automatic fusion that occurs during prepare_fx.\n        '
        (mod, sparsifier, _) = _get_model_and_sparsifier_and_sparse_config()
        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qconfig('fbgemm')
        qconfig_mapping = tq.QConfigMapping().set_module_name('4', qconfig).set_module_name('5', qconfig)
        mod = prepare_fx(mod, qconfig_mapping, (example,))
        sparse_config = [{'tensor_fqn': '5.0.weight', 'sparsity_level': 0.7, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}, {'tensor_fqn': '0.0.weight'}]
        sparsifier.prepare(mod, config=sparse_config)
        self.assertTrue(hasattr(fqn_to_module(mod, '0.0'), 'parametrizations'))
        self.assertTrue(hasattr(fqn_to_module(mod, '5.0'), 'parametrizations'))
        self.assertTrue(_module_has_activation_post_process(mod, '5'))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, '5.0.weight'))
        mod(example)
        mod = convert_fx(mod)
        self.assertTrue(isinstance(fqn_to_module(mod, '5'), torch.ao.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, '5')._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(sparsity_level, sparse_config[0]['sparsity_level'])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]['sparsity_level'])

    def test_q_prep_fx_s_prep_ref_conv(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This checks that the ordering: prepare_fx -> sparse prepare -> convert_to_reference_fx\n        compose cleanly without issue and that the final result is sparsified without\n        having to call squash mask before convert_to_reference_fx.\n        '
        (mod, sparsifier, _) = _get_model_and_sparsifier_and_sparse_config()
        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qconfig('fbgemm')
        qconfig_mapping = tq.QConfigMapping().set_module_name('4', qconfig).set_module_name('5', qconfig)
        mod = prepare_fx(mod, qconfig_mapping, (example,))
        sparse_config = [{'tensor_fqn': '5.0.weight', 'sparsity_level': 0.7, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}, {'tensor_fqn': '0.0.weight'}]
        sparsifier.prepare(mod, config=sparse_config)
        self.assertTrue(hasattr(fqn_to_module(mod, '0.0'), 'parametrizations'))
        self.assertTrue(hasattr(fqn_to_module(mod, '5.0'), 'parametrizations'))
        self.assertTrue(_module_has_activation_post_process(mod, '5'))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, '5.0.weight'))
        mod(example)
        mod = convert_to_reference_fx(mod)
        self.assertTrue(isinstance(fqn_to_module(mod, '5'), torch.ao.nn.intrinsic.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))
        self.assertTrue(isinstance(fqn_to_module(mod, '5.0'), torch.ao.nn.quantized.reference.Linear))
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, '5.0.weight'))
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(sparsity_level, sparse_config[0]['sparsity_level'])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]['sparsity_level'])

    def test_s_prep_before_q_prep_fx(self):
        if False:
            print('Hello World!')
        '\n        This test checks that the ordering of sparse prepare -> prepare_fx -> convert_fx\n        compose cleanly without issue and that the final result is sparsified without\n        having to call squash mask before convert_fx.\n        '
        (mod, sparsifier, sparse_config) = _get_model_and_sparsifier_and_sparse_config()
        sparsifier.prepare(mod, config=sparse_config)
        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qconfig('fbgemm')
        qconfig_mapping = tq.QConfigMapping().set_module_name('4', qconfig).set_module_name('5', qconfig)
        mod = prepare_fx(mod, qconfig_mapping, (example,))
        self.assertTrue(hasattr(fqn_to_module(mod, '0.0'), 'parametrizations'))
        self.assertTrue(hasattr(fqn_to_module(mod, '5.0'), 'parametrizations'))
        self.assertTrue(_module_has_activation_post_process(mod, '5'))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, '5.0.weight'))
        mod(example)
        mod = convert_fx(mod)
        self.assertTrue(isinstance(fqn_to_module(mod, '5'), torch.ao.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, '5')._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(sparsity_level, sparse_config[0]['sparsity_level'])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]['sparsity_level'])

    def test_s_prep_before_qat_prep_fx(self):
        if False:
            while True:
                i = 10
        '\n        This test checks that the ordering of sparse prepare -> prepare_qat_fx -> convert_fx\n        compose cleanly without issue and that the final result is sparsified without\n        having to call squash mask before convert_fx.\n        '
        (mod, sparsifier, sparse_config) = _get_model_and_sparsifier_and_sparse_config()
        sparsifier.prepare(mod, config=sparse_config)
        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qat_qconfig('fbgemm')
        qconfig_mapping = tq.QConfigMapping().set_module_name('4', qconfig).set_module_name('5', qconfig)
        mod = prepare_qat_fx(mod, qconfig_mapping, (example,))
        self.assertTrue(hasattr(fqn_to_module(mod, '0.0'), 'parametrizations'))
        self.assertTrue(hasattr(fqn_to_module(mod, '5'), 'parametrizations'))
        self.assertTrue(isinstance(fqn_to_module(mod, '5'), torch.ao.nn.intrinsic.qat.LinearReLU))
        self.assertTrue(_module_has_activation_post_process(mod, '5'))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, '5.weight'))
        mod(example)
        mod = convert_fx(mod)
        self.assertTrue(isinstance(fqn_to_module(mod, '5'), torch.ao.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, '5')._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(sparsity_level, sparse_config[0]['sparsity_level'])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]['sparsity_level'])

    def test_s_prep_q_prep_fx_ref(self):
        if False:
            print('Hello World!')
        '\n        This checks that the ordering: sparse prepare -> prepare_fx -> convert_to_reference_fx\n        compose cleanly without issue and that the final result is sparsified without\n        having to call squash mask before convert_to_reference_fx.\n        '
        (mod, sparsifier, sparse_config) = _get_model_and_sparsifier_and_sparse_config()
        sparsifier.prepare(mod, config=sparse_config)
        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qconfig('fbgemm')
        qconfig_mapping = tq.QConfigMapping().set_module_name('4', qconfig).set_module_name('5', qconfig)
        mod = prepare_fx(mod, qconfig_mapping, (example,))
        self.assertTrue(hasattr(fqn_to_module(mod, '0.0'), 'parametrizations'))
        self.assertTrue(hasattr(fqn_to_module(mod, '5.0'), 'parametrizations'))
        self.assertTrue(_module_has_activation_post_process(mod, '5'))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, '5.0.weight'))
        mod(example)
        mod = convert_to_reference_fx(mod)
        self.assertTrue(isinstance(fqn_to_module(mod, '5'), torch.ao.nn.intrinsic.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))
        self.assertTrue(isinstance(fqn_to_module(mod, '5.0'), torch.ao.nn.quantized.reference.Linear))
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, '5.0.weight'))
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(sparsity_level, sparse_config[0]['sparsity_level'])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]['sparsity_level'])