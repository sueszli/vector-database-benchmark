import unittest
import warnings
from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches
from torch.fx.experimental import _config as fx_config
from torch.testing._internal.common_utils import TEST_Z3
try:
    from . import test_aot_autograd, test_ctx_manager, test_export, test_functions, test_higher_order_ops, test_misc, test_modules, test_repros, test_subgraphs
except ImportError:
    import test_aot_autograd
    import test_ctx_manager
    import test_export
    import test_functions
    import test_higher_order_ops
    import test_misc
    import test_modules
    import test_repros
    import test_subgraphs
test_classes = {}

def make_dynamic_cls(cls):
    if False:
        while True:
            i = 10
    suffix = '_dynamic_shapes'
    cls_prefix = 'DynamicShapes'
    test_class = make_test_cls_with_patches(cls, cls_prefix, suffix, (config, 'assume_static_by_default', False), (config, 'specialize_int', False), (fx_config, 'translation_validation', TEST_Z3), (fx_config, 'check_shape_env_recorded_events', True), (fx_config, 'validate_shape_env_verison_key', True), xfail_prop='_expected_failure_dynamic')
    test_classes[test_class.__name__] = test_class
    globals()[test_class.__name__] = test_class
    return test_class
tests = [test_ctx_manager.CtxManagerTests, test_functions.FunctionTests, test_misc.MiscTests, test_repros.ReproTests, test_modules.NNModuleTests, test_export.ExportTests, test_subgraphs.SubGraphTests, test_higher_order_ops.HigherOrderOpTests, test_higher_order_ops.FuncTorchHigherOrderOpTests, test_aot_autograd.AotAutogradFallbackTests]
for test in tests:
    make_dynamic_cls(test)
del test
if TEST_Z3:
    unittest.expectedFailure(DynamicShapesReproTests.test_dynamic_shapes_float_guard_dynamic_shapes)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    if not TEST_Z3:
        warnings.warn('translation validation is off. Testing with translation validation requires Z3.')
    run_tests()