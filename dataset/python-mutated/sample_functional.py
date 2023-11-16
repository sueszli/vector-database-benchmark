import torch
import torch.nn.functional as F
from torch.testing._internal.common_nn import wrap_functional
'\n`sample_functional` is used by `test_cpp_api_parity.py` to test that Python / C++ API\nparity test harness works for `torch.nn.functional` functions.\n\nWhen `has_parity=true` is passed to `sample_functional`, behavior of `sample_functional`\nis the same as the C++ equivalent.\n\nWhen `has_parity=false` is passed to `sample_functional`, behavior of `sample_functional`\nis different from the C++ equivalent.\n'

def sample_functional(x, has_parity):
    if False:
        while True:
            i = 10
    if has_parity:
        return x * 2
    else:
        return x * 4
torch.nn.functional.sample_functional = sample_functional
SAMPLE_FUNCTIONAL_CPP_SOURCE = '\n\nnamespace torch {\nnamespace nn {\nnamespace functional {\n\nstruct C10_EXPORT SampleFunctionalFuncOptions {\n  SampleFunctionalFuncOptions(bool has_parity) : has_parity_(has_parity) {}\n\n  TORCH_ARG(bool, has_parity);\n};\n\nTensor sample_functional(Tensor x, SampleFunctionalFuncOptions options) {\n    return x * 2;\n}\n\n} // namespace functional\n} // namespace nn\n} // namespace torch\n'
functional_tests = [dict(constructor=wrap_functional(F.sample_functional, has_parity=True), cpp_options_args='F::SampleFunctionalFuncOptions(true)', input_size=(1, 2, 3), fullname='sample_functional_has_parity', has_parity=True), dict(constructor=wrap_functional(F.sample_functional, has_parity=False), cpp_options_args='F::SampleFunctionalFuncOptions(false)', input_size=(1, 2, 3), fullname='sample_functional_no_parity', has_parity=False), dict(constructor=wrap_functional(F.sample_functional, has_parity=False), cpp_options_args='F::SampleFunctionalFuncOptions(false)', input_size=(1, 2, 3), fullname='sample_functional_THIS_TEST_SHOULD_BE_SKIPPED', test_cpp_api_parity=False)]