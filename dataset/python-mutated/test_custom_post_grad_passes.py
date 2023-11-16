import contextlib
import torch
import torch._inductor.pattern_matcher as pattern_matcher
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.lowering import lowerings as L
from torch._inductor.pattern_matcher import Arg, CallFunction, PatternMatcherPass
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CPU

@config.patch({'freezing': True})
class TestCustomPassBase(TestCase):

    def _clone_inputs(self, inputs):
        if False:
            i = 10
            return i + 15

        def clone(x):
            if False:
                while True:
                    i = 10
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()
        return tuple((clone(x) for x in inputs))

    def _test_common(self, mod, inputs, matcher_count, matcher_nodes, atol=1e-05, rtol=1.3e-06):
        if False:
            print('Hello World!')
        counters.clear()
        maybe_autocast = contextlib.nullcontext()
        with torch.no_grad(), maybe_autocast:
            clone_inputs = self._clone_inputs(inputs)
            expected = mod(*inputs)
            actual = torch.compile(mod)(*clone_inputs)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            self.assertEqual(counters['inductor']['pattern_matcher_count'], matcher_count)
            self.assertEqual(counters['inductor']['pattern_matcher_nodes'], matcher_nodes)
aten = torch.ops.aten
mkldnn = torch.ops.mkldnn

class TestPostGradCustomPrePostPass(TestCustomPassBase):

    def _register_mkldnn_conv_relu_fusion(self, custom_pass_dict):
        if False:
            i = 10
            return i + 15

        def _mkldnn_conv_relu_pattern():
            if False:
                i = 10
                return i + 15
            return CallFunction(aten.relu, CallFunction(mkldnn._convolution_pointwise.default, Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), Arg(), _users=1))

        def _register_fusion_lowering(pattern, custom_pass_dict):
            if False:
                return 10

            def dummy_check(m):
                if False:
                    return 10
                return True

            def register_custom_lowering_pattern(pattern, extra_check, custom_pass_dict):
                if False:
                    i = 10
                    return i + 15
                return pattern_matcher.register_lowering_pattern(pattern, extra_check, pass_dict=custom_pass_dict)

            @register_custom_lowering_pattern(pattern, dummy_check, custom_pass_dict)
            def fn(match, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                computation_args = list(args)[:-3] + ['relu', [], '']
                return L[mkldnn._convolution_pointwise.default](*computation_args)
            return fn
        _register_fusion_lowering(_mkldnn_conv_relu_pattern(), custom_pass_dict)

    class _CustomPass(PatternMatcherPass):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super().__init__()

        def __call__(self, g: torch.fx.graph.Graph):
            if False:
                i = 10
                return i + 15
            self.apply(g)

    class _ConvReLU(torch.nn.Module):

        def __init__(self, ic, oc):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.conv = torch.nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            if False:
                i = 10
                return i + 15
            x1 = self.conv(x)
            return x1.relu()

    def test_custom_pre_pass(self):
        if False:
            return 10
        dafault_pattern_matcher = config.pattern_matcher
        config.pattern_matcher = False
        config.post_grad_custom_pre_pass = self._CustomPass()
        config.post_grad_custom_post_pass = None
        self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_pre_pass)
        mod = self._ConvReLU(16, 16).eval()
        x = torch.randn((1, 16, 56, 56), dtype=torch.float32)
        match_count = 1
        match_nodes = 2
        other_match_count = 1
        other_match_nodes = 1
        self._test_common(mod, (x,), match_count + other_match_count, match_nodes + other_match_nodes)
        config.pattern_matcher = dafault_pattern_matcher

    def test_custom_post_pass(self):
        if False:
            i = 10
            return i + 15
        dafault_pattern_matcher = config.pattern_matcher
        config.pattern_matcher = False
        config.post_grad_custom_pre_pass = None
        config.post_grad_custom_post_pass = self._CustomPass()
        self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_post_pass)
        mod = self._ConvReLU(16, 16).eval()
        x = torch.randn((1, 16, 56, 56), dtype=torch.float32)
        match_count = 1
        match_nodes = 2
        other_match_count = 1
        other_match_nodes = 1
        self._test_common(mod, (x,), match_count + other_match_count, match_nodes + other_match_nodes)
        config.pattern_matcher = dafault_pattern_matcher
if __name__ == '__main__':
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        run_tests()