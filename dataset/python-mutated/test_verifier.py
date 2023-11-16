import unittest
import torch
from functorch.experimental import control_flow
from torch import Tensor
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export import export
from torch._export.verifier import SpecViolationError, Verifier
from torch.export.exported_program import InputKind, InputSpec, TensorArgument
from torch.testing._internal.common_utils import run_tests, TestCase

@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestVerifier(TestCase):

    def test_verifier_basic(self) -> None:
        if False:
            while True:
                i = 10

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if False:
                print('Hello World!')
            return x + y
        ep = export(f, (torch.randn(100), torch.randn(100)))
        verifier = Verifier()
        verifier.check(ep)

    def test_verifier_call_module(self) -> None:
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self) -> None:
                if False:
                    print('Hello World!')
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    while True:
                        i = 10
                return self.linear(x)
        gm = torch.fx.symbolic_trace(M())
        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier._check_graph_module(gm)

    def test_verifier_no_functional(self) -> None:
        if False:
            i = 10
            return i + 15

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            return x + y
        ep = export(f, (torch.randn(100), torch.randn(100)))
        for node in ep.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor
        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

    def test_verifier_higher_order(self) -> None:
        if False:
            i = 10
            return i + 15

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if False:
                while True:
                    i = 10

            def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return x + y

            def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return x - y
            return control_flow.cond(x.shape[0] > 2, true_fn, false_fn, [x, y])
        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)))
        verifier = Verifier()
        verifier.check(ep)

    def test_verifier_nested_invalid_module(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            if False:
                print('Hello World!')

            def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                return x + y

            def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    return 10
                return x - y
            return control_flow.cond(x.shape[0] > 2, true_fn, false_fn, [x, y])
        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)))
        for node in ep.graph_module.true_graph_0.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor
        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

    def test_ep_verifier_basic(self) -> None:
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self) -> None:
                if False:
                    return 10
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x: Tensor) -> Tensor:
                if False:
                    i = 10
                    return i + 15
                return self.linear(x)
        ep = export(M(), (torch.randn(10, 10),))
        ep._validate()

    def test_ep_verifier_invalid_param(self) -> None:
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self) -> None:
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.register_parameter(name='a', param=torch.nn.Parameter(torch.randn(100)))

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return x + y + self.a
        ep = export(M(), (torch.randn(100), torch.randn(100)))
        ep.graph_signature.input_specs[0] = InputSpec(kind=InputKind.PARAMETER, arg=TensorArgument(name='arg0_1'), target='bad_param')
        with self.assertRaisesRegex(SpecViolationError, 'not in the state dict'):
            ep._validate()
        ep.state_dict['bad_param'] = torch.randn(100)
        with self.assertRaisesRegex(SpecViolationError, 'not an instance of torch.nn.Parameter'):
            ep._validate()

    def test_ep_verifier_invalid_buffer(self) -> None:
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = torch.tensor(3.0)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return x + y + self.a
        ep = export(M(), (torch.randn(100), torch.randn(100)))
        ep.graph_signature.input_specs[0] = InputSpec(kind=InputKind.BUFFER, arg=TensorArgument(name='arg0_1'), target='bad_buffer')
        with self.assertRaisesRegex(SpecViolationError, 'not in the state dict'):
            ep._validate()

    def test_ep_verifier_buffer_mutate(self) -> None:
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))
                self.register_buffer('my_buffer1', torch.tensor(3.0))
                self.register_buffer('my_buffer2', torch.tensor(4.0))

            def forward(self, x1, x2):
                if False:
                    print('Hello World!')
                output = (x1 + self.my_parameter) * self.my_buffer1 + x2 * self.my_buffer2
                self.my_buffer2.add_(1.0)
                return output
        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)))
        ep._validate()

    def test_ep_verifier_invalid_output(self) -> None:
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))
                self.register_buffer('my_buffer1', torch.tensor(3.0))
                self.register_buffer('my_buffer2', torch.tensor(4.0))

            def forward(self, x1, x2):
                if False:
                    for i in range(10):
                        print('nop')
                output = (x1 + self.my_parameter) * self.my_buffer1 + x2 * self.my_buffer2
                self.my_buffer2.add_(1.0)
                return output
        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)))
        output_node = list(ep.graph.nodes)[-1]
        output_node.args = ((output_node.args[0][0], list(ep.graph.nodes)[0], output_node.args[0][1]),)
        with self.assertRaisesRegex(SpecViolationError, 'Number of output nodes'):
            ep._validate()
if __name__ == '__main__':
    run_tests()