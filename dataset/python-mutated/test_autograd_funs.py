import pytorch_test_common
import torch
from onnx_test_common import run_model_test
from torch.onnx import OperatorExportTypes
from torch.onnx._globals import GLOBALS
from torch.onnx.utils import _model_to_graph
from torch.testing._internal import common_utils

class TestAutogradFuns(pytorch_test_common.ExportTestCase):
    opset_version = GLOBALS.export_onnx_opset_version
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_single_output(self):
        if False:
            while True:
                i = 10

        class SingleOut(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                if False:
                    print('Hello World!')
                result = i.exp()
                result = result.log()
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    i = 10
                    return i + 15
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                result = input + 5
                return SingleOut.apply(result) + 3
        model = Caller()
        input = torch.ones(1)
        run_model_test(self, model, input_args=(input,))

    def test_multi_output(self):
        if False:
            for i in range(10):
                print('nop')

        class MultiOut(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                if False:
                    i = 10
                    return i + 15
                result_exp = i.exp()
                result_log = result_exp.log()
                ctx.save_for_backward(result_exp, result_log)
                return (result_exp, result_log)

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    while True:
                        i = 10
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return MultiOut.apply(input)
        model = Caller()
        input = torch.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    def test_partial_output(self):
        if False:
            print('Hello World!')

        class PartialOut(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    print('Hello World!')
                ctx.save_for_backward(input)
                (values, indices) = torch.topk(input, 3)
                return values

        class Caller(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return PartialOut.apply(input)
        model = Caller()
        input = torch.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    def test_nested_autograd(self):
        if False:
            return 10

        class Child(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                if False:
                    for i in range(10):
                        print('nop')
                result = i.log()
                result_log = result.log()
                ctx.save_for_backward(result_log)
                return result_log

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    print('Hello World!')
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Parent(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                if False:
                    i = 10
                    return i + 15
                result_exp = i.exp()
                result_log = Child.apply(result_exp)
                ctx.save_for_backward(result_exp, result_log)
                return (result_exp, result_log)

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    while True:
                        i = 10
                (result,) = ctx.saved_tensors
                return grad_output * result

        class Caller(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return Parent.apply(input)
        model = Caller()
        input = torch.ones(1, 5)
        run_model_test(self, model, input_args=(input,))

    def test_aten_unsupported(self):
        if False:
            print('Hello World!')

        class Erf(torch.autograd.Function):

            @staticmethod
            def forward(ctx, x):
                if False:
                    for i in range(10):
                        print('nop')
                erf_out = torch.special.erf(x)
                ctx.save_for_backward(erf_out)
                return erf_out

            @staticmethod
            def backward(ctx, grad_output):
                if False:
                    return 10
                result = ctx.saved_tensors
                return (torch.special.erfinv(result), None)

        class Caller(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return Erf.apply(input)
        model = Caller()
        input = torch.ones(1, 5)
        (graph, _, _) = _model_to_graph(model, (input,), operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH)
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), 'prim::PythonOp')
        (graph, _, _) = _model_to_graph(model, (input,), operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
        iter = graph.nodes()
        self.assertEqual(next(iter).kind(), 'aten::ATen')

    def test_inline_and_symbolic(self):
        if False:
            for i in range(10):
                print('nop')

        class Exp(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                if False:
                    while True:
                        i = 10
                ctx.save_for_backward(input)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                if False:
                    print('Hello World!')
                return g.op('Exp', input)

        class LogLog(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                if False:
                    print('Hello World!')
                ctx.save_for_backward(input)
                return i.log().log()

        class Caller(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)
        model = Caller()
        input = torch.ones(1)
        run_model_test(self, model, input_args=(input,))

    def test_inline_with_scoped_tracing(self):
        if False:
            print('Hello World!')

        class Exp(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                if False:
                    while True:
                        i = 10
                ctx.save_for_backward(input)
                return i.exp()

            @staticmethod
            def symbolic(g, input):
                if False:
                    i = 10
                    return i + 15
                return g.op('Exp', input)

        class LogLog(torch.autograd.Function):

            @staticmethod
            def forward(ctx, i):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.save_for_backward(input)
                return i.log().log()

        class Caller(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                exp_result = Exp.apply(input)
                return LogLog.apply(exp_result)
        model = Caller()
        input = torch.ones(1)
        torch.jit._trace._trace_module_map = {_m: torch.typename(type(_m)) for _m in model.modules()}
        run_model_test(self, model, input_args=(input,))
        torch.jit._trace._trace_module_map = None
if __name__ == '__main__':
    common_utils.run_tests()