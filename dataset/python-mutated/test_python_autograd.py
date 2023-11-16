from typing import Callable, Dict, List, NamedTuple, Optional
import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter, same
'\nThis is an example of a pure-python version of autograd implemented by\n@zdevito.  It represents a rather challenging test case for TorchDynamo\nto push the limits of what it can do.\n'
_name: int = 0

def fresh_name() -> str:
    if False:
        while True:
            i = 10
    'create a new unique name for a variable: v0, v1, v2'
    global _name
    r = f'v{_name}'
    _name += 1
    return r

class Variable:

    def __init__(self, value: torch.Tensor, name: str=None):
        if False:
            print('Hello World!')
        self.value = value
        self.name = name or fresh_name()

    @staticmethod
    def constant(value: torch.Tensor, name: str=None):
        if False:
            while True:
                i = 10
        return Variable(value, name)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return repr(self.value)

    def __mul__(self, rhs: 'Variable') -> 'Variable':
        if False:
            print('Hello World!')
        return operator_mul(self, rhs)

    def __add__(self, rhs: 'Variable') -> 'Variable':
        if False:
            while True:
                i = 10
        return operator_add(self, rhs)

    def sum(self, name: Optional[str]=None) -> 'Variable':
        if False:
            while True:
                i = 10
        return operator_sum(self, name)

    def expand(self, sizes: List[int]) -> 'Variable':
        if False:
            i = 10
            return i + 15
        return operator_expand(self, sizes)

class TapeEntry(NamedTuple):
    inputs: List[str]
    outputs: List[str]
    propagate: 'Callable[List[Variable], List[Variable]]'
gradient_tape: List[TapeEntry] = []

def reset_tape():
    if False:
        print('Hello World!')
    gradient_tape.clear()
    global _name
    _name = 0

def grad(L, desired_results: List[Variable]) -> List[Variable]:
    if False:
        print('Hello World!')
    dL_d: Dict[str, Variable] = {}
    dL_d[L.name] = Variable(torch.ones(()))

    def gather_grad(entries: List[str]):
        if False:
            while True:
                i = 10
        return [dL_d[entry] if entry in dL_d else None for entry in entries]
    for entry in reversed(gradient_tape):
        dL_doutputs = gather_grad(entry.outputs)
        if all((dL_doutput is None for dL_doutput in dL_doutputs)):
            continue
        dL_dinputs = entry.propagate(dL_doutputs)
        for (input, dL_dinput) in zip(entry.inputs, dL_dinputs):
            if input not in dL_d:
                dL_d[input] = dL_dinput
            else:
                dL_d[input].value += dL_dinput.value
    return gather_grad((desired.name for desired in desired_results))

def operator_mul(self: Variable, rhs: Variable) -> Variable:
    if False:
        while True:
            i = 10
    if isinstance(rhs, float) and rhs == 1.0:
        return self
    r = Variable(self.value * rhs.value)
    inputs = [self.name, rhs.name]
    outputs = [r.name]

    def propagate(dL_doutputs: List[Variable]):
        if False:
            while True:
                i = 10
        (dL_dr,) = dL_doutputs
        dr_dself = rhs
        dr_drhs = self
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        dL_dinputs = [dL_dself, dL_drhs]
        return dL_dinputs
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return r

def operator_add(self: Variable, rhs: Variable) -> Variable:
    if False:
        for i in range(10):
            print('nop')
    r = Variable(self.value + rhs.value)

    def propagate(dL_doutputs: List[Variable]):
        if False:
            while True:
                i = 10
        (dL_dr,) = dL_doutputs
        dr_dself = 1.0
        dr_drhs = 1.0
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        return [dL_dself, dL_drhs]
    gradient_tape.append(TapeEntry(inputs=[self.name, rhs.name], outputs=[r.name], propagate=propagate))
    return r

def operator_sum(self: Variable, name: Optional[str]) -> 'Variable':
    if False:
        print('Hello World!')
    r = Variable(torch.sum(self.value), name=name)

    def propagate(dL_doutputs: List[Variable]):
        if False:
            i = 10
            return i + 15
        (dL_dr,) = dL_doutputs
        size = self.value.size()
        return [dL_dr.expand(*size)]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate))
    return r

def operator_expand(self: Variable, sizes: List[int]) -> 'Variable':
    if False:
        print('Hello World!')
    assert self.value.dim() == 0
    r = Variable(self.value.expand(sizes))

    def propagate(dL_doutputs: List[Variable]):
        if False:
            for i in range(10):
                print('nop')
        (dL_dr,) = dL_doutputs
        return [dL_dr.sum()]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate))
    return r

def simple(a, b):
    if False:
        for i in range(10):
            print('nop')
    t = a + b
    return t * b

class TestPythonAutograd(TestCase):

    def _common(self, fn, expected_ops):
        if False:
            print('Hello World!')
        args1 = [torch.randn(10), torch.randn(10)]
        args2 = [torch.randn(10), torch.randn(10)]
        cnt = CompileCounter()
        fn_dynamo = torch._dynamo.optimize_assert(cnt)(fn)
        reset_tape()
        res1 = fn_dynamo(*args1)
        reset_tape()
        res2 = fn_dynamo(*args2)
        reset_tape()
        self.assertTrue(same(res1, fn(*args1)))
        reset_tape()
        self.assertTrue(same(res2, fn(*args2)))
        reset_tape()
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, expected_ops)

    def test_forwards1(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(a, b):
            if False:
                print('Hello World!')
            a = Variable.constant(a, name='a')
            b = Variable.constant(b, name='b')
            loss = simple(a, b).sum()
            return loss
        self._common(fn, 3)

    def test_forwards2(self):
        if False:
            return 10

        def fn(a, b):
            if False:
                i = 10
                return i + 15
            reset_tape()
            a = Variable.constant(a, name='a')
            b = Variable.constant(b, name='b')
            loss = simple(a, b).sum()
            reset_tape()
            return loss
        self._common(fn, 3)

    def test_backwards1(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(a, b):
            if False:
                while True:
                    i = 10
            a = Variable.constant(a, name='a')
            b = Variable.constant(b, name='b')
            loss = simple(a, b).sum()
            return grad(loss, [a, b])
        self._common(fn, 8)

    def test_backwards2(self):
        if False:
            print('Hello World!')

        def fn(a, b):
            if False:
                print('Hello World!')
            reset_tape()
            a = Variable.constant(a, name='a')
            b = Variable.constant(b, name='b')
            loss = simple(a, b).sum()
            res = grad(loss, [a, b])
            reset_tape()
            return res
        self._common(fn, 8)

    def test_split(self):
        if False:
            for i in range(10):
                print('nop')
        v1 = Variable.constant(torch.randn(10), name='a')
        v2 = Variable.constant(torch.randn(10), name='b')
        cnt = CompileCounter()

        def forward(a, b):
            if False:
                return 10
            return simple(a, b).sum()
        reset_tape()
        loss1 = forward(v1, v2)
        grad1 = grad(loss1, [v1, v2])
        reset_tape()
        opt_forward = torch._dynamo.optimize_assert(cnt)(forward)
        opt_grad = torch._dynamo.optimize_assert(cnt)(grad)
        loss2 = opt_forward(v1, v2)
        grad2 = opt_grad(loss2, [v1, v2])
        self.assertTrue(same(loss1, loss2))
        self.assertTrue(same(grad1, grad2))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 8)
if __name__ == '__main__':
    run_tests()