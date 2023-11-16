from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor

class AndroidAPIModule(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, input):
        if False:
            return 10
        return None

    @torch.jit.script_method
    def eqBool(self, input: bool) -> bool:
        if False:
            return 10
        return input

    @torch.jit.script_method
    def eqInt(self, input: int) -> int:
        if False:
            while True:
                i = 10
        return input

    @torch.jit.script_method
    def eqFloat(self, input: float) -> float:
        if False:
            return 10
        return input

    @torch.jit.script_method
    def eqStr(self, input: str) -> str:
        if False:
            return 10
        return input

    @torch.jit.script_method
    def eqTensor(self, input: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        return input

    @torch.jit.script_method
    def eqDictStrKeyIntValue(self, input: Dict[str, int]) -> Dict[str, int]:
        if False:
            while True:
                i = 10
        return input

    @torch.jit.script_method
    def eqDictIntKeyIntValue(self, input: Dict[int, int]) -> Dict[int, int]:
        if False:
            for i in range(10):
                print('nop')
        return input

    @torch.jit.script_method
    def eqDictFloatKeyIntValue(self, input: Dict[float, int]) -> Dict[float, int]:
        if False:
            for i in range(10):
                print('nop')
        return input

    @torch.jit.script_method
    def listIntSumReturnTuple(self, input: List[int]) -> Tuple[List[int], int]:
        if False:
            for i in range(10):
                print('nop')
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def listBoolConjunction(self, input: List[bool]) -> bool:
        if False:
            while True:
                i = 10
        res = True
        for x in input:
            res = res and x
        return res

    @torch.jit.script_method
    def listBoolDisjunction(self, input: List[bool]) -> bool:
        if False:
            print('Hello World!')
        res = False
        for x in input:
            res = res or x
        return res

    @torch.jit.script_method
    def tupleIntSumReturnTuple(self, input: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], int]:
        if False:
            while True:
                i = 10
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def optionalIntIsNone(self, input: Optional[int]) -> bool:
        if False:
            print('Hello World!')
        return input is None

    @torch.jit.script_method
    def intEq0None(self, input: int) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        if input == 0:
            return None
        return input

    @torch.jit.script_method
    def str3Concat(self, input: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return input + input + input

    @torch.jit.script_method
    def newEmptyShapeWithItem(self, input):
        if False:
            i = 10
            return i + 15
        return torch.tensor([int(input.item())])[0]

    @torch.jit.script_method
    def testAliasWithOffset(self) -> List[Tensor]:
        if False:
            i = 10
            return i + 15
        x = torch.tensor([100, 200])
        a = [x[0], x[1]]
        return a

    @torch.jit.script_method
    def testNonContiguous(self):
        if False:
            i = 10
            return i + 15
        x = torch.tensor([100, 200, 300])[::2]
        assert not x.is_contiguous()
        assert x[0] == 100
        assert x[1] == 300
        return x

    @torch.jit.script_method
    def conv2d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        if False:
            for i in range(10):
                print('nop')
        r = torch.nn.functional.conv2d(x, w)
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last)
        else:
            r = r.contiguous()
        return r

    @torch.jit.script_method
    def contiguous(self, x: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        return x.contiguous()

    @torch.jit.script_method
    def contiguousChannelsLast(self, x: Tensor) -> Tensor:
        if False:
            return 10
        return x.contiguous(memory_format=torch.channels_last)

    @torch.jit.script_method
    def contiguousChannelsLast3d(self, x: Tensor) -> Tensor:
        if False:
            while True:
                i = 10
        return x.contiguous(memory_format=torch.channels_last_3d)