from _torch import float32, int32, mm, Tensor
from typing_extensions import Literal
D1 = Literal[42]
D2 = Literal[16]
D3 = Literal[64]

class T:
    pass
T1: Tensor[int32, [D1, D2]] = Tensor()
T2: Tensor[int32, [D2, D3]] = Tensor()
T3: Tensor[int32, [T, T]] = Tensor()
T4: Tensor[float32, [D1, D2]] = Tensor()
T1p1: Tensor[int32, [D1, D2]] = T1 + T1
T1m2: Tensor[int32, [D1, D3]] = mm(T1, T2)

def incorrects() -> None:
    if False:
        return 10
    Err1 = T1 + T2
    Err2 = mm(T1, T3)
    Err3 = T1 + T3
Tx: Tensor[int32, [D1, D1, D1, D1, D2]] = Tensor()
Txpx: Tensor[int32, [D1, D1, D1, D1, D2]] = Tx + Tx