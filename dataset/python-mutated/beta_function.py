"""
Referenced papar <Implicit Quantile Networks for Distributional Reinforcement Learning>
"""
import torch
from typing import Union
beta_function_map = {}
beta_function_map['uniform'] = lambda x: x

def cpw(x: Union[torch.Tensor, float], eta: float=0.71) -> Union[torch.Tensor, float]:
    if False:
        for i in range(10):
            print('nop')
    return x ** eta / (x ** eta + (1 - x) ** eta) ** (1 / eta)
beta_function_map['CPW'] = cpw

def CVaR(x: Union[torch.Tensor, float], eta: float=0.71) -> Union[torch.Tensor, float]:
    if False:
        i = 10
        return i + 15
    assert eta <= 1.0
    return x * eta
beta_function_map['CVaR'] = CVaR

def Pow(x: Union[torch.Tensor, float], eta: float=0.0) -> Union[torch.Tensor, float]:
    if False:
        i = 10
        return i + 15
    if eta >= 0:
        return x ** (1 / (1 + eta))
    else:
        return 1 - (1 - x) ** (1 / 1 - eta)
beta_function_map['Pow'] = Pow