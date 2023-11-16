import torch
from typing import List, Tuple

class SubmoduleNoForwardInputs(torch.nn.Module):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        super().__init__()
        self.name = name

    def forward(self):
        if False:
            while True:
                i = 10
        assert self.name == 'inner_mod_name'

class ModuleNoForwardInputs(torch.nn.Module):

    def __init__(self, name: str, submodule_name: str):
        if False:
            print('Hello World!')
        super().__init__()
        self.name = name
        self.submodule = SubmoduleNoForwardInputs(submodule_name)

    def forward(self):
        if False:
            return 10
        self.submodule()

class SubmoduleForwardSingleInput(torch.nn.Module):

    def __init__(self, name):
        if False:
            return 10
        super().__init__()
        self.name = name

    def foo(self, input: str):
        if False:
            while True:
                i = 10
        return input

    def forward(self, input: str):
        if False:
            for i in range(10):
                print('nop')
        input = input + '_inner_mod'
        input = self.foo(input)
        return input

class ModuleForwardSingleInput(torch.nn.Module):

    def __init__(self, name: str, submodule_name: str):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        if False:
            while True:
                i = 10
        input = input + '_outermod'
        return self.submodule(input)

class ModuleDirectforwardSubmodCall(torch.nn.Module):

    def __init__(self, name: str, submodule_name: str):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardSingleInput(submodule_name)

    def forward(self, input: str):
        if False:
            i = 10
            return i + 15
        input = input + '_outermod'
        return self.submodule.forward(input)

class SuboduleForwardMultipleInputs(torch.nn.Module):

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.name = name

    def forward(self, input1: List[str], input2: str):
        if False:
            while True:
                i = 10
        input1.append(self.name)
        output2 = input2 + '_'
        return (input1, output2)

class ModuleForwardMultipleInputs(torch.nn.Module):

    def __init__(self, name: str, submodule_name: str):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.name = name
        self.submodule = SuboduleForwardMultipleInputs(submodule_name)

    def forward(self, input1: List[str], input2: str):
        if False:
            print('Hello World!')
        input1.append(self.name)
        return self.submodule(input1, input2)

class SubmoduleForwardTupleInput(torch.nn.Module):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        super().__init__()
        self.name = name

    def forward(self, input: Tuple[int]):
        if False:
            print('Hello World!')
        input_access = input[0]
        return (1,)

class ModuleForwardTupleInput(torch.nn.Module):

    def __init__(self, name: str, submodule_name: str):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.name = name
        self.submodule = SubmoduleForwardTupleInput(submodule_name)

    def forward(self, input: Tuple[int]):
        if False:
            while True:
                i = 10
        input_access = input[0]
        return self.submodule((1,))

def create_module_no_forward_input():
    if False:
        while True:
            i = 10
    m = ModuleNoForwardInputs('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[()]) -> None:
        if False:
            print('Hello World!')
        assert self.name == 'outer_mod_name'

    def forward_hook(self, input: Tuple[()], output: None):
        if False:
            while True:
                i = 10
        assert self.name == 'outer_mod_name'
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)
    return m

def create_submodule_no_forward_input():
    if False:
        print('Hello World!')
    m = ModuleNoForwardInputs('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[()]) -> None:
        if False:
            print('Hello World!')
        assert self.name == 'inner_mod_name'

    def forward_hook(self, input: Tuple[()], output: None):
        if False:
            while True:
                i = 10
        assert self.name == 'inner_mod_name'
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    return m

def create_module_forward_multiple_inputs():
    if False:
        while True:
            i = 10
    m = ModuleForwardMultipleInputs('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if False:
            while True:
                i = 10
        assert self.name == 'outer_mod_name'
        assert input[0][0] == 'a'
        return (['pre_hook_override_name'], 'pre_hook_override')

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        if False:
            print('Hello World!')
        assert self.name == 'outer_mod_name'
        assert input[0][0] == 'pre_hook_override_name'
        output2 = output[1] + 'fh'
        return (output[0], output2)
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)
    return m

def create_module_multiple_hooks_multiple_inputs():
    if False:
        return 10
    m = ModuleForwardMultipleInputs('outer_mod_name', 'inner_mod_name')

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if False:
            i = 10
            return i + 15
        assert self.name == 'outer_mod_name'
        assert input[0][0] == 'a'
        return (['pre_hook_override_name'], 'pre_hook_override')

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if False:
            while True:
                i = 10
        assert self.name == 'outer_mod_name'
        assert input[0][0] == 'pre_hook_override_name'
        return (['pre_hook_override_name2'], 'pre_hook_override')

    def forward_hook1(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        if False:
            while True:
                i = 10
        assert self.name == 'outer_mod_name'
        assert input[0][0] == 'pre_hook_override_name2'
        output2 = output[1] + 'fh1'
        return (output[0], output2)

    def forward_hook2(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        if False:
            return 10
        assert self.name == 'outer_mod_name'
        assert input[0][0] == 'pre_hook_override_name2'
        assert output[1] == 'pre_hook_override_fh1'
        output2 = output[1] + '_fh2'
        return (output[0], output2)
    m.register_forward_pre_hook(pre_hook1)
    m.register_forward_pre_hook(pre_hook2)
    m.register_forward_hook(forward_hook1)
    m.register_forward_hook(forward_hook2)
    return m

def create_module_forward_single_input():
    if False:
        return 10
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        assert self.name == 'outer_mod_name'
        assert input[0] == 'a'
        return ('pre_hook_override_name',)

    def forward_hook(self, input: Tuple[str], output: str):
        if False:
            for i in range(10):
                print('nop')
        assert self.name == 'outer_mod_name'
        assert input == ('pre_hook_override_name',)
        output = output + '_fh'
        return output
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)
    return m

def create_module_same_hook_repeated():
    if False:
        return 10
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if False:
            while True:
                i = 10
        assert self.name == 'outer_mod_name'
        input_change = input[0] + '_ph'
        return (input_change,)

    def forward_hook(self, input: Tuple[str], output: str):
        if False:
            return 10
        assert self.name == 'outer_mod_name'
        assert input == ('a_ph_ph',)
        output = output + '_fh'
        return output
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)
    m.register_forward_hook(forward_hook)
    return m

def create_module_hook_return_nothing():
    if False:
        i = 10
        return i + 15
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[str]) -> None:
        if False:
            while True:
                i = 10
        assert self.name == 'outer_mod_name'
        assert input[0] == 'a'

    def forward_hook(self, input: Tuple[str], output: str):
        if False:
            for i in range(10):
                print('nop')
        assert self.name == 'outer_mod_name'
        assert input == ('a',)
    m.register_forward_pre_hook(pre_hook)
    m.register_forward_hook(forward_hook)
    return m

def create_module_multiple_hooks_single_input():
    if False:
        for i in range(10):
            print('nop')
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        assert self.name == 'outer_mod_name'
        assert input[0] == 'a'
        return ('pre_hook_override_name1',)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        if False:
            print('Hello World!')
        assert self.name == 'outer_mod_name'
        assert input[0] == 'pre_hook_override_name1'
        return ('pre_hook_override_name2',)

    def forward_hook1(self, input: Tuple[str], output: str):
        if False:
            print('Hello World!')
        assert self.name == 'outer_mod_name'
        assert input == ('pre_hook_override_name2',)
        assert output == 'pre_hook_override_name2_outermod_inner_mod'
        output = output + '_fh1'
        return (output, output)

    def forward_hook2(self, input: Tuple[str], output: Tuple[str, str]):
        if False:
            for i in range(10):
                print('nop')
        assert self.name == 'outer_mod_name'
        assert input == ('pre_hook_override_name2',)
        assert output[0] == 'pre_hook_override_name2_outermod_inner_mod_fh1'
        output = output[0] + '_fh2'
        return output
    m.register_forward_pre_hook(pre_hook1)
    m.register_forward_pre_hook(pre_hook2)
    m.register_forward_hook(forward_hook1)
    m.register_forward_hook(forward_hook2)
    return m

def create_submodule_forward_multiple_inputs():
    if False:
        return 10
    m = ModuleForwardMultipleInputs('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if False:
            i = 10
            return i + 15
        assert self.name == 'inner_mod_name'
        assert input[0][1] == 'outer_mod_name'
        return (['pre_hook_override_name'], 'pre_hook_override')

    def forward_hook(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        if False:
            for i in range(10):
                print('nop')
        assert self.name == 'inner_mod_name'
        assert input[0][0] == 'pre_hook_override_name'
        output2 = output[1] + 'fh'
        return (output[0], output2)
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    return m

def create_submodule_multiple_hooks_multiple_inputs():
    if False:
        while True:
            i = 10
    m = ModuleForwardMultipleInputs('outer_mod_name', 'inner_mod_name')

    def pre_hook1(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if False:
            while True:
                i = 10
        assert self.name == 'inner_mod_name'
        assert input[1] == 'no_pre_hook'
        return (['pre_hook_override_name'], 'pre_hook_override1')

    def pre_hook2(self, input: Tuple[List[str], str]) -> Tuple[List[str], str]:
        if False:
            print('Hello World!')
        assert self.name == 'inner_mod_name'
        assert input[1] == 'pre_hook_override1'
        return (['pre_hook_override_name'], 'pre_hook_override2')

    def forward_hook1(self, input: Tuple[List[str], str], output: Tuple[List[str], str]):
        if False:
            for i in range(10):
                print('nop')
        assert self.name == 'inner_mod_name'
        assert input[1] == 'pre_hook_override2'
        assert output[1] == 'pre_hook_override2_'
        output2 = output[1] + 'fh1'
        return (output[0], output2, output2)

    def forward_hook2(self, input: Tuple[List[str], str], output: Tuple[List[str], str, str]):
        if False:
            return 10
        assert self.name == 'inner_mod_name'
        assert input[1] == 'pre_hook_override2'
        assert output[1] == 'pre_hook_override2_fh1'
        output2 = output[1] + '_fh2'
        return (output[0], output2)
    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)
    return m

def create_submodule_forward_single_input():
    if False:
        i = 10
        return i + 15
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if False:
            return 10
        assert self.name == 'inner_mod_name'
        assert input[0] == 'a_outermod'
        return ('pre_hook_override_name',)

    def forward_hook(self, input: Tuple[str], output: str):
        if False:
            while True:
                i = 10
        assert self.name == 'inner_mod_name'
        assert input == ('pre_hook_override_name',)
        return output
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    return m

def create_submodule_to_call_directly_with_hooks():
    if False:
        print('Hello World!')
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if False:
            while True:
                i = 10
        assert self.name == 'inner_mod_name'
        return ('pre_hook_override_name',)

    def forward_hook(self, input: Tuple[str], output: str):
        if False:
            i = 10
            return i + 15
        assert self.name == 'inner_mod_name'
        assert input == ('pre_hook_override_name',)
        return output + '_fh'
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    return m

def create_submodule_same_hook_repeated():
    if False:
        while True:
            i = 10
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[str]) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        assert self.name == 'inner_mod_name'
        changed = input[0] + '_ph'
        return (changed,)

    def forward_hook(self, input: Tuple[str], output: str):
        if False:
            for i in range(10):
                print('nop')
        assert self.name == 'inner_mod_name'
        assert input == ('a_outermod_ph_ph',)
        return output + '_fh'
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    m.submodule.register_forward_hook(forward_hook)
    return m

def create_submodule_hook_return_nothing():
    if False:
        return 10
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[str]) -> None:
        if False:
            i = 10
            return i + 15
        assert self.name == 'inner_mod_name'
        assert input[0] == 'a_outermod'

    def forward_hook(self, input: Tuple[str], output: str):
        if False:
            while True:
                i = 10
        assert self.name == 'inner_mod_name'
        assert input == ('a_outermod',)
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    return m

def create_submodule_multiple_hooks_single_input():
    if False:
        return 10
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook1(self, input: Tuple[str]) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        assert self.name == 'inner_mod_name'
        assert input[0] == 'a_outermod'
        return ('pre_hook_override_name',)

    def pre_hook2(self, input: Tuple[str]) -> Tuple[str]:
        if False:
            i = 10
            return i + 15
        assert self.name == 'inner_mod_name'
        assert input[0] == 'pre_hook_override_name'
        return ('pre_hook_override_name2',)

    def forward_hook1(self, input: Tuple[str], output: str):
        if False:
            return 10
        assert self.name == 'inner_mod_name'
        assert input == ('pre_hook_override_name2',)
        assert output == 'pre_hook_override_name2_inner_mod'
        return output + '_fwh1'

    def forward_hook2(self, input: Tuple[str], output: str):
        if False:
            while True:
                i = 10
        assert self.name == 'inner_mod_name'
        assert input == ('pre_hook_override_name2',)
        assert output == 'pre_hook_override_name2_inner_mod_fwh1'
        return output
    m.submodule.register_forward_pre_hook(pre_hook1)
    m.submodule.register_forward_pre_hook(pre_hook2)
    m.submodule.register_forward_hook(forward_hook1)
    m.submodule.register_forward_hook(forward_hook2)
    return m

def create_forward_tuple_input():
    if False:
        print('Hello World!')
    m = ModuleForwardTupleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook_outermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        if False:
            print('Hello World!')
        return ((11,),)

    def pre_hook_innermod(self, input: Tuple[Tuple[int]]) -> Tuple[Tuple[int]]:
        if False:
            while True:
                i = 10
        return ((22,),)

    def forward_hook_outermod(self, input: Tuple[Tuple[int]], output: int):
        if False:
            print('Hello World!')
        return (11,)

    def forward_hook_innermod(self, input: Tuple[Tuple[int]], output: Tuple[int]):
        if False:
            return 10
        return 22
    m.register_forward_pre_hook(pre_hook_outermod)
    m.submodule.register_forward_pre_hook(pre_hook_innermod)
    m.register_forward_hook(forward_hook_outermod)
    m.submodule.register_forward_hook(forward_hook_innermod)
    return m

def create_submodule_forward_single_input_return_not_tupled():
    if False:
        for i in range(10):
            print('nop')
    m = ModuleForwardSingleInput('outer_mod_name', 'inner_mod_name')

    def pre_hook(self, input: Tuple[str]) -> str:
        if False:
            for i in range(10):
                print('nop')
        assert self.name == 'inner_mod_name'
        assert input[0] == 'a_outermod'
        return 'pre_hook_override_name'

    def forward_hook(self, input: Tuple[str], output: str):
        if False:
            i = 10
            return i + 15
        assert self.name == 'inner_mod_name'
        assert input == ('pre_hook_override_name',)
        output = output + '_fh'
        return output
    m.submodule.register_forward_pre_hook(pre_hook)
    m.submodule.register_forward_hook(forward_hook)
    return m