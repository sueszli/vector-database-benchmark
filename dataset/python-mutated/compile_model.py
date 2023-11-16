import torch
from torch.export import Dim
AOTI_CUSTOM_OP_LIB = 'libaoti_custom_class.so'
torch.classes.load_library(AOTI_CUSTOM_OP_LIB)

class TensorSerializer(torch.nn.Module):

    def __init__(self, data):
        if False:
            i = 10
            return i + 15
        super().__init__()
        for key in data:
            setattr(self, key, data[key])

class SimpleModule(torch.nn.Module):
    """
    a simple module to be compiled
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.fc = torch.nn.Linear(4, 6)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if False:
            while True:
                i = 10
        a = self.fc(x)
        b = self.relu(a)
        return b

class MyAOTIModule(torch.nn.Module):
    """
    a wrapper nn.Module that instantiates its forward method
    on MyAOTIClass
    """

    def __init__(self, lib_path, device):
        if False:
            print('Hello World!')
        super().__init__()
        self.aoti_custom_op = torch.classes.aoti.MyAOTIClass(lib_path, device)

    def forward(self, *x):
        if False:
            while True:
                i = 10
        outputs = self.aoti_custom_op.forward(x)
        return tuple(outputs)

def make_script_module(lib_path, device, *inputs):
    if False:
        for i in range(10):
            print('nop')
    m = MyAOTIModule(lib_path, device)
    m(*inputs)
    return torch.jit.trace(m, inputs)

def compile_model(device, data):
    if False:
        while True:
            i = 10
    module = SimpleModule().to(device)
    x = torch.randn((4, 4), device=device)
    inputs = (x,)
    batch_dim = Dim('batch', min=1, max=1024)
    dynamic_shapes = {'x': {0: batch_dim}}
    with torch.no_grad():
        lib_path = torch._export.aot_compile(module, inputs, dynamic_shapes=dynamic_shapes)
    script_module = make_script_module(lib_path, device, *inputs)
    aoti_script_model = f'script_model_{device}.pt'
    script_module.save(aoti_script_model)
    with torch.no_grad():
        ref_output = module(*inputs)
    data.update({f'inputs_{device}': list(inputs), f'outputs_{device}': [ref_output]})

def main():
    if False:
        return 10
    data = {}
    for device in ['cuda', 'cpu']:
        compile_model(device, data)
    torch.jit.script(TensorSerializer(data)).save('script_data.pt')
if __name__ == '__main__':
    main()