import paddle
from ..dy2static.program_translator import _program_hash, synchronized

class ParametersRecorder:

    def __init__(self):
        if False:
            return 10
        self.params_dict = {}
        self.tensor2opresult = {}

    @synchronized
    def get(self, program, tensor):
        if False:
            while True:
                i = 10
        from paddle.pir.core import create_parameter, vartype_to_datatype
        'use the default_program as key, append tensor the parameter list.'
        key = _program_hash(program)
        if key not in self.params_dict:
            self.params_dict[key] = set()
            self.tensor2opresult[key] = {}
        params = self.params_dict[key]
        mappings = self.tensor2opresult[key]
        if id(tensor) not in mappings:
            non_used_initializer = paddle.nn.initializer.Constant(0.0)
            op_result = create_parameter(dtype=vartype_to_datatype[tensor.dtype], shape=tensor.shape, type=tensor.type, initializer=non_used_initializer)
            if isinstance(tensor, paddle.Tensor):
                params.add(tensor)
            mappings[id(tensor)] = op_result
        return mappings[id(tensor)]

    def pop(self, program):
        if False:
            print('Hello World!')
        hash_id = _program_hash(program)
        params = self.params_dict.get(hash_id)
        if params is None:
            return ([], [])
        params_values = [self.tensor2opresult[hash_id][id(x)] for x in list(params)]
        del self.params_dict[hash_id]
        del self.tensor2opresult[hash_id]
        return (list(params), list(params_values))

class InplaceMap:

    def __init__(self):
        if False:
            print('Hello World!')
        self.params_dict = {}

    @synchronized
    def add(self, program, id, param):
        if False:
            i = 10
            return i + 15
        'use the default_program as key, append param the parameter list.'
        key = _program_hash(program)
        if key not in self.params_dict:
            self.params_dict[key] = {}
        params = self.params_dict[key]
        params[id] = param

    def get(self, program, id):
        if False:
            while True:
                i = 10
        params = self.params_dict.get(_program_hash(program))
        if params is None:
            return None
        if id not in params:
            return None
        root_var = params[id]
        saved = []
        while id(root_var) in params.keys():
            saved.append(root_var)
            root_var = params[id(root_var)]
        for var in saved:
            params[id(var)] = root_var
        return root_var

    def restore_checkpoint(self, checkpoint):
        if False:
            for i in range(10):
                print('nop')
        self.params_dict = checkpoint

    def save_checkpoint(self):
        if False:
            for i in range(10):
                print('nop')
        return dict(self.params_dict.items())
_global_parameter_recorder = ParametersRecorder()
_global_inplace_map = InplaceMap()