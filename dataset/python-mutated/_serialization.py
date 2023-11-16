"""Serialization.

This module contains functionality for serializing TorchScript modules, notably:
    * torch.jit.save
    * torch.jit.load

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
import os
import pathlib
import torch
from torch.jit._recursive import wrap_cpp_module
from torch.serialization import validate_cuda_device

def save(m, f, _extra_files=None):
    if False:
        i = 10
        return i + 15
    '\n    Save an offline version of this module for use in a separate process.\n\n    The saved module serializes all of the methods, submodules, parameters, and\n    attributes of this module. It can be loaded into the C++ API using\n    ``torch::jit::load(filename)`` or into the Python API with\n    :func:`torch.jit.load <torch.jit.load>`.\n\n    To be able to save a module, it must not make any calls to native Python\n    functions.  This means that all submodules must be subclasses of\n    :class:`ScriptModule` as well.\n\n    .. DANGER::\n        All modules, no matter their device, are always loaded onto the CPU\n        during loading.  This is different from :func:`torch.load`\'s semantics\n        and may change in the future.\n\n    Args:\n        m: A :class:`ScriptModule` to save.\n        f: A file-like object (has to implement write and flush) or a string\n           containing a file name.\n        _extra_files: Map from filename to contents which will be stored as part of `f`.\n\n    .. note::\n        torch.jit.save attempts to preserve the behavior of some operators\n        across versions. For example, dividing two integer tensors in\n        PyTorch 1.5 performed floor division, and if the module\n        containing that code is saved in PyTorch 1.5 and loaded in PyTorch 1.6\n        its division behavior will be preserved. The same module saved in\n        PyTorch 1.6 will fail to load in PyTorch 1.5, however, since the\n        behavior of division changed in 1.6, and 1.5 does not know how to\n        replicate the 1.6 behavior.\n\n    Example:\n    .. testcode::\n\n        import torch\n        import io\n\n        class MyModule(torch.nn.Module):\n            def forward(self, x):\n                return x + 10\n\n        m = torch.jit.script(MyModule())\n\n        # Save to file\n        torch.jit.save(m, \'scriptmodule.pt\')\n        # This line is equivalent to the previous\n        m.save("scriptmodule.pt")\n\n        # Save to io.BytesIO buffer\n        buffer = io.BytesIO()\n        torch.jit.save(m, buffer)\n\n        # Save with extra files\n        extra_files = {\'foo.txt\': b\'bar\'}\n        torch.jit.save(m, \'scriptmodule.pt\', _extra_files=extra_files)\n    '
    if _extra_files is None:
        _extra_files = {}
    if isinstance(f, (str, pathlib.Path)):
        m.save(f, _extra_files=_extra_files)
    else:
        ret = m.save_to_buffer(_extra_files=_extra_files)
        f.write(ret)

def load(f, map_location=None, _extra_files=None, _restore_shapes=False):
    if False:
        i = 10
        return i + 15
    '\n    Load a :class:`ScriptModule` or :class:`ScriptFunction` previously saved with :func:`torch.jit.save <torch.jit.save>`.\n\n    All previously saved modules, no matter their device, are first loaded onto CPU,\n    and then are moved to the devices they were saved from. If this fails (e.g.\n    because the run time system doesn\'t have certain devices), an exception is\n    raised.\n\n    Args:\n        f: a file-like object (has to implement read, readline, tell, and seek),\n            or a string containing a file name\n        map_location (string or torch.device): A simplified version of\n            ``map_location`` in `torch.jit.save` used to dynamically remap\n            storages to an alternative set of devices.\n        _extra_files (dictionary of filename to content): The extra\n            filenames given in the map would be loaded and their content\n            would be stored in the provided map.\n        _restore_shapes (bool): Whether or not to retrace the module on load using stored inputs\n\n    Returns:\n        A :class:`ScriptModule` object.\n\n    Example:\n    .. testcode::\n\n        import torch\n        import io\n\n        torch.jit.load(\'scriptmodule.pt\')\n\n        # Load ScriptModule from io.BytesIO object\n        with open(\'scriptmodule.pt\', \'rb\') as f:\n            buffer = io.BytesIO(f.read())\n\n        # Load all tensors to the original device\n        torch.jit.load(buffer)\n\n        # Load all tensors onto CPU, using a device\n        buffer.seek(0)\n        torch.jit.load(buffer, map_location=torch.device(\'cpu\'))\n\n        # Load all tensors onto CPU, using a string\n        buffer.seek(0)\n        torch.jit.load(buffer, map_location=\'cpu\')\n\n        # Load with extra files.\n        extra_files = {\'foo.txt\': \'\'}  # values will be replaced with data\n        torch.jit.load(\'scriptmodule.pt\', _extra_files=extra_files)\n        print(extra_files[\'foo.txt\'])\n\n    .. testoutput::\n        :hide:\n\n        ...\n\n    .. testcleanup::\n\n        import os\n        os.remove("scriptmodule.pt")\n    '
    if isinstance(f, str):
        if not os.path.exists(f):
            raise ValueError(f'The provided filename {f} does not exist')
        if os.path.isdir(f):
            raise ValueError(f'The provided filename {f} is a directory')
    map_location = validate_map_location(map_location)
    if _extra_files is None:
        _extra_files = {}
    cu = torch._C.CompilationUnit()
    if isinstance(f, (str, pathlib.Path)):
        cpp_module = torch._C.import_ir_module(cu, str(f), map_location, _extra_files, _restore_shapes)
    else:
        cpp_module = torch._C.import_ir_module_from_buffer(cu, f.read(), map_location, _extra_files, _restore_shapes)
    return wrap_cpp_module(cpp_module)

def validate_map_location(map_location=None):
    if False:
        print('Hello World!')
    if isinstance(map_location, str):
        map_location = torch.device(map_location)
    elif not (map_location is None or isinstance(map_location, torch.device)):
        raise ValueError('map_location should be either None, string or torch.device, but got type: ' + str(type(map_location)))
    if str(map_location).startswith('cuda'):
        validate_cuda_device(map_location)
    return map_location

def jit_module_from_flatbuffer(f):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(f, (str, pathlib.Path)):
        f = str(f)
        return wrap_cpp_module(torch._C._load_jit_module_from_file(f))
    else:
        return wrap_cpp_module(torch._C._load_jit_module_from_bytes(f.read()))

def save_jit_module_to_flatbuffer(m, f, _extra_files=None):
    if False:
        i = 10
        return i + 15
    "\n    Save an offline version of this module for use in a separate process.\n\n    The saved module serializes all of the methods, submodules, parameters, and\n    attributes of this module. It can be loaded into the C++ API using\n    ``torch::jit::load_jit_module_from_file(filename)`` or into the Python API with\n    :func:`torch.jit.jit_module_from_flatbuffer<torch.jit.jit_module_from_flatbuffer>`.\n\n    To be able to save a module, it must not make any calls to native Python\n    functions.  This means that all submodules must be subclasses of\n    :class:`ScriptModule` as well.\n\n    .. DANGER::\n        All modules, no matter their device, are always loaded onto the CPU\n        during loading.  This is different from :func:`torch.load`'s semantics\n        and may change in the future.\n\n    Args:\n        m: A :class:`ScriptModule` to save.\n        f: A string for file path\n\n\n    Example:\n    .. testcode::\n\n        import torch\n        import io\n\n        class MyModule(torch.nn.Module):\n            def forward(self, x):\n                return x + 10\n\n        m = torch.jit.script(MyModule())\n\n        # Save to file\n        torch.jit.save_jit_module_to_flatbuffer(m, 'scriptmodule.ff')\n    "
    extra_files = _extra_files
    if extra_files is None:
        extra_files = {}
    if isinstance(f, (str, pathlib.Path)):
        f = str(f)
        torch._C._save_jit_module(m._c, f, extra_files)
    else:
        s = torch._C._save_jit_module_to_bytes(m._c, extra_files)
        f.write(s)

def get_flatbuffer_module_info(path_or_file):
    if False:
        return 10
    "Get some information regarding a model file in flatbuffer format.\n\n    Args:\n        path_or_file: Either str, Path or file like object (BytesIO OK).\n            If it's str or Path, we will read the file referenced by that\n            path as Bytes.\n\n    Returns:\n        A dict with metadata on what that file contains, currently looks like\n        this:\n        {\n            'bytecode_version': 4,  # int\n            'operator_version': 4,  # int\n            'function_names': {\n                '__torch__.___torch_mangle_0.Foo.forward'}, # set\n            'type_names': set(),  # set\n            'opname_to_num_args': {'aten::linear': 3} # Dict[str, int]\n        }\n    "
    if isinstance(path_or_file, (str, pathlib.Path)):
        with open(path_or_file, 'rb') as f:
            all_bytes = f.read()
    else:
        all_bytes = path_or_file.read()
    return torch._C._get_module_info_from_flatbuffer(all_bytes)