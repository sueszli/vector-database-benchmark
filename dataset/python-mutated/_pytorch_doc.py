from __future__ import annotations
from types import ModuleType
__all__ = ['PYTORCH_URL', 'pytorch_doc_link']
PYTORCH_URL = 'https://pytorch.org/docs/stable/'

def _mod2page(mod: ModuleType) -> str:
    if False:
        return 10
    'Get the webpage name for a PyTorch module'
    if mod == Tensor:
        return 'tensors.html'
    name = mod.__name__
    name = name.replace('torch.', '').replace('utils.', '')
    if name.startswith('nn.modules'):
        return 'nn.html'
    return f'{name}.html'
import importlib

def pytorch_doc_link(name: str) -> (str, None):
    if False:
        return 10
    'Get the URL to the documentation of a PyTorch module, class or function'
    if name.startswith('F'):
        name = 'torch.nn.functional' + name[1:]
    if not name.startswith('torch.'):
        name = 'torch.' + name
    if name == 'torch.Tensor':
        return f'{PYTORCH_URL}tensors.html'
    try:
        mod = importlib.import_module(name)
        return f'{PYTORCH_URL}{_mod2page(mod)}'
    except:
        pass
    splits = name.split('.')
    (mod_name, fname) = ('.'.join(splits[:-1]), splits[-1])
    if mod_name == 'torch.Tensor':
        return f'{PYTORCH_URL}tensors.html#{name}'
    try:
        mod = importlib.import_module(mod_name)
        page = _mod2page(mod)
        return f'{PYTORCH_URL}{page}#{name}'
    except:
        return None