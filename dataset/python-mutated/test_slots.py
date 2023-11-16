import importlib
import inspect
import os
from pathlib import Path
included = {'CallbackContext'}

def test_class_has_slots_and_no_dict():
    if False:
        for i in range(10):
            print('nop')
    tg_paths = Path('telegram').rglob('*.py')
    for path in tg_paths:
        if '__' in str(path):
            continue
        mod_name = str(path)[:-3].replace(os.sep, '.')
        module = importlib.import_module(mod_name)
        for (name, cls) in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__ or any((x in name for x in ('__class__', '__init__', 'Queue', 'Webhook'))):
                continue
            assert '__slots__' in cls.__dict__, f"class '{name}' in {path} doesn't have __slots__"
            assert not isinstance(cls.__slots__, str), f"{name!r}s slots shouldn't be strings"
            if any((i in included for i in (cls.__module__, name, cls.__base__.__name__))):
                assert '__dict__' in get_slots(cls), f'class {name!r} ({path}) has no __dict__'
                continue
            assert '__dict__' not in get_slots(cls), f"class '{name}' in {path} has __dict__"

def get_slots(_class):
    if False:
        print('Hello World!')
    return [attr for cls in _class.__mro__ if hasattr(cls, '__slots__') for attr in cls.__slots__]