import contextlib
import pickle
import sys
import types
import typing
from copy import deepcopy
from pathlib import Path
from lightning.app.core.work import LightningWork
from lightning.app.utilities.app_helpers import _LightningAppRef
NON_PICKLABLE_WORK_ATTRIBUTES = ['_request_queue', '_response_queue', '_backend', '_setattr_replacement']

@contextlib.contextmanager
def _trimmed_work(work: LightningWork, to_trim: typing.List[str]) -> typing.Iterator[None]:
    if False:
        return 10
    'Context manager to trim the work object to remove attributes that are not picklable.'
    holder = {}
    for arg in to_trim:
        holder[arg] = getattr(work, arg)
        setattr(work, arg, None)
    yield
    for arg in to_trim:
        setattr(work, arg, holder[arg])

def get_picklable_work(work: LightningWork) -> LightningWork:
    if False:
        print('Hello World!')
    'Pickling a LightningWork instance fails if done from the work process\n    itself. This function is safe to call from the work process within both MultiprocessRuntime\n    and Cloud.\n    Note: This function modifies the module information of the work object. Specifically, it injects\n    the relative module path into the __module__ attribute of the work object. If the object is not\n    importable from the CWD, then the pickle load will fail.\n\n    Example:\n        for a directory structure like below and the work class is defined in the app.py where\n        the app.py is the entrypoint for the app, it will inject `foo.bar.app` into the\n        __module__ attribute\n\n        └── foo\n            ├── __init__.py\n            └── bar\n                └── app.py\n    '
    app_ref = _LightningAppRef.get_current()
    if app_ref is None:
        raise RuntimeError('Cannot pickle LightningWork outside of a LightningApp')
    for w in app_ref.works:
        if work.name == w.name:
            with _trimmed_work(w, to_trim=NON_PICKLABLE_WORK_ATTRIBUTES):
                copied_work = deepcopy(w)
            break
    else:
        raise ValueError(f'Work with name {work.name} not found in the app references')
    if '_main__' in copied_work.__class__.__module__:
        work_class_module = sys.modules[copied_work.__class__.__module__]
        work_class_file = work_class_module.__file__
        if not work_class_file:
            raise ValueError(f"Cannot pickle work class {copied_work.__class__.__name__} because we couldn't identify the module file")
        relative_path = Path(work_class_module.__file__).relative_to(Path.cwd())
        expected_module_name = relative_path.as_posix().replace('.py', '').replace('/', '.')
        fake_module = types.ModuleType(expected_module_name)
        fake_module.__dict__.update(work_class_module.__dict__)
        fake_module.__dict__['__name__'] = expected_module_name
        sys.modules[expected_module_name] = fake_module
        for (k, v) in fake_module.__dict__.items():
            if not k.startswith('__') and hasattr(v, '__module__') and ('_main__' in v.__module__):
                v.__module__ = expected_module_name
    return copied_work

def dump(work: LightningWork, f: typing.BinaryIO) -> None:
    if False:
        while True:
            i = 10
    picklable_work = get_picklable_work(work)
    pickle.dump(picklable_work, f)

def load(f: typing.BinaryIO) -> typing.Any:
    if False:
        i = 10
        return i + 15
    sys.path.insert(1, str(Path.cwd()))
    work = pickle.load(f)
    sys.path.pop(1)
    return work