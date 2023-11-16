import importlib
import sys

class LazyImport:
    """
    Lazy import python module until use.

    Example:
        >>> from bigdl.llm.utils.common import LazyImport
        >>> _convert_to_ggml = LazyImport('bigdl.llm.ggml.convert._convert_to_ggml')
        >>> _convert_to_ggml(model_path, outfile_dir)
    """

    def __init__(self, module_name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param module_name: Import module name.\n        '
        self.module_name = module_name

    def __getattr__(self, name):
        if False:
            return 10
        absolute_name = importlib.util.resolve_name(self.module_name)
        try:
            return getattr(sys.modules[absolute_name], name)
        except (KeyError, AttributeError):
            pass
        if '.' in absolute_name:
            (parent_name, _, child_name) = absolute_name.rpartition('.')
        else:
            (parent_name, child_name) = (absolute_name, None)
        try:
            module = importlib.import_module(parent_name)
            module = getattr(module, child_name) if child_name else module
        except AttributeError:
            full_module_name = parent_name + '.' + child_name if child_name else parent_name
            spec = importlib.util.find_spec(full_module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        function_name = self.module_name.rpartition('.')[-1]
        module_name = self.module_name.rpartition(f'.{function_name}')[0]
        try:
            module = sys.modules[module_name]
        except KeyError:
            pass
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        return function(*args, **kwargs)