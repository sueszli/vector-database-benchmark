import copy
import typing as t
import docspec
from pydoc_markdown.interfaces import Context
from pydoc_markdown.contrib.loaders.python import PythonLoader

class CustomPythonLoader(PythonLoader):

    def load(self) -> t.Iterable[docspec.Module]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Load the modules, but include inherited methods in the classes.\n        '
        temp_loader = PythonLoader(search_path=['../../../haystack'])
        temp_loader.init(Context(directory='.'))
        all_modules = list(temp_loader.load())
        classes = {}
        for module in all_modules:
            for member in module.members:
                if isinstance(member, docspec.Class):
                    classes[member.name] = member
        modules = super().load()
        modules = self.include_inherited_methods(modules, classes)
        return modules

    def include_inherited_methods(self, modules: t.Iterable[docspec.Module], classes: t.Dict[str, docspec.Class]) -> t.Iterable[docspec.Module]:
        if False:
            return 10
        '\n        Recursively include inherited methods from the base classes.\n        '
        modules = list(modules)
        for module in modules:
            for cls in module.members:
                if isinstance(cls, docspec.Class):
                    self.include_methods_for_class(cls, classes)
        return modules

    def include_methods_for_class(self, cls: docspec.Class, classes: t.Dict[str, docspec.Class]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Include all methods inherited from base classes to the class.\n        '
        if cls.bases is None:
            return
        for base in cls.bases:
            if base in classes:
                base_cls = classes[base]
                self.include_methods_for_class(base_cls, classes)
                for member in base_cls.members:
                    if isinstance(member, docspec.Function) and (not any((m.name == member.name for m in cls.members))):
                        new_member = copy.deepcopy(member)
                        new_member.parent = cls
                        cls.members.append(new_member)