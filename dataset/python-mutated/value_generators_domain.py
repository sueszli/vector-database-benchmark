"""Classes relating to value generators."""
from __future__ import annotations
import copy
import importlib
import inspect
import os
from core import feconf
from core import utils
from typing import Any, Dict, Type

class BaseValueGenerator:
    """Base value generator class.

    A value generator is a class containing a function that takes in
    customization args and uses them to generate a value. The generated values
    are not typed, so if the caller wants strongly-typed values it would need
    to normalize the output of each generator.

    Each value generator should define a template file and an AngularJS
    directive. The names of these two files should be [ClassName].html and
    [ClassName].js respectively, where [ClassName] is the name of the value
    generator class.
    """

    @property
    def id(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns the Class name as a string, i.e "BaseValueGenerator".\n\n        Returns:\n            str. Class name i.e "BaseValueGenerator".\n        '
        return self.__class__.__name__

    @classmethod
    def get_html_template(cls) -> str:
        if False:
            print('Hello World!')
        'Returns the HTML template for the class.\n\n        Returns:\n            str. The HTML template corresponding to the class.\n        '
        return utils.get_file_contents(os.path.join(os.getcwd(), feconf.VALUE_GENERATORS_DIR, 'templates', '%s.component.html' % cls.__name__))

    def generate_value(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            while True:
                i = 10
        'Generates a new value, using the given customization args.\n\n        The first arg should be context_params.\n        '
        raise NotImplementedError('generate_value() method has not yet been implemented')

class Registry:
    """Maintains a registry of all the value generators.

    Attributes:
        value_generators_dict: dict(str : BaseValueGenerator). Dictionary
            mapping value generator class names to their classes.
    """
    value_generators_dict: Dict[str, Type[BaseValueGenerator]] = {}

    @classmethod
    def _refresh_registry(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Refreshes the dictionary mapping between generator_id and the\n        corresponding generator classes.\n        '
        cls.value_generators_dict.clear()
        module_path_parts = feconf.VALUE_GENERATORS_DIR.split(os.sep)
        module_path_parts.extend(['models', 'generators'])
        module = importlib.import_module('.'.join(module_path_parts))
        for (_, clazz) in inspect.getmembers(module, predicate=inspect.isclass):
            if issubclass(clazz, BaseValueGenerator):
                cls.value_generators_dict[clazz.__name__] = clazz

    @classmethod
    def get_all_generator_classes(cls) -> Dict[str, Type[BaseValueGenerator]]:
        if False:
            return 10
        'Get the dict of all value generator classes.'
        cls._refresh_registry()
        return copy.deepcopy(cls.value_generators_dict)

    @classmethod
    def get_generator_class_by_id(cls, generator_id: str) -> Type[BaseValueGenerator]:
        if False:
            return 10
        'Gets a generator class by its id.\n\n        Refreshes once if the generator is not found; subsequently, throws an\n        error.\n\n        Args:\n            generator_id: str. An id corresponding to a generator class.\n\n        Returns:\n            class(BaseValueGenerator). A generator class mapping to the\n            generator id given.\n\n        Raises:\n            KeyError. The given generator_id is invalid.\n        '
        if generator_id not in cls.value_generators_dict:
            cls._refresh_registry()
        return cls.value_generators_dict[generator_id]