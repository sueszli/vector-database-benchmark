import inspect
from typing import Any, Dict, Optional
import pycaret.utils.generic

class BaseContainer:
    """
    Base container class, for easier definition of containers. Ensures consistent format
    before being turned into a dataframe row.

    Parameters
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool, default = True
        Should the model be used with 'turbo = True' in compare_models().
    args : dict, default = {} (empty dict)
        The arguments to always pass to constructor when initializing object of class_def class.

    Attributes
    ----------
    id : str
        ID used as index.
    name : str
        Full display name.
    class_def : type
        The class used for the model, eg. LogisticRegression.
    is_turbo : bool, default = True
        Should the model be used with 'turbo = True' in compare_models().
    args : dict, default = {} (empty dict)
        The arguments to always pass to constructor when initializing object of class_def class.

    """

    def __init__(self, id: str, name: str, class_def: type, args: Optional[Dict[str, Any]]=None) -> None:
        if False:
            while True:
                i = 10
        if not args:
            args = {}
        if not isinstance(args, dict):
            raise TypeError('args needs to be a dictionary.')
        self.id = id
        self.name = name
        self.class_def = class_def
        self.reference = self.get_class_name()
        self.args = args
        self.active = True

    def get_class_name(self):
        if False:
            print('Hello World!')
        return pycaret.utils.generic.get_class_name(self.class_def)

    def get_package_name(self):
        if False:
            return 10
        return pycaret.utils.generic.get_package_name(self.class_def)

    def get_dict(self, internal: bool=True) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Returns a dictionary of the model properties, to\n        be turned into a pandas DataFrame row.\n\n        Parameters\n        ----------\n        internal : bool, default = True\n            If True, will return all properties. If False, will only\n            return properties intended for the user to see.\n\n        Returns\n        -------\n        dict of str : Any\n\n        '
        d = [('ID', self.id), ('Name', self.name), ('Reference', self.reference)]
        if internal:
            d += [('Class', self.class_def), ('Args', self.args)]
        return dict(d)

def get_all_containers(container_globals: dict, experiment: Any, type_var: type, raise_errors: bool=True) -> Dict[str, BaseContainer]:
    if False:
        for i in range(10):
            print('nop')
    model_container_classes = [obj for (_, obj) in container_globals.items() if inspect.isclass(obj) and type_var in tuple((x for x in inspect.getmro(obj) if x != obj))]
    model_containers = []
    for obj in model_container_classes:
        if raise_errors:
            if hasattr(obj, 'active') and (not obj.active):
                continue
            instance = obj(experiment)
            if instance.active:
                model_containers.append(instance)
        else:
            try:
                if hasattr(obj, 'active') and (not obj.active):
                    continue
                instance = obj(experiment)
                if instance.active:
                    model_containers.append(instance)
            except Exception:
                pass
    return {container.id: container for container in model_containers}