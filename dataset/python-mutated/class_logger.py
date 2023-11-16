"""
Module contains ``ClassLogger`` class.

``ClassLogger`` is used for adding logging to Modin classes and their subclasses.
"""
from typing import Dict, Optional
from .logger_decorator import enable_logging

class ClassLogger:
    """
    Ensure all subclasses of the class being inherited are logged, too.

    Notes
    -----
    This mixin must go first in class bases declaration to have the desired effect.
    """
    _modin_logging_layer = 'PANDAS-API'

    @classmethod
    def __init_subclass__(cls, modin_layer: Optional[str]=None, class_name: Optional[str]=None, log_level: str='info', **kwargs: Dict) -> None:
        if False:
            print('Hello World!')
        '\n        Apply logging decorator to all children of ``ClassLogger``.\n\n        Parameters\n        ----------\n        modin_layer : str, default: "PANDAS-API"\n            Specified by the logger (e.g. PANDAS-API).\n        class_name : str, optional\n            The name of the class the decorator is being applied to.\n            Composed from the decorated class name if not specified.\n        log_level : str, default: "info"\n            The log level (INFO, DEBUG, WARNING, etc.).\n        **kwargs : dict\n        '
        modin_layer = modin_layer or cls._modin_logging_layer
        super().__init_subclass__(**kwargs)
        enable_logging(modin_layer, class_name, log_level)(cls)
        cls._modin_logging_layer = modin_layer