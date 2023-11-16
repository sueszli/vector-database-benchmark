"""Registry for visualizations."""
from __future__ import annotations
import inspect
from extensions.visualizations import models
from typing import Dict, List, Type

class Registry:
    """Registry of all visualizations."""
    visualizations_dict: Dict[str, Type[models.BaseVisualization]] = {}

    @classmethod
    def _refresh_registry(cls) -> None:
        if False:
            return 10
        'Clears and adds new visualization instances to the registry.'
        cls.visualizations_dict.clear()
        for (name, clazz) in inspect.getmembers(models, predicate=inspect.isclass):
            if name.endswith('_test') or name == 'BaseVisualization':
                continue
            ancestor_names = [base_class.__name__ for base_class in inspect.getmro(clazz)]
            if 'BaseVisualization' in ancestor_names:
                cls.visualizations_dict[clazz.__name__] = clazz

    @classmethod
    def get_visualization_class(cls, visualization_id: str) -> Type[models.BaseVisualization]:
        if False:
            return 10
        "Gets a visualization class by its id (which is also its class name).\n\n        The registry will refresh if the desired class is not found. If it's\n        still not found after the refresh, this method will throw an error.\n        "
        if visualization_id not in cls.visualizations_dict:
            cls._refresh_registry()
        if visualization_id not in cls.visualizations_dict:
            raise TypeError("'%s' is not a valid visualization id." % visualization_id)
        return cls.visualizations_dict[visualization_id]

    @classmethod
    def get_all_visualization_ids(cls) -> List[str]:
        if False:
            while True:
                i = 10
        'Gets a visualization class by its id\n        (which is also its class name).\n        '
        if not cls.visualizations_dict:
            cls._refresh_registry()
        return list(cls.visualizations_dict.keys())