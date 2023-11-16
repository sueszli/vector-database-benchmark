"""Registry for calculations."""
from __future__ import annotations
import inspect
from extensions.answer_summarizers import models
from typing import Callable, Dict, Literal, overload

class Registry:
    """Registry of all calculations for summarizing answers."""
    _calculations_dict: Dict[str, Callable[..., models.BaseCalculation]] = {}

    @classmethod
    def _refresh_registry(cls) -> None:
        if False:
            return 10
        'Refreshes the registry to add new visualization instances.'
        cls._calculations_dict.clear()
        for (name, clazz) in inspect.getmembers(models, predicate=inspect.isclass):
            if name.endswith('_test') or name == 'BaseCalculation':
                continue
            ancestor_names = [base_class.__name__ for base_class in inspect.getmro(clazz)]
            if 'BaseCalculation' in ancestor_names:
                cls._calculations_dict[clazz.__name__] = clazz

    @overload
    @classmethod
    def get_calculation_by_id(cls, calculation_id: Literal['AnswerFrequencies']) -> models.AnswerFrequencies:
        if False:
            while True:
                i = 10
        ...

    @overload
    @classmethod
    def get_calculation_by_id(cls, calculation_id: Literal['Top5AnswerFrequencies']) -> models.Top5AnswerFrequencies:
        if False:
            print('Hello World!')
        ...

    @overload
    @classmethod
    def get_calculation_by_id(cls, calculation_id: Literal['Top10AnswerFrequencies']) -> models.Top10AnswerFrequencies:
        if False:
            while True:
                i = 10
        ...

    @overload
    @classmethod
    def get_calculation_by_id(cls, calculation_id: Literal['FrequencyCommonlySubmittedElements']) -> models.FrequencyCommonlySubmittedElements:
        if False:
            return 10
        ...

    @overload
    @classmethod
    def get_calculation_by_id(cls, calculation_id: Literal['TopAnswersByCategorization']) -> models.TopAnswersByCategorization:
        if False:
            while True:
                i = 10
        ...

    @overload
    @classmethod
    def get_calculation_by_id(cls, calculation_id: Literal['TopNUnresolvedAnswersByFrequency']) -> models.TopNUnresolvedAnswersByFrequency:
        if False:
            return 10
        ...

    @overload
    @classmethod
    def get_calculation_by_id(cls, calculation_id: str) -> models.BaseCalculation:
        if False:
            for i in range(10):
                print('nop')
        ...

    @classmethod
    def get_calculation_by_id(cls, calculation_id: str) -> models.BaseCalculation:
        if False:
            return 10
        'Gets a calculation instance by its id (which is also its class name).\n\n        Refreshes once if the class is not found; subsequently, throws an\n        error.\n        '
        if calculation_id not in cls._calculations_dict:
            cls._refresh_registry()
        if calculation_id not in cls._calculations_dict:
            raise TypeError("'%s' is not a valid calculation id." % calculation_id)
        return cls._calculations_dict[calculation_id]()