from __future__ import annotations
from typing import Any
from categorical import CategoricalComparator
from dedupe._typing import PredicateFunction, VariableDefinition
from dedupe.variables.base import DerivedType
from dedupe.variables.categorical_type import CategoricalType

class ExistsType(CategoricalType):
    type = 'Exists'
    _predicate_functions: list[PredicateFunction] = []

    def __init__(self, definition: VariableDefinition):
        if False:
            while True:
                i = 10
        super(CategoricalType, self).__init__(definition)
        self.cat_comparator = CategoricalComparator([0, 1])
        self.higher_vars = []
        for higher_var in self.cat_comparator.dummy_names:
            dummy_var = DerivedType({'name': higher_var, 'type': 'Dummy', 'has missing': self.has_missing})
            self.higher_vars.append(dummy_var)

    def comparator(self, field_1: Any, field_2: Any) -> list[int]:
        if False:
            print('Hello World!')
        if field_1 and field_2:
            return self.cat_comparator(1, 1)
        elif field_1 or field_2:
            return self.cat_comparator(0, 1)
        else:
            return self.cat_comparator(0, 0)
    comparator.missing = True