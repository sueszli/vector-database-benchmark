from __future__ import annotations
from categorical import CategoricalComparator
from dedupe import predicates
from dedupe._typing import PredicateFunction, VariableDefinition
from dedupe.variables.base import DerivedType, FieldType

class CategoricalType(FieldType):
    type = 'Categorical'
    _predicate_functions: list[PredicateFunction] = [predicates.wholeFieldPredicate]

    def _categories(self, definition: VariableDefinition) -> list[str]:
        if False:
            print('Hello World!')
        try:
            categories = definition['categories']
        except KeyError:
            raise ValueError('No "categories" defined')
        return categories

    def __init__(self, definition: VariableDefinition):
        if False:
            while True:
                i = 10
        super(CategoricalType, self).__init__(definition)
        categories = self._categories(definition)
        self.comparator = CategoricalComparator(categories)
        self.higher_vars = []
        for higher_var in self.comparator.dummy_names:
            dummy_var = DerivedType({'name': higher_var, 'type': 'Dummy', 'has missing': self.has_missing})
            self.higher_vars.append(dummy_var)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return len(self.higher_vars)