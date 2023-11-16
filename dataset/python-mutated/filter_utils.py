import logging
from typing import Union, List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
from haystack.document_stores.utils import convert_date_to_rfc3339
from haystack.errors import FilterError
from haystack.lazy_imports import LazyImport
logger = logging.getLogger(__file__)
with LazyImport("Run 'pip install farm-haystack[sql]'") as sql_import:
    from sqlalchemy.sql import select
    from sqlalchemy import and_, or_

def nested_defaultdict() -> defaultdict:
    if False:
        while True:
            i = 10
    "\n    Data structure that recursively adds a dictionary as value if a key does not exist. Advantage: In nested dictionary\n    structures, we don't need to check if a key already exists (which can become hard to maintain in nested dictionaries\n    with many levels) but access the existing value if a key exists and create an empty dictionary if a key does not\n    exist.\n    "
    return defaultdict(nested_defaultdict)

class LogicalFilterClause(ABC):
    """
    Class that is able to parse a filter and convert it to the format that the underlying databases of our
    DocumentStores require.

    Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
    operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`, `"$gte"`, `"$lt"`,
    `"$lte"`) or a metadata field name.
    Logical operator keys take a dictionary of metadata field names and/or logical operators as
    value. Metadata field names take a dictionary of comparison operators as value. Comparison
    operator keys take a single value or (in case of `"$in"`) a list of values as value.
    If no logical operator is provided, `"$and"` is used as default operation. If no comparison
    operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
    operation.
    Example:
        ```python
        filters = {
            "$and": {
                "type": {"$eq": "article"},
                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                "rating": {"$gte": 3},
                "$or": {
                    "genre": {"$in": ["economy", "politics"]},
                    "publisher": {"$eq": "nytimes"}
                }
            }
        }
        # or simpler using default operators
        filters = {
            "type": "article",
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": ["economy", "politics"],
                "publisher": "nytimes"
            }
        }
        ```

    To use the same logical operator multiple times on the same level, logical operators take optionally a list of
    dictionaries as value.

    Example:
        ```python
        filters = {
            "$or": [
                {
                    "$and": {
                        "Type": "News Paper",
                        "Date": {
                            "$lt": "2019-01-01"
                        }
                    }
                },
                {
                    "$and": {
                        "Type": "Blog Post",
                        "Date": {
                            "$gte": "2019-01-01"
                        }
                    }
                }
            ]
        }
        ```

    """

    def __init__(self, conditions: List[Union['LogicalFilterClause', 'ComparisonOperation']]):
        if False:
            for i in range(10):
                print('nop')
        self.conditions = conditions

    @abstractmethod
    def evaluate(self, fields) -> bool:
        if False:
            while True:
                i = 10
        pass

    @classmethod
    def parse(cls, filter_term: Union[dict, List[dict]]) -> Union['LogicalFilterClause', 'ComparisonOperation']:
        if False:
            i = 10
            return i + 15
        '\n        Parses a filter dictionary/list and returns a LogicalFilterClause instance.\n\n        :param filter_term: Dictionary or list that contains the filter definition.\n        '
        conditions: List[Union[LogicalFilterClause, ComparisonOperation]] = []
        if isinstance(filter_term, dict):
            filter_term = [filter_term]
        for item in filter_term:
            for (key, value) in item.items():
                if key == '$not':
                    conditions.append(NotOperation.parse(value))
                elif key == '$and':
                    conditions.append(AndOperation.parse(value))
                elif key == '$or':
                    conditions.append(OrOperation.parse(value))
                else:
                    conditions.extend(ComparisonOperation.parse(key, value))
        if cls == LogicalFilterClause:
            if len(conditions) == 1:
                return conditions[0]
            else:
                return AndOperation(conditions)
        else:
            return cls(conditions)

    @abstractmethod
    def convert_to_elasticsearch(self):
        if False:
            return 10
        '\n        Converts the LogicalFilterClause instance to an Elasticsearch filter.\n        '
        pass

    @abstractmethod
    def convert_to_sql(self, meta_document_orm):
        if False:
            for i in range(10):
                print('nop')
        '\n        Converts the LogicalFilterClause instance to an SQL filter.\n        '
        pass

    def convert_to_weaviate(self):
        if False:
            print('Hello World!')
        '\n        Converts the LogicalFilterClause instance to a Weaviate filter.\n        '
        pass

    def convert_to_pinecone(self):
        if False:
            while True:
                i = 10
        '\n        Converts the LogicalFilterClause instance to a Pinecone filter.\n        '
        pass

    def _merge_es_range_queries(self, conditions: List[Dict]) -> List[Dict[str, Dict]]:
        if False:
            return 10
        '\n        Merges Elasticsearch range queries that perform on the same metadata field.\n        '
        range_conditions = [cond['range'] for cond in filter(lambda condition: 'range' in condition, conditions)]
        if range_conditions:
            conditions = [condition for condition in conditions if 'range' not in condition]
            range_conditions_dict = nested_defaultdict()
            for condition in range_conditions:
                field_name = list(condition.keys())[0]
                operation = list(condition[field_name].keys())[0]
                comparison_value = condition[field_name][operation]
                range_conditions_dict[field_name][operation] = comparison_value
            for (field_name, comparison_operations) in range_conditions_dict.items():
                conditions.append({'range': {field_name: comparison_operations}})
        return conditions

    @abstractmethod
    def invert(self) -> Union['LogicalFilterClause', 'ComparisonOperation']:
        if False:
            i = 10
            return i + 15
        "\n        Inverts the LogicalOperation instance.\n        Necessary for Weaviate as Weaviate doesn't seem to support the 'Not' operator anymore.\n        (https://github.com/semi-technologies/weaviate/issues/1717)\n        "
        pass

class ComparisonOperation(ABC):

    def __init__(self, field_name: str, comparison_value: Union[str, int, float, bool, List]):
        if False:
            return 10
        self.field_name = field_name
        self.comparison_value = comparison_value

    @abstractmethod
    def evaluate(self, fields) -> bool:
        if False:
            return 10
        pass

    @classmethod
    def parse(cls, field_name, comparison_clause: Union[Dict, List, str, float]) -> List['ComparisonOperation']:
        if False:
            return 10
        comparison_operations: List[ComparisonOperation] = []
        if isinstance(comparison_clause, dict):
            for (comparison_operation, comparison_value) in comparison_clause.items():
                if comparison_operation == '$eq':
                    comparison_operations.append(EqOperation(field_name, comparison_value))
                elif comparison_operation == '$in':
                    comparison_operations.append(InOperation(field_name, comparison_value))
                elif comparison_operation == '$ne':
                    comparison_operations.append(NeOperation(field_name, comparison_value))
                elif comparison_operation == '$nin':
                    comparison_operations.append(NinOperation(field_name, comparison_value))
                elif comparison_operation == '$gt':
                    comparison_operations.append(GtOperation(field_name, comparison_value))
                elif comparison_operation == '$gte':
                    comparison_operations.append(GteOperation(field_name, comparison_value))
                elif comparison_operation == '$lt':
                    comparison_operations.append(LtOperation(field_name, comparison_value))
                elif comparison_operation == '$lte':
                    comparison_operations.append(LteOperation(field_name, comparison_value))
        elif isinstance(comparison_clause, list):
            comparison_operations.append(InOperation(field_name, comparison_clause))
        else:
            comparison_operations.append(EqOperation(field_name, comparison_clause))
        return comparison_operations

    @abstractmethod
    def convert_to_elasticsearch(self):
        if False:
            i = 10
            return i + 15
        '\n        Converts the ComparisonOperation instance to an Elasticsearch query.\n        '
        pass

    @abstractmethod
    def convert_to_sql(self, meta_document_orm):
        if False:
            return 10
        '\n        Converts the ComparisonOperation instance to an SQL filter.\n        '
        pass

    @abstractmethod
    def convert_to_weaviate(self):
        if False:
            return 10
        '\n        Converts the ComparisonOperation instance to a Weaviate comparison operator.\n        '
        pass

    def convert_to_pinecone(self):
        if False:
            while True:
                i = 10
        '\n        Converts the ComparisonOperation instance to a Pinecone comparison operator.\n        '
        pass

    @abstractmethod
    def invert(self) -> 'ComparisonOperation':
        if False:
            i = 10
            return i + 15
        "\n        Inverts the ComparisonOperation.\n        Necessary for Weaviate as Weaviate doesn't seem to support the 'Not' operator anymore.\n        (https://github.com/semi-technologies/weaviate/issues/1717)\n        "
        pass

    def _get_weaviate_datatype(self, value: Optional[Union[str, int, float, bool]]=None) -> Tuple[str, Union[str, int, float, bool]]:
        if False:
            while True:
                i = 10
        '\n        Determines the type of the comparison value and converts it to RFC3339 format if it is as date,\n        as Weaviate requires dates to be in RFC3339 format including the time and timezone.\n\n        '
        if value is None:
            assert not isinstance(self.comparison_value, list)
            value = self.comparison_value
        if isinstance(value, str):
            try:
                value = convert_date_to_rfc3339(value)
                data_type = 'valueDate'
            except ValueError:
                if self.field_name == 'content':
                    data_type = 'valueText'
                else:
                    data_type = 'valueString'
        elif isinstance(value, int):
            data_type = 'valueInt'
        elif isinstance(value, float):
            data_type = 'valueNumber'
        elif isinstance(value, bool):
            data_type = 'valueBoolean'
        else:
            raise ValueError(f'Unsupported data type of comparison value for {self.__class__.__name__}.Value needs to be of type str, int, float, or bool.')
        return (data_type, value)

class NotOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'NOT' operations.
    """

    def evaluate(self, fields) -> bool:
        if False:
            while True:
                i = 10
        return not any((condition.evaluate(fields) for condition in self.conditions))

    def convert_to_elasticsearch(self) -> Dict[str, Dict]:
        if False:
            print('Hello World!')
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {'bool': {'must_not': conditions}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            print('Hello World!')
        sql_import.check()
        conditions = [meta_document_orm.document_id.in_(condition.convert_to_sql(meta_document_orm)) for condition in self.conditions]
        return select(meta_document_orm.document_id).filter(~or_(*conditions))

    def convert_to_weaviate(self) -> Dict[str, Union[str, int, float, bool, List[Dict]]]:
        if False:
            while True:
                i = 10
        conditions = [condition.invert().convert_to_weaviate() for condition in self.conditions]
        if len(conditions) > 1:
            return {'operator': 'Or', 'operands': conditions}
        else:
            return conditions[0]

    def convert_to_pinecone(self) -> Dict[str, Union[str, int, float, bool, List[Dict]]]:
        if False:
            for i in range(10):
                print('nop')
        conditions = [condition.invert().convert_to_pinecone() for condition in self.conditions]
        if len(conditions) > 1:
            return {'$or': conditions}
        else:
            return conditions[0]

    def invert(self) -> Union[LogicalFilterClause, ComparisonOperation]:
        if False:
            for i in range(10):
                print('nop')
        if len(self.conditions) > 1:
            return AndOperation(self.conditions)
        else:
            return self.conditions[0]

class AndOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'AND' operations.
    """

    def evaluate(self, fields) -> bool:
        if False:
            while True:
                i = 10
        return all((condition.evaluate(fields) for condition in self.conditions))

    def convert_to_elasticsearch(self) -> Dict[str, Dict]:
        if False:
            for i in range(10):
                print('nop')
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {'bool': {'must': conditions}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            for i in range(10):
                print('nop')
        sql_import.check()
        conditions = [meta_document_orm.document_id.in_(condition.convert_to_sql(meta_document_orm)) for condition in self.conditions]
        return select(meta_document_orm.document_id).filter(and_(*conditions))

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        if False:
            while True:
                i = 10
        conditions = [condition.convert_to_weaviate() for condition in self.conditions]
        return {'operator': 'And', 'operands': conditions}

    def convert_to_pinecone(self) -> Dict[str, Union[str, List[Dict]]]:
        if False:
            while True:
                i = 10
        conditions = [condition.convert_to_pinecone() for condition in self.conditions]
        return {'$and': conditions}

    def invert(self) -> 'OrOperation':
        if False:
            while True:
                i = 10
        return OrOperation([condition.invert() for condition in self.conditions])

class OrOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'OR' operations.
    """

    def evaluate(self, fields) -> bool:
        if False:
            print('Hello World!')
        return any((condition.evaluate(fields) for condition in self.conditions))

    def convert_to_elasticsearch(self) -> Dict[str, Dict]:
        if False:
            for i in range(10):
                print('nop')
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {'bool': {'should': conditions}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            return 10
        sql_import.check()
        conditions = [meta_document_orm.document_id.in_(condition.convert_to_sql(meta_document_orm)) for condition in self.conditions]
        return select(meta_document_orm.document_id).filter(or_(*conditions))

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        if False:
            for i in range(10):
                print('nop')
        conditions = [condition.convert_to_weaviate() for condition in self.conditions]
        return {'operator': 'Or', 'operands': conditions}

    def convert_to_pinecone(self) -> Dict[str, Union[str, List[Dict]]]:
        if False:
            for i in range(10):
                print('nop')
        conditions = [condition.convert_to_pinecone() for condition in self.conditions]
        return {'$or': conditions}

    def invert(self) -> AndOperation:
        if False:
            return 10
        return AndOperation([condition.invert() for condition in self.conditions])

class EqOperation(ComparisonOperation):
    """
    Handles conversion of the '$eq' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self.field_name not in fields:
            return False
        return fields[self.field_name] == self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Union[str, int, float, bool, Dict[str, Union[list, Dict[str, str]]]]]]:
        if False:
            i = 10
            return i + 15
        if isinstance(self.comparison_value, list):
            return {'terms_set': {self.field_name: {'terms': self.comparison_value, 'minimum_should_match_script': {'source': f"Math.max(params.num_terms, doc['{self.field_name}'].size())"}}}}
        return {'term': {self.field_name: self.comparison_value}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            i = 10
            return i + 15
        sql_import.check()
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value == self.comparison_value)

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, int, float, bool]]:
        if False:
            i = 10
            return i + 15
        (comp_value_type, comp_value) = self._get_weaviate_datatype()
        return {'path': [self.field_name], 'operator': 'Equal', comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[List[str], str, int, float, bool]]]:
        if False:
            i = 10
            return i + 15
        return {self.field_name: {'$eq': self.comparison_value}}

    def invert(self) -> 'NeOperation':
        if False:
            return 10
        return NeOperation(self.field_name, self.comparison_value)

class InOperation(ComparisonOperation):
    """
    Handles conversion of the '$in' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if False:
            return 10
        if self.field_name not in fields:
            return False
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$in' operation requires comparison value to be a list.")
        if isinstance(fields[self.field_name], list):
            return any((field in self.comparison_value for field in fields[self.field_name]))
        return fields[self.field_name] in self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, List]]:
        if False:
            print('Hello World!')
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$in' operation requires comparison value to be a list.")
        return {'terms': {self.field_name: self.comparison_value}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            print('Hello World!')
        sql_import.check()
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value.in_(self.comparison_value))

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        if False:
            print('Hello World!')
        filter_dict: Dict[str, Union[str, List[Dict]]] = {'operator': 'Or', 'operands': []}
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$in' operation requires comparison value to be a list.")
        for value in self.comparison_value:
            (comp_value_type, comp_value) = self._get_weaviate_datatype(value)
            assert isinstance(filter_dict['operands'], list)
            filter_dict['operands'].append({'path': [self.field_name], 'operator': 'Equal', comp_value_type: comp_value})
        return filter_dict

    def convert_to_pinecone(self) -> Dict[str, Dict[str, List]]:
        if False:
            return 10
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$in' operation requires comparison value to be a list.")
        return {self.field_name: {'$in': self.comparison_value}}

    def invert(self) -> 'NinOperation':
        if False:
            for i in range(10):
                print('nop')
        return NinOperation(self.field_name, self.comparison_value)

class NeOperation(ComparisonOperation):
    """
    Handles conversion of the '$ne' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if False:
            while True:
                i = 10
        if self.field_name not in fields:
            return False
        return fields[self.field_name] != self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Dict[str, Union[str, int, float, bool]]]]]:
        if False:
            return 10
        if isinstance(self.comparison_value, list):
            raise FilterError("Use '$nin' operation for lists as comparison values.")
        return {'bool': {'must_not': {'term': {self.field_name: self.comparison_value}}}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            while True:
                i = 10
        sql_import.check()
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value != self.comparison_value)

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, int, float, bool]]:
        if False:
            while True:
                i = 10
        (comp_value_type, comp_value) = self._get_weaviate_datatype()
        return {'path': [self.field_name], 'operator': 'NotEqual', comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[List[str], str, int, float, bool]]]:
        if False:
            return 10
        return {self.field_name: {'$ne': self.comparison_value}}

    def invert(self) -> 'EqOperation':
        if False:
            print('Hello World!')
        return EqOperation(self.field_name, self.comparison_value)

class NinOperation(ComparisonOperation):
    """
    Handles conversion of the '$nin' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if False:
            i = 10
            return i + 15
        if self.field_name not in fields:
            return True
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$nin' operation requires comparison value to be a list.")
        if isinstance(fields[self.field_name], list):
            return not any((field in self.comparison_value for field in fields[self.field_name]))
        return fields[self.field_name] not in self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Dict[str, List]]]]:
        if False:
            print('Hello World!')
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$nin' operation requires comparison value to be a list.")
        return {'bool': {'must_not': {'terms': {self.field_name: self.comparison_value}}}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            for i in range(10):
                print('nop')
        sql_import.check()
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value.notin_(self.comparison_value))

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        if False:
            while True:
                i = 10
        filter_dict: Dict[str, Union[str, List[Dict]]] = {'operator': 'And', 'operands': []}
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$nin' operation requires comparison value to be a list.")
        for value in self.comparison_value:
            (comp_value_type, comp_value) = self._get_weaviate_datatype(value)
            assert isinstance(filter_dict['operands'], list)
            filter_dict['operands'].append({'path': [self.field_name], 'operator': 'NotEqual', comp_value_type: comp_value})
        return filter_dict

    def convert_to_pinecone(self) -> Dict[str, Dict[str, List]]:
        if False:
            print('Hello World!')
        if not isinstance(self.comparison_value, list):
            raise FilterError("'$in' operation requires comparison value to be a list.")
        return {self.field_name: {'$nin': self.comparison_value}}

    def invert(self) -> 'InOperation':
        if False:
            i = 10
            return i + 15
        return InOperation(self.field_name, self.comparison_value)

class GtOperation(ComparisonOperation):
    """
    Handles conversion of the '$gt' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if False:
            while True:
                i = 10
        if self.field_name not in fields:
            return False
        if isinstance(fields[self.field_name], list):
            return any((field > self.comparison_value for field in fields[self.field_name]))
        return fields[self.field_name] > self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.comparison_value, list):
            raise FilterError("Comparison value for '$gt' operation must not be a list.")
        return {'range': {self.field_name: {'gt': self.comparison_value}}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            i = 10
            return i + 15
        sql_import.check()
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value > self.comparison_value)

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, float, int]]:
        if False:
            for i in range(10):
                print('nop')
        (comp_value_type, comp_value) = self._get_weaviate_datatype()
        if isinstance(comp_value, list):
            raise FilterError("Comparison value for '$gt' operation must not be a list.")
        return {'path': [self.field_name], 'operator': 'GreaterThan', comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if False:
            i = 10
            return i + 15
        if not isinstance(self.comparison_value, (float, int)):
            raise FilterError("Comparison value for '$gt' operation must be a float or int.")
        return {self.field_name: {'$gt': self.comparison_value}}

    def invert(self) -> 'LteOperation':
        if False:
            return 10
        return LteOperation(self.field_name, self.comparison_value)

class GteOperation(ComparisonOperation):
    """
    Handles conversion of the '$gte' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if False:
            print('Hello World!')
        if self.field_name not in fields:
            return False
        if isinstance(fields[self.field_name], list):
            return any((field >= self.comparison_value for field in fields[self.field_name]))
        return fields[self.field_name] >= self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.comparison_value, list):
            raise FilterError("Comparison value for '$gte' operation must not be a list.")
        return {'range': {self.field_name: {'gte': self.comparison_value}}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            return 10
        sql_import.check()
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value >= self.comparison_value)

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, float, int]]:
        if False:
            print('Hello World!')
        (comp_value_type, comp_value) = self._get_weaviate_datatype()
        if isinstance(comp_value, list):
            raise FilterError("Comparison value for '$gte' operation must not be a list.")
        return {'path': [self.field_name], 'operator': 'GreaterThanEqual', comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(self.comparison_value, (float, int)):
            raise FilterError("Comparison value for '$gte' operation must be a float or int.")
        return {self.field_name: {'$gte': self.comparison_value}}

    def invert(self) -> 'LtOperation':
        if False:
            i = 10
            return i + 15
        return LtOperation(self.field_name, self.comparison_value)

class LtOperation(ComparisonOperation):
    """
    Handles conversion of the '$lt' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if False:
            while True:
                i = 10
        if self.field_name not in fields:
            return False
        if isinstance(fields[self.field_name], list):
            return any((field < self.comparison_value for field in fields[self.field_name]))
        return fields[self.field_name] < self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        if False:
            return 10
        if isinstance(self.comparison_value, list):
            raise FilterError("Comparison value for '$lt' operation must not be a list.")
        return {'range': {self.field_name: {'lt': self.comparison_value}}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            return 10
        sql_import.check()
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value < self.comparison_value)

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, float, int]]:
        if False:
            return 10
        (comp_value_type, comp_value) = self._get_weaviate_datatype()
        if isinstance(comp_value, list):
            raise FilterError("Comparison value for '$lt' operation must not be a list.")
        return {'path': [self.field_name], 'operator': 'LessThan', comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if False:
            return 10
        if not isinstance(self.comparison_value, (float, int)):
            raise FilterError("Comparison value for '$lt' operation must be a float or int.")
        return {self.field_name: {'$lt': self.comparison_value}}

    def invert(self) -> 'GteOperation':
        if False:
            i = 10
            return i + 15
        return GteOperation(self.field_name, self.comparison_value)

class LteOperation(ComparisonOperation):
    """
    Handles conversion of the '$lte' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if False:
            print('Hello World!')
        if self.field_name not in fields:
            return False
        if isinstance(fields[self.field_name], list):
            return any((field <= self.comparison_value for field in fields[self.field_name]))
        return fields[self.field_name] <= self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        if False:
            return 10
        if isinstance(self.comparison_value, list):
            raise FilterError("Comparison value for '$lte' operation must not be a list.")
        return {'range': {self.field_name: {'lte': self.comparison_value}}}

    def convert_to_sql(self, meta_document_orm):
        if False:
            i = 10
            return i + 15
        sql_import.check()
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value <= self.comparison_value)

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, float, int]]:
        if False:
            while True:
                i = 10
        (comp_value_type, comp_value) = self._get_weaviate_datatype()
        if isinstance(comp_value, list):
            raise FilterError("Comparison value for '$lte' operation must not be a list.")
        return {'path': [self.field_name], 'operator': 'LessThanEqual', comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        if False:
            return 10
        if not isinstance(self.comparison_value, (float, int)):
            raise FilterError("Comparison value for '$lte' operation must be a float or int.")
        return {self.field_name: {'$lte': self.comparison_value}}

    def invert(self) -> 'GtOperation':
        if False:
            for i in range(10):
                print('nop')
        return GtOperation(self.field_name, self.comparison_value)