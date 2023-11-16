"""Data API Connector base class."""
import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from types import TracebackType
from typing import Any, Dict, List, Optional, Type, Union
import awswrangler.pandas as pd

class DataApiConnector(ABC):
    """Base class for Data API (RDS, Redshift, etc.) connectors."""

    def execute(self, sql: str, database: Optional[str]=None, transaction_id: Optional[str]=None, parameters: Optional[List[Dict[str, Any]]]=None) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        'Execute SQL statement against a Data API Service.\n\n        Parameters\n        ----------\n        sql: str\n            SQL statement to execute.\n\n        Returns\n        -------\n        A Pandas DataFrame containing the execution results.\n        '
        request_id: str = self._execute_statement(sql, database=database, transaction_id=transaction_id, parameters=parameters)
        return self._get_statement_result(request_id)

    def batch_execute(self, sql: Union[str, List[str]], database: Optional[str]=None, transaction_id: Optional[str]=None, parameter_sets: Optional[List[List[Dict[str, Any]]]]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Batch execute SQL statements against a Data API Service.\n\n        Parameters\n        ----------\n        sql: str\n            SQL statement to execute.\n        '
        self._batch_execute_statement(sql, database=database, transaction_id=transaction_id, parameter_sets=parameter_sets)

    def __enter__(self) -> 'DataApiConnector':
        if False:
            for i in range(10):
                print('nop')
        return self

    @abstractmethod
    def close(self) -> None:
        if False:
            while True:
                i = 10
        'Close underlying endpoint connections.'
        pass

    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException], traceback: Optional[TracebackType]) -> Optional[bool]:
        if False:
            while True:
                i = 10
        self.close()
        return None

    @abstractmethod
    def begin_transaction(self, database: Optional[str]=None, schema: Optional[str]=None) -> str:
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def commit_transaction(self, transaction_id: str) -> str:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def rollback_transaction(self, transaction_id: str) -> str:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def _execute_statement(self, sql: str, database: Optional[str]=None, transaction_id: Optional[str]=None, parameters: Optional[List[Dict[str, Any]]]=None) -> str:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def _batch_execute_statement(self, sql: Union[str, List[str]], database: Optional[str]=None, transaction_id: Optional[str]=None, parameter_sets: Optional[List[List[Dict[str, Any]]]]=None) -> str:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def _get_statement_result(self, request_id: str) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def _get_column_value(column_value: Dict[str, Any], col_type: Optional[str]=None) -> Any:
        if False:
            while True:
                i = 10
        'Return the first non-null key value for a given dictionary.\n\n        The key names for a given record depend on the column type: stringValue, longValue, etc.\n\n        Therefore, a record in the response does not have consistent key names. The ColumnMetadata\n        typeName information could be used to infer the key, but there is no direct mapping here\n        that could be easily parsed with creating a static dictionary:\n            varchar -> stringValue\n            int2 -> longValue\n            timestamp -> stringValue\n\n        What has been observed is that each record appears to have a single key, so this function\n        iterates over the keys and returns the first non-null value. If none are found, None is\n        returned.\n\n        Documentation:\n            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/redshift-data.html#RedshiftDataAPIService.Client.get_statement_result\n            https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rds-data.html#RDSDataService.Client.execute_statement\n        '
        for key in column_value:
            if column_value[key] is not None:
                if key == 'isNull' and column_value[key]:
                    return None
                if key == 'arrayValue':
                    raise ValueError(f'arrayValue not supported yet - could not extract {column_value[key]}')
                if key == 'stringValue':
                    if col_type == 'DATETIME':
                        return dt.datetime.strptime(column_value[key], '%Y-%m-%d %H:%M:%S')
                    if col_type == 'DATE':
                        return dt.datetime.strptime(column_value[key], '%Y-%m-%d').date()
                    if col_type == 'TIME':
                        return dt.datetime.strptime(column_value[key], '%H:%M:%S').time()
                    if col_type == 'DECIMAL':
                        return Decimal(column_value[key])
                return column_value[key]
        return None

@dataclass
class WaitConfig:
    """Holds standard wait configuration values."""
    sleep: float
    backoff: float
    retries: int