import logging
from abc import abstractmethod
from typing import Any, Mapping, Optional, Sequence
from dagster import get_dagster_logger
from .types import DbtOutput

class DbtClient:
    """Base class for a client allowing users to interface with dbt."""

    def __init__(self, logger: Optional[logging.Logger]=None):
        if False:
            return 10
        'Constructor.\n\n        Args:\n            logger (Optional[Any]): A property for injecting a logger dependency.\n                Default is ``None``.\n        '
        self._logger = logger or get_dagster_logger()

    def _format_params(self, flags: Mapping[str, Any], replace_underscores: bool=False) -> Mapping[str, Any]:
        if False:
            print('Hello World!')
        'Reformats arguments that are easier to express as a list into the format that dbt expects,\n        and deletes and keys with no value.\n        '
        if replace_underscores:
            flags = {k.replace('_', '-'): v for (k, v) in flags.items() if v is not None}
        else:
            flags = {k: v for (k, v) in flags.items() if v is not None}
        for param in ['select', 'exclude', 'models']:
            if param in flags:
                if isinstance(flags[param], list):
                    flags[param] = ' '.join(set(flags[param]))
        return flags

    @property
    def logger(self) -> logging.Logger:
        if False:
            while True:
                i = 10
        'logging.Logger: A property for injecting a logger dependency.'
        return self._logger

    @abstractmethod
    def compile(self, models: Optional[Sequence[str]]=None, exclude: Optional[Sequence[str]]=None, **kwargs) -> DbtOutput:
        if False:
            i = 10
            return i + 15
        'Run the ``compile`` command on a dbt project. kwargs are passed in as additional parameters.\n\n        Args:\n            models (List[str], optional): the models to include in compilation.\n            exclude (List[str]), optional): the models to exclude from compilation.\n\n        Returns:\n            DbtOutput: object containing parsed output from dbt\n        '

    @abstractmethod
    def run(self, models: Optional[Sequence[str]]=None, exclude: Optional[Sequence[str]]=None, **kwargs) -> DbtOutput:
        if False:
            print('Hello World!')
        'Run the ``run`` command on a dbt project. kwargs are passed in as additional parameters.\n\n        Args:\n            models (List[str], optional): the models to include in the run.\n            exclude (List[str]), optional): the models to exclude from the run.\n\n        Returns:\n            DbtOutput: object containing parsed output from dbt\n        '

    @abstractmethod
    def snapshot(self, select: Optional[Sequence[str]]=None, exclude: Optional[Sequence[str]]=None, **kwargs) -> DbtOutput:
        if False:
            return 10
        'Run the ``snapshot`` command on a dbt project. kwargs are passed in as additional parameters.\n\n        Args:\n            select (List[str], optional): the snapshots to include in the run.\n            exclude (List[str], optional): the snapshots to exclude from the run.\n\n        Returns:\n            DbtOutput: object containing parsed output from dbt\n        '

    @abstractmethod
    def test(self, models: Optional[Sequence[str]]=None, exclude: Optional[Sequence[str]]=None, data: bool=True, schema: bool=True, **kwargs) -> DbtOutput:
        if False:
            return 10
        'Run the ``test`` command on a dbt project. kwargs are passed in as additional parameters.\n\n        Args:\n            models (List[str], optional): the models to include in testing.\n            exclude (List[str], optional): the models to exclude from testing.\n            data (bool, optional): If ``True`` (default), then run data tests.\n            schema (bool, optional): If ``True`` (default), then run schema tests.\n\n        Returns:\n            DbtOutput: object containing parsed output from dbt\n        '

    @abstractmethod
    def seed(self, show: bool=False, select: Optional[Sequence[str]]=None, exclude: Optional[Sequence[str]]=None, **kwargs) -> DbtOutput:
        if False:
            return 10
        'Run the ``seed`` command on a dbt project. kwargs are passed in as additional parameters.\n\n        Args:\n            show (bool, optional): If ``True``, then show a sample of the seeded data in the\n                response. Defaults to ``False``.\n            select (List[str], optional): the snapshots to include in the run.\n            exclude (List[str], optional): the snapshots to exclude from the run.\n\n\n        Returns:\n            DbtOutput: object containing parsed output from dbt\n        '

    @abstractmethod
    def ls(self, select: Optional[Sequence[str]]=None, models: Optional[Sequence[str]]=None, exclude: Optional[Sequence[str]]=None, **kwargs) -> DbtOutput:
        if False:
            for i in range(10):
                print('nop')
        'Run the ``ls`` command on a dbt project. kwargs are passed in as additional parameters.\n\n        Args:\n            select (List[str], optional): the resources to include in the output.\n            models (List[str], optional): the models to include in the output.\n            exclude (List[str], optional): the resources to exclude from the output.\n\n\n        Returns:\n            DbtOutput: object containing parsed output from dbt\n        '

    @abstractmethod
    def build(self, select: Optional[Sequence[str]]=None, **kwargs) -> DbtOutput:
        if False:
            i = 10
            return i + 15
        'Run the ``build`` command on a dbt project. kwargs are passed in as additional parameters.\n\n        Args:\n            select (List[str], optional): the models/resources to include in the run.\n\n        Returns:\n            DbtOutput: object containing parsed output from dbt\n        '
        raise NotImplementedError()

    @abstractmethod
    def generate_docs(self, compile_project: bool=False, **kwargs) -> DbtOutput:
        if False:
            while True:
                i = 10
        'Run the ``docs generate`` command on a dbt project. kwargs are passed in as additional parameters.\n\n        Args:\n            compile_project (bool, optional): If true, compile the project before generating a catalog.\n\n        Returns:\n            DbtOutput: object containing parsed output from dbt\n        '

    @abstractmethod
    def run_operation(self, macro: str, args: Optional[Mapping[str, Any]]=None, **kwargs) -> DbtOutput:
        if False:
            for i in range(10):
                print('nop')
        'Run the ``run-operation`` command on a dbt project. kwargs are passed in as additional parameters.\n\n        Args:\n            macro (str): the dbt macro to invoke.\n            args (Dict[str, Any], optional): the keyword arguments to be supplied to the macro.\n\n        Returns:\n            DbtOutput: object containing parsed output from dbt\n        '

    @abstractmethod
    def get_run_results_json(self, **kwargs) -> Optional[Mapping[str, Any]]:
        if False:
            i = 10
            return i + 15
        'Get a parsed version of the run_results.json file for the relevant dbt project.\n\n        Returns:\n            Dict[str, Any]: dictionary containing the parsed contents of the run_results json file\n                for this dbt project.\n        '

    @abstractmethod
    def get_manifest_json(self, **kwargs) -> Optional[Mapping[str, Any]]:
        if False:
            print('Hello World!')
        'Get a parsed version of the manifest.json file for the relevant dbt project.\n\n        Returns:\n            Dict[str, Any]: dictionary containing the parsed contents of the manifest json file\n                for this dbt project.\n        '

class DbtResource(DbtClient):
    pass