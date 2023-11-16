import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import dbt.flags as flags
import pandas as pd
from dbt.adapters.base import BaseRelation, Credentials
from dbt.adapters.factory import Adapter, cleanup_connections, get_adapter, register_adapter, reset_adapters
from dbt.config.profile import read_user_config
from dbt.config.runtime import RuntimeConfig
from dbt.contracts.connection import AdapterResponse
from dbt.contracts.relation import RelationType
from mage_ai.data_preparation.models.block.dbt.profiles import Profiles

@dataclass
class DBTAdapterConfig:
    """
    Minimal config needed in order to setup dbt adapter
    """
    project_dir: Union[str, os.PathLike]
    profiles_dir: Union[str, os.PathLike]
    profile: Optional[str] = None
    target: Optional[str] = None

class DBTAdapter:

    def __init__(self, project_path: Union[str, os.PathLike], variables: Optional[Dict[str, Any]]=None, target: Optional[str]=None):
        if False:
            print('Hello World!')
        '\n        Set up dbt adapter. This allows to use any dbt based connections.\n\n        Args:\n            project_path (Union[str, os.PathLike]):\n                Project, which should be used for setting up the dbt adapter\n            variables (Optional[Dict[str, Any]], optional):\n                Variables for interpolating the profiles.yml. Defaults to None.\n            target (Optional[str], optional):\n                Whether to use a target other than the one configured in profiles.yml.\n                Defaults to None.\n        '
        self.project_path: Union[str, os.PathLike] = project_path
        self.variables: Optional[Dict[str, Any]] = variables
        self.target: Optional[str] = target
        self.__adapter: Optional[Adapter] = None
        self.__profiles: Optional[Profiles] = None

    @property
    def credentials(self) -> Credentials:
        if False:
            while True:
                i = 10
        '\n        The credentials object, which has all database credentials.\n\n        Returns:\n            Credentials: Database credentials of the adapter\n        '
        return self.__adapter.connections.profile.credentials

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Close connection, which was opened by the adapter\n        '
        self.__adapter.release_connection()
        cleanup_connections()
        self.__profiles.clean()

    def execute(self, sql: str, fetch: bool=False) -> Tuple[AdapterResponse, pd.DataFrame]:
        if False:
            i = 10
            return i + 15
        '\n        Executes any sql statement using the dbt adapter.\n\n        Args:\n            sql (str): The sql statement to execute.\n            fetch (bool, optional):\n                Whether to fetch results from the sql statement. Defaults to False.\n\n        Returns:\n            Tuple[AdapterResponse, pd.DataFrame]: Adapter Response and the result dataframe.\n        '
        (res, table) = self.__adapter.execute(sql, fetch=fetch)
        df = pd.DataFrame(table.rows, table.column_names)
        return (res, df)

    def execute_macro(self, macro_name: str, package: Optional[str]=None, context_overide: Optional[Dict[str, Any]]=None, **kwargs) -> Optional[Any]:
        if False:
            return 10
        '\n        Executes any dbt macro by name.\n\n        Args:\n            macro_name (str): Name of the macro\n            package (Optional[str], optional):\n                Name of the package of the macro.\n                Defaults to None, which uses the project macros-path only.\n            context_overide (Optional[Dict[str, Any]], optional):\n                Additional context for the macro execution. E.g. can be used to inject functions\n                or variables like the common dbt `this`. Defaults to None.\n\n        Returns:\n            Optional[Any]: Macro result\n        '
        self.__adapter.connections.begin()
        from dbt.parser.manifest import ManifestLoader
        manifest = ManifestLoader.load_macros(self.__adapter.config, self.__adapter.connections.set_query_header, base_macros_only=False)
        macro = manifest.find_macro_by_name(macro_name, self.__adapter.config.project_name, package)
        from dbt.context.providers import generate_runtime_macro_context
        macro_context = generate_runtime_macro_context(macro=macro, config=self.__adapter.config, manifest=manifest, package_name=package)
        if context_overide:
            macro_context.update(context_overide)
        from dbt.clients.jinja import MacroGenerator
        macro_function = MacroGenerator(macro, macro_context)
        with self.__adapter.connections.exception_handler(f'macro {macro_name}'):
            result = macro_function(**kwargs)
            self.__adapter.connections.commit()
        return result

    def get_relation(self, database: Optional[str]=None, schema: Optional[str]=None, identifier: Optional[str]=None, type: Optional[RelationType]=RelationType.Table) -> BaseRelation:
        if False:
            i = 10
            return i + 15
        '\n        Gets a relation, which can be used in conjunction with dbt macros.\n\n        Args:\n            database (Optional[str], optional):\n                The database to use. Defaults to None.\n            schema (Optional[str], optional):\n                The schema to use. Defaults to None.\n            identifier (Optional[str], optional):\n                The identifier to use. Defaults to None.\n            type (Optional[RelationType], optional):\n                Of which type the relation is (e.g. table/view). Defaults to RelationType.Table.\n\n        Returns:\n            BaseRelation: initialized dbt Relation\n        '
        return self.__adapter.Relation.create(database=database, schema=schema, identifier=identifier, quote_policy=self.__adapter.Relation.get_default_quote_policy().to_dict(omit_none=True), type=type)

    def open(self) -> 'DBTAdapter':
        if False:
            return 10
        '\n        Opens the connection to database configured by dbt\n\n        Returns:\n            DBTAdapter: DBTAdapter with opened connection\n        '
        self.__profiles = Profiles(self.project_path, self.variables)
        profiles_path = self.__profiles.interpolate()
        user_config = read_user_config(profiles_path)
        adapter_config = DBTAdapterConfig(project_dir=self.project_path, profiles_dir=profiles_path, target=self.target)
        flags.set_from_args(adapter_config, user_config)
        config = RuntimeConfig.from_args(adapter_config)
        reset_adapters()
        register_adapter(config)
        self.__adapter = get_adapter(config)
        self.__adapter.acquire_connection('mage_dbt_adapter_' + uuid.uuid4().hex)
        return self

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.open()

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        self.close()