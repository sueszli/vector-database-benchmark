from typing import List
from dagster import job, op

@op
def op_1():
    if False:
        return 10
    return []

@op
def op_2(_a):
    if False:
        for i in range(10):
            print('nop')
    return []

def write_dataframe_to_table(**_kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass

def read_dataframe_from_table(**_kwargs):
    if False:
        for i in range(10):
            print('nop')
    return []

def read_csv(_path):
    if False:
        i = 10
        return i + 15
    pass

def write_csv(_path, _obj):
    if False:
        i = 10
        return i + 15
    pass
from dagster import ConfigurableIOManager, InputContext, OutputContext

class MyIOManager(ConfigurableIOManager):
    path_prefix: List[str] = []

    def _get_path(self, context) -> str:
        if False:
            print('Hello World!')
        return '/'.join(self.path_prefix + context.asset_key.path)

    def handle_output(self, context: OutputContext, obj):
        if False:
            while True:
                i = 10
        write_csv(self._get_path(context), obj)

    def load_input(self, context: InputContext):
        if False:
            while True:
                i = 10
        return read_csv(self._get_path(context))
from dagster import IOManager, ConfigurableIOManagerFactory, OutputContext, InputContext
import requests

class ExternalIOManager(IOManager):

    def __init__(self, api_token):
        if False:
            print('Hello World!')
        self._api_token = api_token
        self._cache = {}

    def handle_output(self, context: OutputContext, obj):
        if False:
            while True:
                i = 10
        ...

    def load_input(self, context: InputContext):
        if False:
            while True:
                i = 10
        if context.asset_key in self._cache:
            return self._cache[context.asset_key]
        ...

class ConfigurableExternalIOManager(ConfigurableIOManagerFactory):
    api_token: str

    def create_io_manager(self, context) -> ExternalIOManager:
        if False:
            i = 10
            return i + 15
        return ExternalIOManager(self.api_token)

class MyPartitionedIOManager(IOManager):

    def _get_path(self, context) -> str:
        if False:
            while True:
                i = 10
        if context.has_partition_key:
            return '/'.join(context.asset_key.path + [context.asset_partition_key])
        else:
            return '/'.join(context.asset_key.path)

    def handle_output(self, context: OutputContext, obj):
        if False:
            i = 10
            return i + 15
        write_csv(self._get_path(context), obj)

    def load_input(self, context: InputContext):
        if False:
            for i in range(10):
                print('nop')
        return read_csv(self._get_path(context))
from dagster import ConfigurableIOManager, io_manager

class DataframeTableIOManager(ConfigurableIOManager):

    def handle_output(self, context: OutputContext, obj):
        if False:
            for i in range(10):
                print('nop')
        table_name = context.name
        write_dataframe_to_table(name=table_name, dataframe=obj)

    def load_input(self, context: InputContext):
        if False:
            i = 10
            return i + 15
        if context.upstream_output:
            table_name = context.upstream_output.name
            return read_dataframe_from_table(name=table_name)

@job(resource_defs={'io_manager': DataframeTableIOManager()})
def my_job():
    if False:
        return 10
    op_2(op_1())

class DataframeTableIOManagerWithMetadata(ConfigurableIOManager):

    def handle_output(self, context: OutputContext, obj):
        if False:
            return 10
        table_name = context.name
        write_dataframe_to_table(name=table_name, dataframe=obj)
        context.add_output_metadata({'num_rows': len(obj), 'table_name': table_name})

    def load_input(self, context: InputContext):
        if False:
            return 10
        if context.upstream_output:
            table_name = context.upstream_output.name
            return read_dataframe_from_table(name=table_name)

@job(resource_defs={'io_manager': DataframeTableIOManagerWithMetadata()})
def my_job_with_metadata():
    if False:
        i = 10
        return i + 15
    op_2(op_1())