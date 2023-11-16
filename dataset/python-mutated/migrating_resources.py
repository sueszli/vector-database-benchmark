from typing import Iterator
from dagster import Definitions, define_asset_job

def initial_code_base() -> Definitions:
    if False:
        print('Hello World!')
    from dagster import AssetExecutionContext, Definitions, InitResourceContext, asset, resource

    class FancyDbResource:

        def __init__(self, conn_string: str) -> None:
            if False:
                print('Hello World!')
            self.conn_string = conn_string

        def execute(self, query: str) -> None:
            if False:
                print('Hello World!')
            ...

    @resource(config_schema={'conn_string': str})
    def fancy_db_resource(context: InitResourceContext) -> FancyDbResource:
        if False:
            while True:
                i = 10
        return FancyDbResource(context.resource_config['conn_string'])

    @asset(required_resource_keys={'fancy_db'})
    def asset_one(context: AssetExecutionContext) -> None:
        if False:
            while True:
                i = 10
        assert context.resources.fancy_db

    @asset(required_resource_keys={'fancy_db'})
    def asset_two(context: AssetExecutionContext) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert context.resources.fancy_db
    defs = Definitions(assets=[asset_one, asset_two], resources={'fancy_db': fancy_db_resource.configured({'conn_string': 'some_value'})})
    return defs

def convert_resource() -> Definitions:
    if False:
        print('Hello World!')
    from dagster import ConfigurableResource

    class FancyDbResource(ConfigurableResource):
        conn_string: str

        def execute(self, query: str) -> None:
            if False:
                i = 10
                return i + 15
            ...
    from dagster import InitResourceContext, resource

    @resource(config_schema=FancyDbResource.to_config_schema())
    def fancy_db_resource(context: InitResourceContext) -> FancyDbResource:
        if False:
            for i in range(10):
                print('nop')
        return FancyDbResource.from_resource_context(context)
    defs = Definitions(resources={'fancy_db': fancy_db_resource.configured({'conn_string': 'some_value'})})
    return defs

def new_style_resource_on_context() -> Definitions:
    if False:
        i = 10
        return i + 15
    from dagster import AssetExecutionContext, ConfigurableResource, Definitions, asset

    class FancyDbResource(ConfigurableResource):
        conn_string: str

        def execute(self, query: str) -> None:
            if False:
                while True:
                    i = 10
            ...

    @asset(required_resource_keys={'fancy_db'})
    def asset_one(context: AssetExecutionContext) -> None:
        if False:
            i = 10
            return i + 15
        assert context.resources.fancy_db
    defs = Definitions(assets=[asset_one], resources={'fancy_db': FancyDbResource(conn_string='some_value')})
    return defs

def new_style_resource_on_param() -> Definitions:
    if False:
        while True:
            i = 10
    from dagster import ConfigurableResource, Definitions, OpExecutionContext, asset

    class FancyDbResource(ConfigurableResource):
        conn_string: str

        def execute(self, query: str) -> None:
            if False:
                i = 10
                return i + 15
            ...
    from dagster import AssetExecutionContext, asset

    @asset
    def asset_one(context: AssetExecutionContext, fancy_db: FancyDbResource) -> None:
        if False:
            while True:
                i = 10
        assert fancy_db
    return Definitions(assets=[asset_one], resources={'fancy_db': FancyDbResource(conn_string='some_value')})

def old_third_party_resource() -> Definitions:
    if False:
        print('Hello World!')

    class FancyDbClient:

        def __init__(self, conn_string: str) -> None:
            if False:
                while True:
                    i = 10
            self.conn_string = conn_string

        def execute_query(self, query: str) -> None:
            if False:
                i = 10
                return i + 15
            ...
    from dagster import AssetExecutionContext, InitResourceContext, asset, resource

    @resource(config_schema={'conn_string': str})
    def fancy_db_resource(context: InitResourceContext) -> FancyDbClient:
        if False:
            print('Hello World!')
        return FancyDbClient(context.resource_config['conn_string'])

    @asset(required_resource_keys={'fancy_db'})
    def existing_asset(context: AssetExecutionContext) -> None:
        if False:
            return 10
        context.resources.fancy_db.execute_query('SELECT * FROM foo')
    defs = Definitions(assets=[existing_asset], resources={'fancy_db': fancy_db_resource.configured({'conn_string': 'something'})})
    return defs

def some_expensive_setup() -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

def some_expensive_teardown() -> None:
    if False:
        i = 10
        return i + 15
    ...

def old_resource_code_contextmanager() -> Definitions:
    if False:
        while True:
            i = 10

    class FancyDbClient:

        def __init__(self, conn_string: str) -> None:
            if False:
                return 10
            self.conn_string = conn_string

        def execute_query(self, query: str) -> None:
            if False:
                while True:
                    i = 10
            ...
    from dagster import AssetExecutionContext, InitResourceContext, asset, resource

    @resource(config_schema={'conn_string': str})
    def fancy_db_resource(context: InitResourceContext) -> Iterator[FancyDbClient]:
        if False:
            while True:
                i = 10
        some_expensive_setup()
        try:
            yield FancyDbClient(context.resource_config['conn_string'])
        finally:
            some_expensive_teardown()

    @asset(required_resource_keys={'fancy_db'})
    def asset_one(context: AssetExecutionContext) -> None:
        if False:
            while True:
                i = 10
        context.resources.fancy_db.execute_query('SELECT * FROM foo')
    return Definitions(assets=[asset_one], resources={'fancy_db': fancy_db_resource.configured({'conn_string': 'something'})})

def new_resource_code_contextmanager() -> Definitions:
    if False:
        for i in range(10):
            print('nop')

    class FancyDbClient:

        def __init__(self, conn_string: str) -> None:
            if False:
                return 10
            self.conn_string = conn_string

        def execute_query(self, query: str) -> None:
            if False:
                print('Hello World!')
            ...
    from contextlib import contextmanager
    from dagster import ConfigurableResource, asset

    class FancyDbResource(ConfigurableResource):
        conn_string: str

        @contextmanager
        def get_client(self) -> Iterator[FancyDbClient]:
            if False:
                i = 10
                return i + 15
            try:
                some_expensive_setup()
                yield FancyDbClient(self.conn_string)
            finally:
                some_expensive_teardown()

    @asset
    def asset_one(fancy_db: FancyDbResource) -> None:
        if False:
            print('Hello World!')
        with fancy_db.get_client() as client:
            client.execute_query('SELECT * FROM foo')
    return Definitions(assets=[asset_one], resources={'fancy_db': FancyDbResource(conn_string='something')})

def new_third_party_resource_old_code_broken() -> Definitions:
    if False:
        return 10

    class FancyDbClient:

        def __init__(self, conn_string: str) -> None:
            if False:
                i = 10
                return i + 15
            self.conn_string = conn_string

        def execute_query(self, query: str) -> None:
            if False:
                print('Hello World!')
            ...
    from dagster import AssetExecutionContext, ConfigurableResource, asset

    class FancyDbResource(ConfigurableResource):
        conn_string: str

        def get_client(self) -> FancyDbClient:
            if False:
                return 10
            return FancyDbClient(self.conn_string)

    @asset
    def new_asset(fancy_db: FancyDbResource) -> None:
        if False:
            print('Hello World!')
        client = fancy_db.get_client()
        client.execute_query('SELECT * FROM foo')

    @asset(required_resource_keys={'fancy_db'})
    def existing_asset(context: AssetExecutionContext) -> None:
        if False:
            while True:
                i = 10
        context.resources.fancy_db.execute_query('SELECT * FROM foo')
    defs = Definitions(assets=[new_asset, existing_asset], jobs=[define_asset_job('new_asset_job', 'new_asset'), define_asset_job('existing_asset_job', 'existing_asset')], resources={'fancy_db': FancyDbResource(conn_string='some_value')})
    return defs

def new_third_party_resource_fixed() -> Definitions:
    if False:
        i = 10
        return i + 15

    class FancyDbClient:

        def __init__(self, conn_string: str) -> None:
            if False:
                return 10
            self.conn_string = conn_string

        def execute_query(self, query: str) -> None:
            if False:
                print('Hello World!')
            ...
    from dagster import AssetExecutionContext, ConfigurableResource, IAttachDifferentObjectToOpContext, asset

    class FancyDbResource(ConfigurableResource, IAttachDifferentObjectToOpContext):
        conn_string: str

        def get_object_to_set_on_execution_context(self) -> FancyDbClient:
            if False:
                i = 10
                return i + 15
            return self.get_client()

        def get_client(self) -> FancyDbClient:
            if False:
                return 10
            return FancyDbClient(self.conn_string)

    @asset
    def new_asset(fancy_db: FancyDbResource) -> None:
        if False:
            i = 10
            return i + 15
        client = fancy_db.get_client()
        client.execute_query('SELECT * FROM foo')

    @asset(required_resource_keys={'fancy_db'})
    def existing_asset(context: AssetExecutionContext) -> None:
        if False:
            for i in range(10):
                print('nop')
        context.resources.fancy_db.execute_query('SELECT * FROM foo')
    defs = Definitions(assets=[new_asset, existing_asset], jobs=[define_asset_job('new_asset_job', 'new_asset'), define_asset_job('existing_asset_job', 'existing_asset')], resources={'fancy_db': FancyDbResource(conn_string='some_value')})
    return defs