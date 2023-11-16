from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dagster import Definitions

def new_resource_testing() -> None:
    if False:
        for i in range(10):
            print('nop')
    from dagster import ConfigurableResource

    class MyResource(ConfigurableResource):
        value: str

        def get_value(self) -> str:
            if False:
                i = 10
                return i + 15
            return self.value

    def test_my_resource():
        if False:
            while True:
                i = 10
        assert MyResource(value='foo').get_value() == 'foo'
    test_my_resource()

def new_resource_testing_with_nesting() -> None:
    if False:
        i = 10
        return i + 15
    from dagster import ConfigurableResource

    class StringHolderResource(ConfigurableResource):
        value: str

    class MyResourceRequiresAnother(ConfigurableResource):
        foo: StringHolderResource
        bar: str

    def test_my_resource_with_nesting():
        if False:
            return 10
        string_holder = StringHolderResource(value='foo')
        resource = MyResourceRequiresAnother(foo=string_holder, bar='bar')
        assert resource.foo.value == 'foo'
        assert resource.bar == 'bar'
    test_my_resource_with_nesting()
from typing import TYPE_CHECKING, Dict, Any

def new_resources_assets_defs() -> 'Definitions':
    if False:
        while True:
            i = 10
    from dagster import asset, Definitions
    from dagster import ResourceParam
    import requests
    from typing import Dict, Any

    @asset
    def data_from_url(data_url: ResourceParam[str]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return requests.get(data_url).json()
    defs = Definitions(assets=[data_from_url], resources={'data_url': 'https://dagster.io'})
    return defs

def new_resources_ops_defs() -> 'Definitions':
    if False:
        return 10
    from dagster import op, Definitions, job, ResourceParam
    import requests

    @op
    def print_data_from_resource(data_url: ResourceParam[str]):
        if False:
            for i in range(10):
                print('nop')
        print(requests.get(data_url).json())

    @job
    def print_data_from_url_job():
        if False:
            for i in range(10):
                print('nop')
        print_data_from_resource()
    defs = Definitions(jobs=[print_data_from_url_job], resources={'data_url': 'https://dagster.io'})
    return defs

def new_resources_configurable_defs() -> 'Definitions':
    if False:
        for i in range(10):
            print('nop')
    from dagster import asset, Definitions, ConfigurableResource
    import requests
    from requests import Response

    class MyConnectionResource(ConfigurableResource):
        username: str

        def request(self, endpoint: str) -> Response:
            if False:
                print('Hello World!')
            return requests.get(f'https://my-api.com/{endpoint}', headers={'user-agent': 'dagster'})

    @asset
    def data_from_service(my_conn: MyConnectionResource) -> Dict[str, Any]:
        if False:
            return 10
        return my_conn.request('/fetch_data').json()
    defs = Definitions(assets=[data_from_service], resources={'my_conn': MyConnectionResource(username='my_user')})
    return defs

def new_resources_configurable_defs_ops() -> 'Definitions':
    if False:
        return 10
    from dagster import Definitions, job, op, ConfigurableResource
    import requests
    from requests import Response

    class MyConnectionResource(ConfigurableResource):
        username: str

        def request(self, endpoint: str) -> Response:
            if False:
                for i in range(10):
                    print('nop')
            return requests.get(f'https://my-api.com/{endpoint}', headers={'user-agent': 'dagster'})

    @op
    def update_service(my_conn: MyConnectionResource):
        if False:
            for i in range(10):
                print('nop')
        my_conn.request('/update')

    @job
    def update_service_job():
        if False:
            print('Hello World!')
        update_service()
    defs = Definitions(jobs=[update_service_job], resources={'my_conn': MyConnectionResource(username='my_user')})
    return defs

def new_resource_runtime() -> 'Definitions':
    if False:
        i = 10
        return i + 15
    from dagster import ConfigurableResource, Definitions, asset

    class DatabaseResource(ConfigurableResource):
        table: str

        def read(self):
            if False:
                while True:
                    i = 10
            ...

    @asset
    def data_from_database(db_conn: DatabaseResource):
        if False:
            print('Hello World!')
        return db_conn.read()
    defs = Definitions(assets=[data_from_database], resources={'db_conn': DatabaseResource.configure_at_launch()})
    from dagster import sensor, define_asset_job, RunRequest, RunConfig
    update_data_job = define_asset_job(name='update_data_job', selection=[data_from_database])

    @sensor(job=update_data_job)
    def table_update_sensor():
        if False:
            for i in range(10):
                print('nop')
        tables = ...
        for table_name in tables:
            yield RunRequest(run_config=RunConfig(resources={'db_conn': DatabaseResource(table=table_name)}))
    defs = Definitions(assets=[data_from_database], jobs=[update_data_job], resources={'db_conn': DatabaseResource.configure_at_launch()}, sensors=[table_update_sensor])
    return defs

def get_filestore_client(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass

def new_resources_nesting() -> 'Definitions':
    if False:
        return 10
    from dagster import asset

    @asset
    def my_asset():
        if False:
            while True:
                i = 10
        pass
    from dagster import Definitions, ConfigurableResource

    class CredentialsResource(ConfigurableResource):
        username: str
        password: str

    class FileStoreBucket(ConfigurableResource):
        credentials: CredentialsResource
        region: str

        def write(self, data: str):
            if False:
                for i in range(10):
                    print('nop')
            get_filestore_client(username=self.credentials.username, password=self.credentials.password, region=self.region).write(data)
    defs = Definitions(assets=[my_asset], resources={'bucket': FileStoreBucket(credentials=CredentialsResource(username='my_user', password='my_password'), region='us-east-1')})
    credentials = CredentialsResource.configure_at_launch()
    defs = Definitions(assets=[my_asset], resources={'credentials': credentials, 'bucket': FileStoreBucket(credentials=credentials, region='us-east-1')})
    return defs

def new_resources_env_vars() -> None:
    if False:
        print('Hello World!')
    from dagster import EnvVar, Definitions, ConfigurableResource

    class CredentialsResource(ConfigurableResource):
        username: str
        password: str
    defs = Definitions(assets=..., resources={'credentials': CredentialsResource(username=EnvVar('MY_USERNAME'), password=EnvVar('MY_PASSWORD'))})

class GitHubOrganization:

    def __init__(self, name: str):
        if False:
            print('Hello World!')
        self.name = name

    def repositories(self):
        if False:
            print('Hello World!')
        return ['dagster', 'dagster-webserver', 'dagster-graphql']

class GitHub:

    def __init__(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def organization(self, name: str):
        if False:
            i = 10
            return i + 15
        return GitHubOrganization(name)

def raw_github_resource() -> None:
    if False:
        i = 10
        return i + 15
    from dagster import Definitions, asset, ResourceParam

    @asset
    def public_github_repos(github: ResourceParam[GitHub]):
        if False:
            for i in range(10):
                print('nop')
        return github.organization('dagster-io').repositories()
    defs = Definitions(assets=[public_github_repos], resources={'github': GitHub(...)})
from contextlib import AbstractContextManager

class Connection(AbstractContextManager):

    def execute(self, query: str):
        if False:
            i = 10
            return i + 15
        return None

    def __enter__(self) -> 'Connection':
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        return False

class Engine:

    def connect(self) -> Connection:
        if False:
            i = 10
            return i + 15
        return Connection()

def create_engine(*args, **kwargs):
    if False:
        while True:
            i = 10
    return Engine()

def raw_github_resource_dep() -> None:
    if False:
        while True:
            i = 10
    from dagster import ConfigurableResource, ResourceDependency, Definitions

    class DBResource(ConfigurableResource):
        engine: ResourceDependency[Engine]

        def query(self, query: str):
            if False:
                print('Hello World!')
            with self.engine.connect() as conn:
                return conn.execute(query)
    engine = create_engine(...)
    defs = Definitions(assets=..., resources={'db': DBResource(engine=engine)})

def resource_adapter() -> None:
    if False:
        i = 10
        return i + 15
    from dagster import resource, Definitions, ResourceDefinition, asset, ConfigurableLegacyResourceAdapter

    class Writer:

        def __init__(self, prefix: str):
            if False:
                print('Hello World!')
            self._prefix = prefix

        def output(self, text: str) -> None:
            if False:
                i = 10
                return i + 15
            print(self._prefix + text)

    @resource(config_schema={'prefix': str})
    def writer_resource(context):
        if False:
            for i in range(10):
                print('nop')
        prefix = context.resource_config['prefix']
        return Writer(prefix)

    class WriterResource(ConfigurableLegacyResourceAdapter):
        prefix: str

        @property
        def wrapped_resource(self) -> ResourceDefinition:
            if False:
                print('Hello World!')
            return writer_resource

    @asset
    def my_asset(writer: Writer):
        if False:
            i = 10
            return i + 15
        writer.output('hello, world!')
    defs = Definitions(assets=[my_asset], resources={'writer': WriterResource(prefix='greeting: ')})

def io_adapter() -> None:
    if False:
        for i in range(10):
            print('nop')
    from dagster import Definitions, IOManagerDefinition, io_manager, IOManager, InputContext, ConfigurableLegacyIOManagerAdapter, OutputContext
    import os

    class OldFileIOManager(IOManager):

        def __init__(self, base_path: str):
            if False:
                i = 10
                return i + 15
            self.base_path = base_path

        def handle_output(self, context: OutputContext, obj):
            if False:
                for i in range(10):
                    print('nop')
            with open(os.path.join(self.base_path, context.step_key, context.name), 'w') as fd:
                fd.write(obj)

        def load_input(self, context: InputContext):
            if False:
                print('Hello World!')
            with open(os.path.join(self.base_path, context.upstream_output.step_key, context.upstream_output.name), 'r') as fd:
                return fd.read()

    @io_manager(config_schema={'base_path': str})
    def old_file_io_manager(context):
        if False:
            return 10
        base_path = context.resource_config['base_path']
        return OldFileIOManager(base_path)

    class MyIOManager(ConfigurableLegacyIOManagerAdapter):
        base_path: str

        @property
        def wrapped_io_manager(self) -> IOManagerDefinition:
            if False:
                for i in range(10):
                    print('nop')
            return old_file_io_manager
    defs = Definitions(assets=..., resources={'io_manager': MyIOManager(base_path='/tmp/')})

def impl_details_resolve() -> None:
    if False:
        while True:
            i = 10
    from dagster import ConfigurableResource

    class CredentialsResource(ConfigurableResource):
        username: str
        password: str

    class FileStoreBucket(ConfigurableResource):
        credentials: CredentialsResource
        region: str

        def write(self, data: str):
            if False:
                return 10
            get_filestore_client(username=self.credentials.username, password=self.credentials.password, region=self.region).write(data)
    unconfigured_credentials_resource = CredentialsResource.configure_at_launch()
    bucket = FileStoreBucket(credentials=unconfigured_credentials_resource, region='us-east-1')

def write_csv(path: str, obj: Any):
    if False:
        print('Hello World!')
    pass

def read_csv(path: str):
    if False:
        return 10
    pass

def new_io_manager() -> None:
    if False:
        for i in range(10):
            print('nop')
    from dagster import Definitions, AssetKey, OutputContext, InputContext, ConfigurableIOManager

    class MyIOManager(ConfigurableIOManager):
        root_path: str

        def _get_path(self, asset_key: AssetKey) -> str:
            if False:
                return 10
            return self.root_path + '/'.join(asset_key.path)

        def handle_output(self, context: OutputContext, obj):
            if False:
                print('Hello World!')
            write_csv(self._get_path(context.asset_key), obj)

        def load_input(self, context: InputContext):
            if False:
                return 10
            return read_csv(self._get_path(context.asset_key))
    defs = Definitions(assets=..., resources={'io_manager': MyIOManager(root_path='/tmp/')})

def raw_github_resource_factory() -> None:
    if False:
        while True:
            i = 10
    from dagster import ConfigurableResourceFactory, Resource, asset, Definitions, EnvVar

    class GitHubResource(ConfigurableResourceFactory[GitHub]):
        access_token: str

        def create_resource(self, _context) -> GitHub:
            if False:
                i = 10
                return i + 15
            return GitHub(self.access_token)

    @asset
    def public_github_repos(github: Resource[GitHub]):
        if False:
            for i in range(10):
                print('nop')
        return github.organization('dagster-io').repositories()
    defs = Definitions(assets=[public_github_repos], resources={'github': GitHubResource(access_token=EnvVar('GITHUB_ACCESS_TOKEN'))})

def new_resource_testing_with_context():
    if False:
        i = 10
        return i + 15
    from dagster import ConfigurableResource, build_init_resource_context, DagsterInstance
    from typing import Optional

    class MyContextResource(ConfigurableResource[GitHub]):
        base_path: Optional[str] = None

        def effective_base_path(self) -> str:
            if False:
                while True:
                    i = 10
            if self.base_path:
                return self.base_path
            instance = self.get_resource_context().instance
            assert instance
            return instance.storage_directory()

    def test_my_context_resource():
        if False:
            i = 10
            return i + 15
        with DagsterInstance.ephemeral() as instance:
            context = build_init_resource_context(instance=instance)
            assert MyContextResource(base_path=None).with_resource_context(context).effective_base_path() == instance.storage_directory()

def with_state_example() -> None:
    if False:
        for i in range(10):
            print('nop')
    from dagster import ConfigurableResource, asset
    import requests
    from pydantic import PrivateAttr

    class MyClientResource(ConfigurableResource):
        username: str
        password: str
        _api_token: str = PrivateAttr()

        def setup_for_execution(self, context) -> None:
            if False:
                i = 10
                return i + 15
            self._api_token = requests.get('https://my-api.com/token', auth=(self.username, self.password)).text

        def get_all_users(self):
            if False:
                while True:
                    i = 10
            return requests.get('https://my-api.com/users', headers={'Authorization': self._api_token})

    @asset
    def my_asset(client: MyClientResource):
        if False:
            return 10
        return client.get_all_users()

def with_complex_state_example() -> None:
    if False:
        return 10
    from dagster import ConfigurableResource, asset
    from contextlib import contextmanager
    from pydantic import PrivateAttr

    class DBConnection:
        ...

        def query(self, body: str):
            if False:
                i = 10
                return i + 15
            ...

    @contextmanager
    def get_database_connection(username: str, password: str):
        if False:
            print('Hello World!')
        ...

    class MyClientResource(ConfigurableResource):
        username: str
        password: str
        _db_connection: DBConnection = PrivateAttr()

        @contextmanager
        def yield_for_execution(self, context):
            if False:
                while True:
                    i = 10
            with get_database_connection(self.username, self.password) as conn:
                self._db_connection = conn
                yield self

        def query(self, body: str):
            if False:
                return 10
            return self._db_connection.query(body)

    @asset
    def my_asset(client: MyClientResource):
        if False:
            print('Hello World!')
        client.query('SELECT * FROM my_table')

def new_resource_testing_with_state_ops() -> None:
    if False:
        i = 10
        return i + 15
    from dagster import ConfigurableResource, op
    import mock

    class MyClient:
        ...

        def query(self, body: str):
            if False:
                for i in range(10):
                    print('nop')
            ...

    class MyClientResource(ConfigurableResource):
        username: str
        password: str

        def get_client(self):
            if False:
                for i in range(10):
                    print('nop')
            return MyClient(self.username, self.password)

    @op
    def my_op(client: MyClientResource):
        if False:
            i = 10
            return i + 15
        return client.get_client().query('SELECT * FROM my_table')

    def test_my_op():
        if False:
            return 10

        class FakeClient:

            def query(self, body: str):
                if False:
                    i = 10
                    return i + 15
                assert body == 'SELECT * FROM my_table'
                return 'my_result'
        mocked_client_resource = mock.Mock()
        mocked_client_resource.get_client.return_value = FakeClient()
        assert my_op(mocked_client_resource) == 'my_result'

def new_resource_on_sensor() -> None:
    if False:
        print('Hello World!')
    from dagster import sensor, RunRequest, SensorEvaluationContext, ConfigurableResource, job, Definitions, RunConfig
    import requests
    from typing import List

    class UsersAPI(ConfigurableResource):
        url: str

        def fetch_users(self) -> List[str]:
            if False:
                i = 10
                return i + 15
            return requests.get(self.url).json()

    @job
    def process_user():
        if False:
            while True:
                i = 10
        ...

    @sensor(job=process_user)
    def process_new_users_sensor(context: SensorEvaluationContext, users_api: UsersAPI):
        if False:
            return 10
        last_user = int(context.cursor) if context.cursor else 0
        users = users_api.fetch_users()
        num_users = len(users)
        for user_id in users[last_user:]:
            yield RunRequest(run_key=user_id, tags={'user_id': user_id})
        context.update_cursor(str(num_users))
    defs = Definitions(jobs=[process_user], sensors=[process_new_users_sensor], resources={'users_api': UsersAPI(url='https://my-api.com/users')})
    from dagster import build_sensor_context, validate_run_config

    def test_process_new_users_sensor():
        if False:
            while True:
                i = 10

        class FakeUsersAPI:

            def fetch_users(self) -> List[str]:
                if False:
                    print('Hello World!')
                return ['1', '2', '3']
        context = build_sensor_context()
        run_requests = process_new_users_sensor(context, users_api=FakeUsersAPI())
        assert len(run_requests) == 3

def new_resource_on_schedule() -> None:
    if False:
        i = 10
        return i + 15
    from dagster import schedule, ScheduleEvaluationContext, ConfigurableResource, job, RunRequest, RunConfig, Definitions
    from datetime import datetime
    from typing import List

    class DateFormatter(ConfigurableResource):
        format: str

        def strftime(self, dt: datetime) -> str:
            if False:
                while True:
                    i = 10
            return dt.strftime(self.format)

    @job
    def process_data():
        if False:
            print('Hello World!')
        ...

    @schedule(job=process_data, cron_schedule='* * * * *')
    def process_data_schedule(context: ScheduleEvaluationContext, date_formatter: DateFormatter):
        if False:
            for i in range(10):
                print('nop')
        formatted_date = date_formatter.strftime(context.scheduled_execution_time)
        return RunRequest(run_key=None, tags={'date': formatted_date})
    defs = Definitions(jobs=[process_data], schedules=[process_data_schedule], resources={'date_formatter': DateFormatter(format='%Y-%m-%d')})
    from dagster import build_schedule_context, validate_run_config

    def test_process_data_schedule():
        if False:
            i = 10
            return i + 15
        context = build_schedule_context(scheduled_execution_time=datetime.datetime(2020, 1, 1))
        run_request = process_data_schedule(context, date_formatter=DateFormatter(format='%Y-%m-%d'))
        assert run_request.run_config['ops']['fetch_data']['config']['date'] == '2020-01-01'