from typing import Dict, List

class Engine:

    def execute(self, query: str):
        if False:
            print('Hello World!')
        ...

def get_engine(connection_url: str) -> Engine:
    if False:
        for i in range(10):
            print('nop')
    return Engine()

def basic_resource_config() -> None:
    if False:
        return 10
    from dagster import op, ConfigurableResource

    class MyDatabaseResource(ConfigurableResource):
        connection_url: str

        def query(self, query: str):
            if False:
                while True:
                    i = 10
            return get_engine(self.connection_url).execute(query)

def permissive_schema_config() -> None:
    if False:
        print('Hello World!')
    from dagster import asset, PermissiveConfig
    from typing import Optional
    import requests

    class FilterConfig(PermissiveConfig):
        title: Optional[str] = None
        description: Optional[str] = None

    @asset
    def filtered_listings(config: FilterConfig):
        if False:
            while True:
                i = 10
        url_params = config.dict()
        return requests.get('https://my-api.com/listings', params=url_params).json()
    filtered_listings(FilterConfig(title='hotel', beds=4))

def execute_with_config() -> None:
    if False:
        for i in range(10):
            print('nop')
    from dagster import op, Config

    class MyOpConfig(Config):
        person_name: str

    @op
    def print_greeting(config: MyOpConfig):
        if False:
            while True:
                i = 10
        print(f'hello {config.person_name}')
    from dagster import asset, Config

    class MyAssetConfig(Config):
        person_name: str

    @asset
    def greeting(config: MyAssetConfig) -> str:
        if False:
            print('Hello World!')
        return f'hello {config.person_name}'
    from dagster import job, materialize, op, RunConfig

    @job
    def greeting_job():
        if False:
            i = 10
            return i + 15
        print_greeting()
    job_result = greeting_job.execute_in_process(run_config=RunConfig({'print_greeting': MyOpConfig(person_name='Alice')}))
    asset_result = materialize([greeting], run_config=RunConfig({'greeting': MyAssetConfig(person_name='Alice')}))
    from dagster import job, materialize, op, RunConfig, EnvVar
    job_result = greeting_job.execute_in_process(run_config=RunConfig({'print_greeting': MyOpConfig(person_name=EnvVar('PERSON_NAME'))}))
    asset_result = materialize([greeting], run_config=RunConfig({'greeting': MyAssetConfig(person_name=EnvVar('PERSON_NAME'))}))

def basic_data_structures_config() -> None:
    if False:
        i = 10
        return i + 15
    from dagster import Config, materialize, asset, RunConfig
    from typing import List, Dict

    class MyDataStructuresConfig(Config):
        user_names: List[str]
        user_scores: Dict[str, int]

    @asset
    def scoreboard(config: MyDataStructuresConfig):
        if False:
            i = 10
            return i + 15
        ...
    result = materialize([scoreboard], run_config=RunConfig({'scoreboard': MyDataStructuresConfig(user_names=['Alice', 'Bob'], user_scores={'Alice': 10, 'Bob': 20})}))

def nested_schema_config() -> None:
    if False:
        while True:
            i = 10
    from dagster import asset, materialize, Config, RunConfig
    from typing import Dict

    class UserData(Config):
        age: int
        email: str
        profile_picture_url: str

    class MyNestedConfig(Config):
        user_data: Dict[str, UserData]

    @asset
    def average_age(config: MyNestedConfig):
        if False:
            i = 10
            return i + 15
        ...
    result = materialize([average_age], run_config=RunConfig({'average_age': MyNestedConfig(user_data={'Alice': UserData(age=10, email='alice@gmail.com', profile_picture_url=...), 'Bob': UserData(age=20, email='bob@gmail.com', profile_picture_url=...)})}))

def union_schema_config() -> None:
    if False:
        for i in range(10):
            print('nop')
    from dagster import asset, materialize, Config, RunConfig
    from pydantic import Field
    from typing import Union
    from typing_extensions import Literal

    class Cat(Config):
        pet_type: Literal['cat'] = 'cat'
        meows: int

    class Dog(Config):
        pet_type: Literal['dog'] = 'dog'
        barks: float

    class ConfigWithUnion(Config):
        pet: Union[Cat, Dog] = Field(discriminator='pet_type')

    @asset
    def pet_stats(config: ConfigWithUnion):
        if False:
            while True:
                i = 10
        if isinstance(config.pet, Cat):
            return f'Cat meows {config.pet.meows} times'
        else:
            return f'Dog barks {config.pet.barks} times'
    result = materialize([pet_stats], run_config=RunConfig({'pet_stats': ConfigWithUnion(pet=Cat(meows=10))}))

def metadata_config() -> None:
    if False:
        i = 10
        return i + 15
    from dagster import Config
    from pydantic import Field

    class MyMetadataConfig(Config):
        person_name: str = Field(description='The name of the person to greet')
        age: int = Field(gt=0, lt=100, description='The age of the person to greet')
    MyMetadataConfig(person_name='Alice', age=200)

def optional_config() -> None:
    if False:
        for i in range(10):
            print('nop')
    from typing import Optional
    from dagster import asset, Config, materialize, RunConfig
    from pydantic import Field

    class MyAssetConfig(Config):
        person_name: Optional[str] = None
        greeting_phrase: str = Field(default='hello', description='The greeting phrase to use.')

    @asset
    def greeting(config: MyAssetConfig) -> str:
        if False:
            i = 10
            return i + 15
        if config.person_name:
            return f'{config.greeting_phrase} {config.person_name}'
        else:
            return config.greeting_phrase
    asset_result = materialize([greeting], run_config=RunConfig({'greeting': MyAssetConfig()}))

def execute_with_bad_config() -> None:
    if False:
        print('Hello World!')
    from dagster import op, job, materialize, Config, RunConfig

    class MyOpConfig(Config):
        person_name: str

    @op
    def print_greeting(config: MyOpConfig):
        if False:
            for i in range(10):
                print('nop')
        print(f'hello {config.person_name}')
    from dagster import asset, Config

    class MyAssetConfig(Config):
        person_name: str

    @asset
    def greeting(config: MyAssetConfig) -> str:
        if False:
            return 10
        return f'hello {config.person_name}'

    @job
    def greeting_job():
        if False:
            i = 10
            return i + 15
        print_greeting()
    op_result = greeting_job.execute_in_process(run_config=RunConfig({'print_greeting': MyOpConfig(nonexistent_config_value=1)}))
    asset_result = materialize([greeting], run_config=RunConfig({'greeting': MyAssetConfig(nonexistent_config_value=1)}))

def enum_schema_config() -> None:
    if False:
        while True:
            i = 10
    from dagster import Config, RunConfig, op, job
    from enum import Enum

    class UserPermissions(Enum):
        GUEST = 'guest'
        MEMBER = 'member'
        ADMIN = 'admin'

    class ProcessUsersConfig(Config):
        users_list: Dict[str, UserPermissions]

    @op
    def process_users(config: ProcessUsersConfig):
        if False:
            return 10
        for (user, permission) in config.users_list.items():
            if permission == UserPermissions.ADMIN:
                print(f'{user} is an admin')

    @job
    def process_users_job():
        if False:
            return 10
        process_users()
    op_result = process_users_job.execute_in_process(run_config=RunConfig({'process_users': ProcessUsersConfig(users_list={'Bob': UserPermissions.GUEST, 'Alice': UserPermissions.ADMIN})}))

def validated_schema_config() -> None:
    if False:
        for i in range(10):
            print('nop')
    from dagster import Config, RunConfig, op, job
    from pydantic import validator

    class UserConfig(Config):
        name: str
        username: str

        @validator('name')
        def name_must_contain_space(cls, v):
            if False:
                while True:
                    i = 10
            if ' ' not in v:
                raise ValueError('must contain a space')
            return v.title()

        @validator('username')
        def username_alphanumeric(cls, v):
            if False:
                return 10
            assert v.isalnum(), 'must be alphanumeric'
            return v
    executed = {}

    @op
    def greet_user(config: UserConfig) -> None:
        if False:
            i = 10
            return i + 15
        print(f'Hello {config.name}!')
        executed['greet_user'] = True

    @job
    def greet_user_job() -> None:
        if False:
            print('Hello World!')
        greet_user()
    op_result = greet_user_job.execute_in_process(run_config=RunConfig({'greet_user': UserConfig(name='Alice Smith', username='alice123')}))
    op_result = greet_user_job.execute_in_process(run_config=RunConfig({'greet_user': UserConfig(name='John', username='johndoe44')}))

def required_config() -> None:
    if False:
        print('Hello World!')
    from typing import Optional, Callable
    from dagster import asset, Config
    from pydantic import Field

    class MyAssetConfig(Config):
        person_first_name: Optional[str] = ...
        person_last_name: Optional[Callable] = Field(default=..., description='The last name of the person to greet')

    @asset
    def goodbye(config: MyAssetConfig) -> str:
        if False:
            i = 10
            return i + 15
        full_name = f'{config.person_first_name} {config.person_last_name}'.strip()
        if full_name:
            return f'Goodbye, {full_name}'
        else:
            return 'Goodbye'
    goodbye(MyAssetConfig())
    goodbye(MyAssetConfig(person_first_name='Alice', person_last_name=None))