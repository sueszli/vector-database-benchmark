import re
import pandas as pd
import pandera as pa
import pytest
from dagster import DagsterType, TypeCheck, check_dagster_type
from dagster._core.definitions.metadata import TableSchemaMetadataValue
from dagster._core.definitions.metadata.table import TableColumn, TableColumnConstraints, TableConstraints, TableSchema
from dagster_pandera import pandera_schema_to_dagster_type
from pandera.api.pandas.model_config import BaseConfig
DATA_OK = {'a': [1, 4, 0, 10, 9], 'b': [-1.3, -1.4, -2.9, -10.1, -20.4], 'c': ['value_1', 'value_2', 'value_3', 'value_2', 'value_1']}

@pytest.fixture
def dataframe():
    if False:
        print('Hello World!')
    return pd.DataFrame(DATA_OK)

def sample_dataframe_schema(**kwargs):
    if False:
        return 10
    return pa.DataFrameSchema({'a': pa.Column(int, checks=pa.Check.le(10), description='a desc'), 'b': pa.Column(float, checks=pa.Check.lt(-1.2), description='b desc'), 'c': pa.Column(str, description='c desc', checks=[pa.Check.str_startswith('value_'), pa.Check(lambda s: s.str.split('_', expand=True).shape[1] == 2, description='Two words separated by underscore.')])}, checks=[pa.Check(lambda df: df['a'].sum() > df['b'].sum(), description='sum(a) > sum(b).')], **kwargs)

def make_schema_model_config(**config_attrs):
    if False:
        i = 10
        return i + 15

    class Config(BaseConfig):
        pass
    for (k, v) in config_attrs.items():
        setattr(Config, k, v)
    return Config

def sample_schema_model(**config_attrs):
    if False:
        i = 10
        return i + 15

    class SampleSchemaModel(pa.SchemaModel):
        a: pa.typing.Series[int] = pa.Field(le=10, description='a desc')
        b: pa.typing.Series[float] = pa.Field(lt=-1.2, description='b desc')
        c: pa.typing.Series[str] = pa.Field(str_startswith='value_', description='c desc')

        @pa.check('c')
        def c_check(cls, series: pa.typing.Series[str]) -> pa.typing.Series[bool]:
            if False:
                while True:
                    i = 10
            'Two words separated by underscore.'
            return series.str.split('_', expand=True).shape[1] == 2

        @pa.dataframe_check
        def a_gt_b(cls, df):
            if False:
                print('Hello World!')
            'sum(a) > sum(b).'
            return df['a'].sum() > df['b'].sum()
        Config = make_schema_model_config(**config_attrs)
    return SampleSchemaModel

@pytest.fixture(params=[sample_dataframe_schema, sample_schema_model], ids=['regular_schema', 'schema_model'])
def schema(request):
    if False:
        print('Hello World!')
    return request.param()

@pytest.fixture
def dagster_type():
    if False:
        print('Hello World!')
    return pandera_schema_to_dagster_type(sample_schema_model())

def test_pandera_schema_to_dagster_type(schema):
    if False:
        i = 10
        return i + 15
    dagster_type = pandera_schema_to_dagster_type(schema)
    assert isinstance(dagster_type, DagsterType)
    assert len(dagster_type.metadata) == 1
    schema_metadata = dagster_type.metadata['schema']
    assert isinstance(schema_metadata, TableSchemaMetadataValue)
    assert schema_metadata.schema == TableSchema(constraints=TableConstraints(other=['sum(a) > sum(b).']), columns=[TableColumn(name='a', type='int64', description='a desc', constraints=TableColumnConstraints(nullable=False, other=['<= 10'])), TableColumn(name='b', type='float64', description='b desc', constraints=TableColumnConstraints(nullable=False, other=['< -1.2'])), TableColumn(name='c', type='str', description='c desc', constraints=TableColumnConstraints(nullable=False, other=["str_startswith('value_')", 'Two words separated by underscore.']))])

def test_name_extraction():
    if False:
        i = 10
        return i + 15
    schema = sample_schema_model()
    assert pandera_schema_to_dagster_type(schema).key == schema.__name__
    schema = sample_schema_model(name='foo')
    assert pandera_schema_to_dagster_type(schema).key == 'foo'
    schema = sample_schema_model(title='foo', name='bar')
    assert pandera_schema_to_dagster_type(schema).key == 'foo'
    schema = sample_dataframe_schema()
    assert re.match('DagsterPanderaDataframe\\d+', pandera_schema_to_dagster_type(schema).key)
    schema = sample_dataframe_schema(title='foo', name='bar')
    assert pandera_schema_to_dagster_type(schema).key == 'foo'
    schema = sample_dataframe_schema(name='foo')
    assert pandera_schema_to_dagster_type(schema).key == 'foo'

def test_validate_ok(dagster_type, dataframe):
    if False:
        i = 10
        return i + 15
    result = check_dagster_type(dagster_type, dataframe)
    assert isinstance(result, TypeCheck)
    assert result.success

def test_validate_inv_bad_value(dagster_type, dataframe):
    if False:
        while True:
            i = 10
    dataframe.loc[0, 'a'] = 11
    result = check_dagster_type(dagster_type, dataframe)
    assert not result.success

def test_validate_inv_missing_column(dagster_type, dataframe):
    if False:
        print('Hello World!')
    dataframe.drop('a', axis=1, inplace=True)
    result = check_dagster_type(dagster_type, dataframe)
    assert not result.success