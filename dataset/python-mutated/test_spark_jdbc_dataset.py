import pytest
from kedro.extras.datasets.spark import SparkJDBCDataSet
from kedro.io import DatasetError

@pytest.fixture
def spark_jdbc_args():
    if False:
        for i in range(10):
            print('nop')
    return {'url': 'dummy_url', 'table': 'dummy_table'}

@pytest.fixture
def spark_jdbc_args_credentials(spark_jdbc_args):
    if False:
        return 10
    args = spark_jdbc_args
    args.update({'credentials': {'user': 'dummy_user', 'password': 'dummy_pw'}})
    return args

@pytest.fixture
def spark_jdbc_args_credentials_with_none_password(spark_jdbc_args):
    if False:
        return 10
    args = spark_jdbc_args
    args.update({'credentials': {'user': 'dummy_user', 'password': None}})
    return args

@pytest.fixture
def spark_jdbc_args_save_load(spark_jdbc_args):
    if False:
        while True:
            i = 10
    args = spark_jdbc_args
    connection_properties = {'properties': {'driver': 'dummy_driver'}}
    args.update({'save_args': connection_properties, 'load_args': connection_properties})
    return args

def test_missing_url():
    if False:
        for i in range(10):
            print('nop')
    error_message = "'url' argument cannot be empty. Please provide a JDBC URL of the form 'jdbc:subprotocol:subname'."
    with pytest.raises(DatasetError, match=error_message):
        SparkJDBCDataSet(url=None, table='dummy_table')

def test_missing_table():
    if False:
        print('Hello World!')
    error_message = "'table' argument cannot be empty. Please provide the name of the table to load or save data to."
    with pytest.raises(DatasetError, match=error_message):
        SparkJDBCDataSet(url='dummy_url', table=None)

def test_save(mocker, spark_jdbc_args):
    if False:
        i = 10
        return i + 15
    mock_data = mocker.Mock()
    data_set = SparkJDBCDataSet(**spark_jdbc_args)
    data_set.save(mock_data)
    mock_data.write.jdbc.assert_called_with('dummy_url', 'dummy_table')

def test_save_credentials(mocker, spark_jdbc_args_credentials):
    if False:
        i = 10
        return i + 15
    mock_data = mocker.Mock()
    data_set = SparkJDBCDataSet(**spark_jdbc_args_credentials)
    data_set.save(mock_data)
    mock_data.write.jdbc.assert_called_with('dummy_url', 'dummy_table', properties={'user': 'dummy_user', 'password': 'dummy_pw'})

def test_save_args(mocker, spark_jdbc_args_save_load):
    if False:
        print('Hello World!')
    mock_data = mocker.Mock()
    data_set = SparkJDBCDataSet(**spark_jdbc_args_save_load)
    data_set.save(mock_data)
    mock_data.write.jdbc.assert_called_with('dummy_url', 'dummy_table', properties={'driver': 'dummy_driver'})

def test_except_bad_credentials(mocker, spark_jdbc_args_credentials_with_none_password):
    if False:
        for i in range(10):
            print('nop')
    pattern = "Credential property 'password' cannot be None(.+)"
    with pytest.raises(DatasetError, match=pattern):
        mock_data = mocker.Mock()
        data_set = SparkJDBCDataSet(**spark_jdbc_args_credentials_with_none_password)
        data_set.save(mock_data)

def test_load(mocker, spark_jdbc_args):
    if False:
        print('Hello World!')
    spark = mocker.patch.object(SparkJDBCDataSet, '_get_spark').return_value
    data_set = SparkJDBCDataSet(**spark_jdbc_args)
    data_set.load()
    spark.read.jdbc.assert_called_with('dummy_url', 'dummy_table')

def test_load_credentials(mocker, spark_jdbc_args_credentials):
    if False:
        return 10
    spark = mocker.patch.object(SparkJDBCDataSet, '_get_spark').return_value
    data_set = SparkJDBCDataSet(**spark_jdbc_args_credentials)
    data_set.load()
    spark.read.jdbc.assert_called_with('dummy_url', 'dummy_table', properties={'user': 'dummy_user', 'password': 'dummy_pw'})

def test_load_args(mocker, spark_jdbc_args_save_load):
    if False:
        i = 10
        return i + 15
    spark = mocker.patch.object(SparkJDBCDataSet, '_get_spark').return_value
    data_set = SparkJDBCDataSet(**spark_jdbc_args_save_load)
    data_set.load()
    spark.read.jdbc.assert_called_with('dummy_url', 'dummy_table', properties={'driver': 'dummy_driver'})