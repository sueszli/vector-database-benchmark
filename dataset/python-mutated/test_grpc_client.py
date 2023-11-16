import uuid
from unittest.mock import AsyncMock
import pytest
from api.v1 import api_pb2
from grpc_client import GRPCClient

@pytest.fixture()
def mock_run_code(mocker):
    if False:
        for i in range(10):
            print('nop')
    async_mock = AsyncMock(return_value=str(uuid.uuid4()))
    mocker.patch('grpc_client.GRPCClient.run_code', side_effect=async_mock)
    return async_mock

@pytest.fixture()
def mock_check_status(mocker):
    if False:
        while True:
            i = 10
    async_mock = AsyncMock(return_value=api_pb2.STATUS_FINISHED)
    mocker.patch('grpc_client.GRPCClient.check_status', side_effect=async_mock)
    return async_mock

@pytest.fixture()
def mock_get_run_error(mocker):
    if False:
        while True:
            i = 10
    async_mock = AsyncMock(return_value='MOCK_ERROR')
    mocker.patch('grpc_client.GRPCClient.get_run_error', side_effect=async_mock)
    return async_mock

@pytest.fixture()
def mock_get_run_output(mocker):
    if False:
        i = 10
        return i + 15
    async_mock = AsyncMock(return_value='MOCK_RUN_OUTPUT')
    mocker.patch('grpc_client.GRPCClient.get_run_output', side_effect=async_mock)
    return async_mock

@pytest.fixture()
def mock_get_compile_output(mocker):
    if False:
        return 10
    async_mock = AsyncMock(return_value='MOCK_COMPILE_OUTPUT')
    mocker.patch('grpc_client.GRPCClient.get_compile_output', side_effect=async_mock)
    return async_mock

class TestGRPCClient:

    @pytest.mark.asyncio
    async def test_run_code(self, mock_run_code):
        result = await GRPCClient().run_code('', api_pb2.SDK_GO, '', [])
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_check_status(self, mock_check_status):
        result = await GRPCClient().check_status(str(uuid.uuid4()))
        assert result == api_pb2.STATUS_FINISHED

    @pytest.mark.asyncio
    async def test_get_run_error(self, mock_get_run_error):
        result = await GRPCClient().get_run_error(str(uuid.uuid4()))
        assert result == 'MOCK_ERROR'

    @pytest.mark.asyncio
    async def test_get_run_output(self, mock_get_run_output):
        result = await GRPCClient().get_run_output(str(uuid.uuid4()))
        assert result == 'MOCK_RUN_OUTPUT'

    @pytest.mark.asyncio
    async def test_get_compile_output(self, mock_get_compile_output):
        result = await GRPCClient().get_compile_output(str(uuid.uuid4()))
        assert result == 'MOCK_COMPILE_OUTPUT'