from typing import Any
from unittest.mock import MagicMock
import pytest
from opentelemetry.trace import SpanKind
from pytest_mock import MockerFixture
import strawberry
from strawberry.extensions.tracing.opentelemetry import OpenTelemetryExtension, OpenTelemetryExtensionSync

@pytest.fixture
def global_tracer_mock(mocker: MockerFixture) -> MagicMock:
    if False:
        print('Hello World!')
    return mocker.patch('strawberry.extensions.tracing.opentelemetry.trace.get_tracer')

@strawberry.type
class Person:
    name: str = 'Jess'

@strawberry.type
class Query:

    @strawberry.field
    async def person(self) -> Person:
        return Person()

@pytest.mark.asyncio
async def test_opentelemetry_uses_global_tracer(global_tracer_mock):
    schema = strawberry.Schema(query=Query, extensions=[OpenTelemetryExtension])
    query = '\n        query {\n            person {\n                name\n            }\n        }\n    '
    await schema.execute(query)
    global_tracer_mock.assert_called_once_with('strawberry')

@pytest.mark.asyncio
async def test_opentelemetry_sync_uses_global_tracer(global_tracer_mock):
    schema = strawberry.Schema(query=Query, extensions=[OpenTelemetryExtensionSync])
    query = '\n        query {\n            person {\n                name\n            }\n        }\n    '
    await schema.execute(query)
    global_tracer_mock.assert_called_once_with('strawberry')

def _instrumentation_stages(mocker, query):
    if False:
        while True:
            i = 10
    return [mocker.call('GraphQL Query', kind=SpanKind.SERVER), mocker.call().set_attribute('component', 'graphql'), mocker.call().set_attribute('query', query), mocker.call('GraphQL Parsing', context=mocker.ANY), mocker.call().end(), mocker.call('GraphQL Validation', context=mocker.ANY), mocker.call().end(), mocker.call().end()]

@pytest.mark.asyncio
async def test_open_tracing(global_tracer_mock, mocker):
    schema = strawberry.Schema(query=Query, extensions=[OpenTelemetryExtension])
    query = '\n        query {\n            person {\n                name\n            }\n        }\n    '
    await schema.execute(query)
    global_tracer_mock.return_value.start_span.assert_has_calls(_instrumentation_stages(mocker, query))
    global_tracer_mock.return_value.start_as_current_span.assert_has_calls([mocker.call('GraphQL Resolving: person', context=mocker.ANY), mocker.call().__enter__(), mocker.call().__enter__().set_attribute('component', 'graphql'), mocker.call().__enter__().set_attribute('graphql.parentType', 'Query'), mocker.call().__enter__().set_attribute('graphql.path', 'person'), mocker.call().__exit__(None, None, None)])

@pytest.mark.asyncio
async def test_open_tracing_uses_operation_name(global_tracer_mock, mocker):
    schema = strawberry.Schema(query=Query, extensions=[OpenTelemetryExtension])
    query = '\n        query Example {\n            person {\n                name\n            }\n        }\n    '
    await schema.execute(query, operation_name='Example')
    global_tracer_mock.return_value.start_span.assert_has_calls([mocker.call('GraphQL Query: Example', kind=SpanKind.SERVER), *_instrumentation_stages(mocker, query)[1:]])

@pytest.mark.asyncio
async def test_open_tracing_gets_operation_name(global_tracer_mock, mocker):
    schema = strawberry.Schema(query=Query, extensions=[OpenTelemetryExtension])
    query = '\n        query Example {\n            person {\n                name\n            }\n        }\n    '
    tracers = []

    def generate_trace(*args: str, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        nonlocal tracers
        tracer = mocker.Mock()
        tracers.append(tracer)
        return tracer
    global_tracer_mock.return_value.start_span.side_effect = generate_trace
    await schema.execute(query)
    tracers[0].update_name.assert_has_calls([mocker.call('GraphQL Query: Example')])

@pytest.mark.asyncio
async def test_tracing_add_kwargs(global_tracer_mock, mocker):

    @strawberry.type
    class Query:

        @strawberry.field
        def hi(self, name: str) -> str:
            if False:
                return 10
            return f'Hi {name}'
    schema = strawberry.Schema(query=Query, extensions=[OpenTelemetryExtension])
    query = '\n        query {\n            hi(name: "Patrick")\n        }\n    '
    await schema.execute(query)
    global_tracer_mock.return_value.start_as_current_span.assert_has_calls([mocker.call().__enter__().set_attribute('graphql.parentType', 'Query'), mocker.call().__enter__().set_attribute('graphql.path', 'hi'), mocker.call().__enter__().set_attribute('graphql.param.name', 'Patrick')])

@pytest.mark.asyncio
async def test_tracing_filter_kwargs(global_tracer_mock, mocker):

    def arg_filter(kwargs, info):
        if False:
            i = 10
            return i + 15
        return {'name': '[...]'}

    @strawberry.type
    class Query:

        @strawberry.field
        def hi(self, name: str) -> str:
            if False:
                return 10
            return f'Hi {name}'
    schema = strawberry.Schema(query=Query, extensions=[OpenTelemetryExtension(arg_filter=arg_filter)])
    query = '\n        query {\n            hi(name: "Patrick")\n        }\n    '
    await schema.execute(query)
    global_tracer_mock.return_value.start_as_current_span.assert_has_calls([mocker.call().__enter__().set_attribute('graphql.parentType', 'Query'), mocker.call().__enter__().set_attribute('graphql.path', 'hi'), mocker.call().__enter__().set_attribute('graphql.param.name', '[...]')])