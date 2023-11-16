import logging
from contextlib import contextmanager
import pytest
from dagster._utils.log import define_structured_logger
from dagster._utils.test import create_test_pipeline_execution_context

@contextmanager
def construct_structured_logger(constructor=lambda x: x):
    if False:
        for i in range(10):
            print('nop')
    messages = []

    def _append_message(logger_message):
        if False:
            while True:
                i = 10
        messages.append(constructor(logger_message))
    logger_def = define_structured_logger('some_name', _append_message, level=logging.DEBUG)
    yield (logger_def, messages)

def test_structured_logger_in_context():
    if False:
        print('Hello World!')
    with construct_structured_logger() as (logger, messages):
        context = create_test_pipeline_execution_context(logger_defs={'structured_log': logger})
        context.log.debug('from_context', extra={'foo': 2})
        assert len(messages) == 1
        message = messages[0]
        assert message.name == 'some_name'
        assert message.level == logging.DEBUG
        assert message.record.__dict__['foo'] == 2
        assert message.meta['orig_message'] == 'from_context'

def test_structured_logger_in_context_with_bad_log_level():
    if False:
        print('Hello World!')
    messages = []

    def _append_message(logger_message):
        if False:
            for i in range(10):
                print('nop')
        messages.append(logger_message)
    logger = define_structured_logger('some_name', _append_message, level=logging.DEBUG)
    context = create_test_pipeline_execution_context(logger_defs={'structured_logger': logger})
    with pytest.raises(AttributeError):
        context.log.gargle('from_context')