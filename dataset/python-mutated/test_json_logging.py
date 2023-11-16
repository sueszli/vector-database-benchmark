import json
import logging
from dagster._core.definitions.graph_definition import GraphDefinition
from dagster._core.execution.context.logger import InitLoggerContext
from dagster._utils.log import define_json_file_logger
from dagster._utils.test import create_test_pipeline_execution_context, get_temp_file_name

def setup_json_file_logger(tf_name, name='foo', level=logging.DEBUG):
    if False:
        while True:
            i = 10
    logger_def = define_json_file_logger(name, tf_name, level)
    init_logger_context = InitLoggerContext({}, logger_def, job_def=GraphDefinition(node_defs=[], name='test').to_job(logger_defs={'json': logger_def}), run_id='')
    return logger_def.logger_fn(init_logger_context)

def test_basic_logging():
    if False:
        while True:
            i = 10
    with get_temp_file_name() as tf_name:
        logger = setup_json_file_logger(tf_name)
        logger.debug('bar')
        data = list(parse_json_lines(tf_name))
    assert len(data) == 1
    assert data[0]['name'] == 'foo'
    assert data[0]['msg'] == 'bar'

def parse_json_lines(tf_name):
    if False:
        print('Hello World!')
    with open(tf_name, encoding='utf8') as f:
        for line in f:
            yield json.loads(line)

def test_no_double_write_diff_names():
    if False:
        print('Hello World!')
    with get_temp_file_name() as tf_name:
        foo_logger = setup_json_file_logger(tf_name)
        baaz_logger = setup_json_file_logger(tf_name, 'baaz')
        foo_logger.debug('foo message')
        baaz_logger.debug('baaz message')
        data = list(parse_json_lines(tf_name))
        assert len(data) == 2
        assert data[0]['name'] == 'foo'
        assert data[0]['msg'] == 'foo message'
        assert data[1]['name'] == 'baaz'
        assert data[1]['msg'] == 'baaz message'

def test_no_double_write_same_names():
    if False:
        return 10
    with get_temp_file_name() as tf_name:
        foo_logger_one = setup_json_file_logger(tf_name)
        foo_logger_two = setup_json_file_logger(tf_name, 'foo', logging.INFO)
        foo_logger_one.debug('logger one message')
        foo_logger_two.debug('logger two message')
        data = list(parse_json_lines(tf_name))
        assert len(data) == 1
        assert data[0]['name'] == 'foo'
        assert data[0]['msg'] == 'logger one message'

def test_write_dagster_meta():
    if False:
        print('Hello World!')
    with get_temp_file_name() as tf_name:
        execution_context = create_test_pipeline_execution_context(logger_defs={'json': define_json_file_logger('foo', tf_name, logging.DEBUG)})
        execution_context.log.debug('some_debug_message', extra={'context_key': 'context_value'})
        data = list(parse_json_lines(tf_name))
        assert len(data) == 1
        assert data[0]['name'] == 'foo'
        assert data[0]['orig_message'] == 'some_debug_message'
        assert data[0]['context_key'] == 'context_value'