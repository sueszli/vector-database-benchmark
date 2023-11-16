from datetime import datetime
from unittest import mock
import pytest
from flask.ctx import AppContext
from pytest_mock import MockFixture
import superset.utils.database
from superset.exceptions import SupersetTemplateException
from superset.jinja_context import get_template_processor

def test_process_template(app_context: AppContext) -> None:
    if False:
        print('Hello World!')
    maindb = superset.utils.database.get_example_database()
    template = "SELECT '{{ 1+1 }}'"
    tp = get_template_processor(database=maindb)
    assert tp.process_template(template) == "SELECT '2'"

def test_get_template_kwarg(app_context: AppContext) -> None:
    if False:
        for i in range(10):
            print('nop')
    maindb = superset.utils.database.get_example_database()
    template = '{{ foo }}'
    tp = get_template_processor(database=maindb, foo='bar')
    assert tp.process_template(template) == 'bar'

def test_template_kwarg(app_context: AppContext) -> None:
    if False:
        i = 10
        return i + 15
    maindb = superset.utils.database.get_example_database()
    template = '{{ foo }}'
    tp = get_template_processor(database=maindb)
    assert tp.process_template(template, foo='bar') == 'bar'

def test_get_template_kwarg_dict(app_context: AppContext) -> None:
    if False:
        print('Hello World!')
    maindb = superset.utils.database.get_example_database()
    template = '{{ foo.bar }}'
    tp = get_template_processor(database=maindb, foo={'bar': 'baz'})
    assert tp.process_template(template) == 'baz'

def test_template_kwarg_dict(app_context: AppContext) -> None:
    if False:
        print('Hello World!')
    maindb = superset.utils.database.get_example_database()
    template = '{{ foo.bar }}'
    tp = get_template_processor(database=maindb)
    assert tp.process_template(template, foo={'bar': 'baz'}) == 'baz'

def test_get_template_kwarg_lambda(app_context: AppContext) -> None:
    if False:
        i = 10
        return i + 15
    maindb = superset.utils.database.get_example_database()
    template = '{{ foo() }}'
    tp = get_template_processor(database=maindb, foo=lambda : 'bar')
    with pytest.raises(SupersetTemplateException):
        tp.process_template(template)

def test_template_kwarg_lambda(app_context: AppContext) -> None:
    if False:
        i = 10
        return i + 15
    maindb = superset.utils.database.get_example_database()
    template = '{{ foo() }}'
    tp = get_template_processor(database=maindb)
    with pytest.raises(SupersetTemplateException):
        tp.process_template(template, foo=lambda : 'bar')

def test_get_template_kwarg_module(app_context: AppContext) -> None:
    if False:
        return 10
    maindb = superset.utils.database.get_example_database()
    template = '{{ dt(2017, 1, 1).isoformat() }}'
    tp = get_template_processor(database=maindb, dt=datetime)
    with pytest.raises(SupersetTemplateException):
        tp.process_template(template)

def test_template_kwarg_module(app_context: AppContext) -> None:
    if False:
        for i in range(10):
            print('nop')
    maindb = superset.utils.database.get_example_database()
    template = '{{ dt(2017, 1, 1).isoformat() }}'
    tp = get_template_processor(database=maindb)
    with pytest.raises(SupersetTemplateException):
        tp.process_template(template, dt=datetime)

def test_get_template_kwarg_nested_module(app_context: AppContext) -> None:
    if False:
        while True:
            i = 10
    maindb = superset.utils.database.get_example_database()
    template = '{{ foo.dt }}'
    tp = get_template_processor(database=maindb, foo={'dt': datetime})
    with pytest.raises(SupersetTemplateException):
        tp.process_template(template)

def test_template_kwarg_nested_module(app_context: AppContext) -> None:
    if False:
        i = 10
        return i + 15
    maindb = superset.utils.database.get_example_database()
    template = '{{ foo.dt }}'
    tp = get_template_processor(database=maindb)
    with pytest.raises(SupersetTemplateException):
        tp.process_template(template, foo={'bar': datetime})

def test_template_hive(app_context: AppContext, mocker: MockFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    lp_mock = mocker.patch('superset.jinja_context.HiveTemplateProcessor.latest_partition')
    lp_mock.return_value = 'the_latest'
    db = mock.Mock()
    db.backend = 'hive'
    template = "{{ hive.latest_partition('my_table') }}"
    tp = get_template_processor(database=db)
    assert tp.process_template(template) == 'the_latest'

def test_template_trino(app_context: AppContext, mocker: MockFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    lp_mock = mocker.patch('superset.jinja_context.TrinoTemplateProcessor.latest_partition')
    lp_mock.return_value = 'the_latest'
    db = mock.Mock()
    db.backend = 'trino'
    template = "{{ trino.latest_partition('my_table') }}"
    tp = get_template_processor(database=db)
    assert tp.process_template(template) == 'the_latest'
    template = "{{ presto.latest_partition('my_table') }}"
    tp = get_template_processor(database=db)
    assert tp.process_template(template) == 'the_latest'

def test_template_context_addons(app_context: AppContext, mocker: MockFixture) -> None:
    if False:
        i = 10
        return i + 15
    addons_mock = mocker.patch('superset.jinja_context.context_addons')
    addons_mock.return_value = {'datetime': datetime}
    maindb = superset.utils.database.get_example_database()
    template = "SELECT '{{ datetime(2017, 1, 1).isoformat() }}'"
    tp = get_template_processor(database=maindb)
    assert tp.process_template(template) == "SELECT '2017-01-01T00:00:00'"

def test_custom_process_template(app_context: AppContext, mocker: MockFixture) -> None:
    if False:
        while True:
            i = 10
    'Test macro defined in custom template processor works.'
    mock_dt = mocker.patch('tests.integration_tests.superset_test_custom_template_processors.datetime')
    mock_dt.utcnow = mock.Mock(return_value=datetime(1970, 1, 1))
    db = mock.Mock()
    db.backend = 'db_for_macros_testing'
    tp = get_template_processor(database=db)
    template = "SELECT '$DATE()'"
    assert tp.process_template(template) == f"SELECT '1970-01-01'"
    template = "SELECT '$DATE(1, 2)'"
    assert tp.process_template(template) == "SELECT '1970-01-02'"

def test_custom_get_template_kwarg(app_context: AppContext) -> None:
    if False:
        while True:
            i = 10
    'Test macro passed as kwargs when getting template processor\n    works in custom template processor.'
    db = mock.Mock()
    db.backend = 'db_for_macros_testing'
    template = '$foo()'
    tp = get_template_processor(database=db, foo=lambda : 'bar')
    assert tp.process_template(template) == 'bar'

def test_custom_template_kwarg(app_context: AppContext) -> None:
    if False:
        print('Hello World!')
    'Test macro passed as kwargs when processing template\n    works in custom template processor.'
    db = mock.Mock()
    db.backend = 'db_for_macros_testing'
    template = '$foo()'
    tp = get_template_processor(database=db)
    assert tp.process_template(template, foo=lambda : 'bar') == 'bar'

def test_custom_template_processors_overwrite(app_context: AppContext) -> None:
    if False:
        i = 10
        return i + 15
    'Test template processor for presto gets overwritten by custom one.'
    db = mock.Mock()
    db.backend = 'db_for_macros_testing'
    tp = get_template_processor(database=db)
    template = "SELECT '{{ datetime(2017, 1, 1).isoformat() }}'"
    assert tp.process_template(template) == template
    template = "SELECT '{{ DATE(1, 2) }}'"
    assert tp.process_template(template) == template

def test_custom_template_processors_ignored(app_context: AppContext) -> None:
    if False:
        while True:
            i = 10
    'Test custom template processor is ignored for a difference backend\n    database.'
    maindb = superset.utils.database.get_example_database()
    template = "SELECT '$DATE()'"
    tp = get_template_processor(database=maindb)
    assert tp.process_template(template) == template