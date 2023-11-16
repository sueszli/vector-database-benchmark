from __future__ import annotations
import datetime
import operator
import os
from typing import TYPE_CHECKING
from unittest import mock
from unittest.mock import MagicMock
import pytest
from airflow.configuration import conf
from airflow.models.dagrun import DagRun, DagRunType
from airflow.models.taskinstance import TaskInstance
from airflow.models.taskinstancekey import TaskInstanceKey
from airflow.models.xcom import BaseXCom, XCom, resolve_xcom_backend
from airflow.operators.empty import EmptyOperator
from airflow.settings import json
from airflow.utils import timezone
from airflow.utils.session import create_session
from airflow.utils.xcom import XCOM_RETURN_KEY
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test
if TYPE_CHECKING:
    from sqlalchemy.orm import Session

class CustomXCom(BaseXCom):
    orm_deserialize_value = mock.Mock()

@pytest.fixture(autouse=True)
def reset_db():
    if False:
        for i in range(10):
            print('nop')
    'Reset XCom entries.'
    with create_session() as session:
        session.query(DagRun).delete()
        session.query(XCom).delete()

@pytest.fixture()
def task_instance_factory(request, session: Session):
    if False:
        return 10

    def func(*, dag_id, task_id, execution_date):
        if False:
            i = 10
            return i + 15
        run_id = DagRun.generate_run_id(DagRunType.SCHEDULED, execution_date)
        run = DagRun(dag_id=dag_id, run_type=DagRunType.SCHEDULED, run_id=run_id, execution_date=execution_date)
        session.add(run)
        ti = TaskInstance(EmptyOperator(task_id=task_id), run_id=run_id)
        ti.dag_id = dag_id
        session.add(ti)
        session.commit()

        def cleanup_database():
            if False:
                return 10
            session.query(DagRun).filter_by(id=run.id).delete()
            session.commit()
        request.addfinalizer(cleanup_database)
        return ti
    return func

@pytest.fixture()
def task_instance(task_instance_factory):
    if False:
        i = 10
        return i + 15
    return task_instance_factory(dag_id='dag', task_id='task_1', execution_date=timezone.datetime(2021, 12, 3, 4, 56))

@pytest.fixture()
def task_instances(session, task_instance):
    if False:
        i = 10
        return i + 15
    ti2 = TaskInstance(EmptyOperator(task_id='task_2'), run_id=task_instance.run_id)
    ti2.dag_id = task_instance.dag_id
    session.add(ti2)
    session.commit()
    return (task_instance, ti2)

class TestXCom:

    @conf_vars({('core', 'xcom_backend'): 'tests.models.test_xcom.CustomXCom'})
    def test_resolve_xcom_class(self):
        if False:
            while True:
                i = 10
        cls = resolve_xcom_backend()
        assert issubclass(cls, CustomXCom)

    @conf_vars({('core', 'xcom_backend'): '', ('core', 'enable_xcom_pickling'): 'False'})
    def test_resolve_xcom_class_fallback_to_basexcom(self):
        if False:
            while True:
                i = 10
        cls = resolve_xcom_backend()
        assert issubclass(cls, BaseXCom)
        assert cls.serialize_value([1]) == b'[1]'

    @conf_vars({('core', 'enable_xcom_pickling'): 'False'})
    @conf_vars({('core', 'xcom_backend'): 'to be removed'})
    def test_resolve_xcom_class_fallback_to_basexcom_no_config(self):
        if False:
            i = 10
            return i + 15
        conf.remove_option('core', 'xcom_backend')
        cls = resolve_xcom_backend()
        assert issubclass(cls, BaseXCom)
        assert cls.serialize_value([1]) == b'[1]'

    def test_xcom_deserialize_with_json_to_pickle_switch(self, task_instance, session):
        if False:
            return 10
        ti_key = TaskInstanceKey(dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id)
        with conf_vars({('core', 'enable_xcom_pickling'): 'False'}):
            XCom.set(key='xcom_test3', value={'key': 'value'}, dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, session=session)
        with conf_vars({('core', 'enable_xcom_pickling'): 'True'}):
            ret_value = XCom.get_value(key='xcom_test3', ti_key=ti_key, session=session)
        assert ret_value == {'key': 'value'}

    def test_xcom_deserialize_with_pickle_to_json_switch(self, task_instance, session):
        if False:
            for i in range(10):
                print('nop')
        with conf_vars({('core', 'enable_xcom_pickling'): 'True'}):
            XCom.set(key='xcom_test3', value={'key': 'value'}, dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, session=session)
        with conf_vars({('core', 'enable_xcom_pickling'): 'False'}):
            ret_value = XCom.get_one(key='xcom_test3', dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, session=session)
        assert ret_value == {'key': 'value'}

    @conf_vars({('core', 'xcom_enable_pickling'): 'False'})
    def test_xcom_disable_pickle_type_fail_on_non_json(self, task_instance, session):
        if False:
            i = 10
            return i + 15

        class PickleRce:

            def __reduce__(self):
                if False:
                    print('Hello World!')
                return (os.system, ('ls -alt',))
        with pytest.raises(TypeError):
            XCom.set(key='xcom_test3', value=PickleRce(), dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, session=session)

    @mock.patch('airflow.models.xcom.XCom.orm_deserialize_value')
    def test_xcom_init_on_load_uses_orm_deserialize_value(self, mock_orm_deserialize):
        if False:
            return 10
        instance = BaseXCom(key='key', value='value', timestamp=timezone.utcnow(), execution_date=timezone.utcnow(), task_id='task_id', dag_id='dag_id')
        instance.init_on_load()
        mock_orm_deserialize.assert_called_once_with()

    @conf_vars({('core', 'xcom_backend'): 'tests.models.test_xcom.CustomXCom'})
    def test_get_one_custom_backend_no_use_orm_deserialize_value(self, task_instance, session):
        if False:
            for i in range(10):
                print('nop')
        'Test that XCom.get_one does not call orm_deserialize_value'
        XCom = resolve_xcom_backend()
        XCom.set(key=XCOM_RETURN_KEY, value={'key': 'value'}, dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, session=session)
        value = XCom.get_one(dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, session=session)
        assert value == {'key': 'value'}
        XCom.orm_deserialize_value.assert_not_called()

    @conf_vars({('core', 'enable_xcom_pickling'): 'False'})
    @mock.patch('airflow.models.xcom.conf.getimport')
    def test_set_serialize_call_old_signature(self, get_import, task_instance):
        if False:
            return 10
        '\n        When XCom.serialize_value takes only param ``value``, other kwargs should be ignored.\n        '
        serialize_watcher = MagicMock()

        class OldSignatureXCom(BaseXCom):

            @staticmethod
            def serialize_value(value, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                serialize_watcher(value=value, **kwargs)
                return json.dumps(value).encode('utf-8')
        get_import.return_value = OldSignatureXCom
        XCom = resolve_xcom_backend()
        XCom.set(key=XCOM_RETURN_KEY, value={'my_xcom_key': 'my_xcom_value'}, dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id)
        serialize_watcher.assert_called_once_with(value={'my_xcom_key': 'my_xcom_value'})

    @conf_vars({('core', 'enable_xcom_pickling'): 'False'})
    @mock.patch('airflow.models.xcom.conf.getimport')
    def test_set_serialize_call_current_signature(self, get_import, task_instance):
        if False:
            print('Hello World!')
        '\n        When XCom.serialize_value includes params execution_date, key, dag_id, task_id and run_id,\n        then XCom.set should pass all of them.\n        '
        serialize_watcher = MagicMock()

        class CurrentSignatureXCom(BaseXCom):

            @staticmethod
            def serialize_value(value, key=None, dag_id=None, task_id=None, run_id=None, map_index=None):
                if False:
                    print('Hello World!')
                serialize_watcher(value=value, key=key, dag_id=dag_id, task_id=task_id, run_id=run_id, map_index=map_index)
                return json.dumps(value).encode('utf-8')
        get_import.return_value = CurrentSignatureXCom
        XCom = resolve_xcom_backend()
        XCom.set(key=XCOM_RETURN_KEY, value={'my_xcom_key': 'my_xcom_value'}, dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, map_index=-1)
        serialize_watcher.assert_called_once_with(key=XCOM_RETURN_KEY, value={'my_xcom_key': 'my_xcom_value'}, dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, map_index=-1)

@pytest.fixture(params=[pytest.param('true', id='enable_xcom_pickling=true'), pytest.param('false', id='enable_xcom_pickling=false')])
def setup_xcom_pickling(request):
    if False:
        return 10
    with conf_vars({('core', 'enable_xcom_pickling'): str(request.param)}):
        yield

@pytest.fixture()
def push_simple_json_xcom(session):
    if False:
        while True:
            i = 10

    def func(*, ti: TaskInstance, key: str, value):
        if False:
            for i in range(10):
                print('nop')
        return XCom.set(key=key, value=value, dag_id=ti.dag_id, task_id=ti.task_id, run_id=ti.run_id, session=session)
    return func

@pytest.mark.usefixtures('setup_xcom_pickling')
class TestXComGet:

    @pytest.fixture()
    def setup_for_xcom_get_one(self, task_instance, push_simple_json_xcom):
        if False:
            print('Hello World!')
        push_simple_json_xcom(ti=task_instance, key='xcom_1', value={'key': 'value'})

    @pytest.mark.usefixtures('setup_for_xcom_get_one')
    def test_xcom_get_one(self, session, task_instance):
        if False:
            i = 10
            return i + 15
        stored_value = XCom.get_one(key='xcom_1', dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, session=session)
        assert stored_value == {'key': 'value'}

    @pytest.mark.usefixtures('setup_for_xcom_get_one')
    def test_xcom_get_one_with_execution_date(self, session, task_instance):
        if False:
            return 10
        with pytest.deprecated_call():
            stored_value = XCom.get_one(key='xcom_1', dag_id=task_instance.dag_id, task_id=task_instance.task_id, execution_date=task_instance.execution_date, session=session)
        assert stored_value == {'key': 'value'}

    @pytest.fixture()
    def tis_for_xcom_get_one_from_prior_date(self, task_instance_factory, push_simple_json_xcom):
        if False:
            while True:
                i = 10
        date1 = timezone.datetime(2021, 12, 3, 4, 56)
        ti1 = task_instance_factory(dag_id='dag', execution_date=date1, task_id='task_1')
        ti2 = task_instance_factory(dag_id='dag', execution_date=date1 + datetime.timedelta(days=1), task_id='task_1')
        push_simple_json_xcom(ti=ti1, key='xcom_1', value={'key': 'value'})
        return (ti1, ti2)

    def test_xcom_get_one_from_prior_date(self, session, tis_for_xcom_get_one_from_prior_date):
        if False:
            print('Hello World!')
        (_, ti2) = tis_for_xcom_get_one_from_prior_date
        retrieved_value = XCom.get_one(run_id=ti2.run_id, key='xcom_1', task_id='task_1', dag_id='dag', include_prior_dates=True, session=session)
        assert retrieved_value == {'key': 'value'}

    def test_xcom_get_one_from_prior_with_execution_date(self, session, tis_for_xcom_get_one_from_prior_date):
        if False:
            print('Hello World!')
        (_, ti2) = tis_for_xcom_get_one_from_prior_date
        with pytest.deprecated_call():
            retrieved_value = XCom.get_one(execution_date=ti2.execution_date, key='xcom_1', task_id='task_1', dag_id='dag', include_prior_dates=True, session=session)
        assert retrieved_value == {'key': 'value'}

    @pytest.fixture()
    def setup_for_xcom_get_many_single_argument_value(self, task_instance, push_simple_json_xcom):
        if False:
            print('Hello World!')
        push_simple_json_xcom(ti=task_instance, key='xcom_1', value={'key': 'value'})

    @pytest.mark.usefixtures('setup_for_xcom_get_many_single_argument_value')
    def test_xcom_get_many_single_argument_value(self, session, task_instance):
        if False:
            for i in range(10):
                print('nop')
        stored_xcoms = XCom.get_many(key='xcom_1', dag_ids=task_instance.dag_id, task_ids=task_instance.task_id, run_id=task_instance.run_id, session=session).all()
        assert len(stored_xcoms) == 1
        assert stored_xcoms[0].key == 'xcom_1'
        assert stored_xcoms[0].value == {'key': 'value'}

    @pytest.mark.usefixtures('setup_for_xcom_get_many_single_argument_value')
    def test_xcom_get_many_single_argument_value_with_execution_date(self, session, task_instance):
        if False:
            while True:
                i = 10
        with pytest.deprecated_call():
            stored_xcoms = XCom.get_many(execution_date=task_instance.execution_date, key='xcom_1', dag_ids=task_instance.dag_id, task_ids=task_instance.task_id, session=session).all()
        assert len(stored_xcoms) == 1
        assert stored_xcoms[0].key == 'xcom_1'
        assert stored_xcoms[0].value == {'key': 'value'}

    @pytest.fixture()
    def setup_for_xcom_get_many_multiple_tasks(self, task_instances, push_simple_json_xcom):
        if False:
            i = 10
            return i + 15
        (ti1, ti2) = task_instances
        push_simple_json_xcom(ti=ti1, key='xcom_1', value={'key1': 'value1'})
        push_simple_json_xcom(ti=ti2, key='xcom_1', value={'key2': 'value2'})

    @pytest.mark.usefixtures('setup_for_xcom_get_many_multiple_tasks')
    def test_xcom_get_many_multiple_tasks(self, session, task_instance):
        if False:
            for i in range(10):
                print('nop')
        stored_xcoms = XCom.get_many(key='xcom_1', dag_ids=task_instance.dag_id, task_ids=['task_1', 'task_2'], run_id=task_instance.run_id, session=session)
        sorted_values = [x.value for x in sorted(stored_xcoms, key=operator.attrgetter('task_id'))]
        assert sorted_values == [{'key1': 'value1'}, {'key2': 'value2'}]

    @pytest.mark.usefixtures('setup_for_xcom_get_many_multiple_tasks')
    def test_xcom_get_many_multiple_tasks_with_execution_date(self, session, task_instance):
        if False:
            print('Hello World!')
        with pytest.deprecated_call():
            stored_xcoms = XCom.get_many(execution_date=task_instance.execution_date, key='xcom_1', dag_ids=task_instance.dag_id, task_ids=['task_1', 'task_2'], session=session)
        sorted_values = [x.value for x in sorted(stored_xcoms, key=operator.attrgetter('task_id'))]
        assert sorted_values == [{'key1': 'value1'}, {'key2': 'value2'}]

    @pytest.fixture()
    def tis_for_xcom_get_many_from_prior_dates(self, task_instance_factory, push_simple_json_xcom):
        if False:
            return 10
        date1 = timezone.datetime(2021, 12, 3, 4, 56)
        date2 = date1 + datetime.timedelta(days=1)
        ti1 = task_instance_factory(dag_id='dag', task_id='task_1', execution_date=date1)
        ti2 = task_instance_factory(dag_id='dag', task_id='task_1', execution_date=date2)
        push_simple_json_xcom(ti=ti1, key='xcom_1', value={'key1': 'value1'})
        push_simple_json_xcom(ti=ti2, key='xcom_1', value={'key2': 'value2'})
        return (ti1, ti2)

    def test_xcom_get_many_from_prior_dates(self, session, tis_for_xcom_get_many_from_prior_dates):
        if False:
            return 10
        (ti1, ti2) = tis_for_xcom_get_many_from_prior_dates
        stored_xcoms = XCom.get_many(run_id=ti2.run_id, key='xcom_1', dag_ids='dag', task_ids='task_1', include_prior_dates=True, session=session)
        assert [x.value for x in stored_xcoms] == [{'key2': 'value2'}, {'key1': 'value1'}]
        assert [x.execution_date for x in stored_xcoms] == [ti2.execution_date, ti1.execution_date]

    def test_xcom_get_many_from_prior_dates_with_execution_date(self, session, tis_for_xcom_get_many_from_prior_dates):
        if False:
            for i in range(10):
                print('nop')
        (ti1, ti2) = tis_for_xcom_get_many_from_prior_dates
        with pytest.deprecated_call():
            stored_xcoms = XCom.get_many(execution_date=ti2.execution_date, key='xcom_1', dag_ids='dag', task_ids='task_1', include_prior_dates=True, session=session)
        assert [x.value for x in stored_xcoms] == [{'key2': 'value2'}, {'key1': 'value1'}]
        assert [x.execution_date for x in stored_xcoms] == [ti2.execution_date, ti1.execution_date]

@pytest.mark.usefixtures('setup_xcom_pickling')
class TestXComSet:

    def test_xcom_set(self, session, task_instance):
        if False:
            print('Hello World!')
        XCom.set(key='xcom_1', value={'key': 'value'}, dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, session=session)
        stored_xcoms = session.query(XCom).all()
        assert stored_xcoms[0].key == 'xcom_1'
        assert stored_xcoms[0].value == {'key': 'value'}
        assert stored_xcoms[0].dag_id == 'dag'
        assert stored_xcoms[0].task_id == 'task_1'
        assert stored_xcoms[0].execution_date == task_instance.execution_date

    def test_xcom_set_with_execution_date(self, session, task_instance):
        if False:
            i = 10
            return i + 15
        with pytest.deprecated_call():
            XCom.set(key='xcom_1', value={'key': 'value'}, dag_id=task_instance.dag_id, task_id=task_instance.task_id, execution_date=task_instance.execution_date, session=session)
        stored_xcoms = session.query(XCom).all()
        assert stored_xcoms[0].key == 'xcom_1'
        assert stored_xcoms[0].value == {'key': 'value'}
        assert stored_xcoms[0].dag_id == 'dag'
        assert stored_xcoms[0].task_id == 'task_1'
        assert stored_xcoms[0].execution_date == task_instance.execution_date

    @pytest.fixture()
    def setup_for_xcom_set_again_replace(self, task_instance, push_simple_json_xcom):
        if False:
            return 10
        push_simple_json_xcom(ti=task_instance, key='xcom_1', value={'key1': 'value1'})

    @pytest.mark.usefixtures('setup_for_xcom_set_again_replace')
    def test_xcom_set_again_replace(self, session, task_instance):
        if False:
            i = 10
            return i + 15
        assert session.query(XCom).one().value == {'key1': 'value1'}
        XCom.set(key='xcom_1', value={'key2': 'value2'}, dag_id=task_instance.dag_id, task_id='task_1', run_id=task_instance.run_id, session=session)
        assert session.query(XCom).one().value == {'key2': 'value2'}

    @pytest.mark.usefixtures('setup_for_xcom_set_again_replace')
    def test_xcom_set_again_replace_with_execution_date(self, session, task_instance):
        if False:
            for i in range(10):
                print('nop')
        assert session.query(XCom).one().value == {'key1': 'value1'}
        with pytest.deprecated_call():
            XCom.set(key='xcom_1', value={'key2': 'value2'}, dag_id=task_instance.dag_id, task_id='task_1', execution_date=task_instance.execution_date, session=session)
        assert session.query(XCom).one().value == {'key2': 'value2'}

@pytest.mark.usefixtures('setup_xcom_pickling')
class TestXComClear:

    @pytest.fixture()
    def setup_for_xcom_clear(self, task_instance, push_simple_json_xcom):
        if False:
            return 10
        push_simple_json_xcom(ti=task_instance, key='xcom_1', value={'key': 'value'})

    @pytest.mark.usefixtures('setup_for_xcom_clear')
    def test_xcom_clear(self, session, task_instance):
        if False:
            for i in range(10):
                print('nop')
        assert session.query(XCom).count() == 1
        XCom.clear(dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id=task_instance.run_id, session=session)
        assert session.query(XCom).count() == 0

    @pytest.mark.usefixtures('setup_for_xcom_clear')
    def test_xcom_clear_with_execution_date(self, session, task_instance):
        if False:
            while True:
                i = 10
        assert session.query(XCom).count() == 1
        with pytest.deprecated_call():
            XCom.clear(dag_id=task_instance.dag_id, task_id=task_instance.task_id, execution_date=task_instance.execution_date, session=session)
        assert session.query(XCom).count() == 0

    @pytest.mark.usefixtures('setup_for_xcom_clear')
    def test_xcom_clear_different_run(self, session, task_instance):
        if False:
            i = 10
            return i + 15
        XCom.clear(dag_id=task_instance.dag_id, task_id=task_instance.task_id, run_id='different_run', session=session)
        assert session.query(XCom).count() == 1

    @pytest.mark.usefixtures('setup_for_xcom_clear')
    def test_xcom_clear_different_execution_date(self, session, task_instance):
        if False:
            return 10
        with pytest.deprecated_call():
            XCom.clear(dag_id=task_instance.dag_id, task_id=task_instance.task_id, execution_date=timezone.utcnow(), session=session)
        assert session.query(XCom).count() == 1