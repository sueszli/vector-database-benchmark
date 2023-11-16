from __future__ import annotations
import datetime
import pickle
from copy import deepcopy
from unittest import mock
from unittest.mock import MagicMock
import pytest
from kubernetes.client import models as k8s
from sqlalchemy import text
from sqlalchemy.exc import StatementError
from airflow import settings
from airflow.models.dag import DAG
from airflow.serialization.enums import DagAttributeTypes, Encoding
from airflow.serialization.serialized_objects import BaseSerialization
from airflow.settings import Session
from airflow.utils.sqlalchemy import ExecutorConfigType, ensure_pod_is_valid_after_unpickling, nowait, prohibit_commit, skip_locked, with_row_locks
from airflow.utils.state import State
from airflow.utils.timezone import utcnow
pytestmark = pytest.mark.db_test
TEST_POD = k8s.V1Pod(spec=k8s.V1PodSpec(containers=[k8s.V1Container(name='base')]))

class TestSqlAlchemyUtils:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        session = Session()
        if session.bind.dialect.name == 'postgresql':
            session.execute(text("SET timezone='Europe/Amsterdam'"))
        self.session = session

    def test_utc_transformations(self):
        if False:
            i = 10
            return i + 15
        '\n        Test whether what we are storing is what we are retrieving\n        for datetimes\n        '
        dag_id = 'test_utc_transformations'
        start_date = utcnow()
        iso_date = start_date.isoformat()
        execution_date = start_date + datetime.timedelta(hours=1, days=1)
        dag = DAG(dag_id=dag_id, start_date=start_date)
        dag.clear()
        run = dag.create_dagrun(run_id=iso_date, state=State.NONE, execution_date=execution_date, start_date=start_date, session=self.session)
        assert execution_date == run.execution_date
        assert start_date == run.start_date
        assert execution_date.utcoffset().total_seconds() == 0.0
        assert start_date.utcoffset().total_seconds() == 0.0
        assert iso_date == run.run_id
        assert run.start_date.isoformat() == run.run_id
        dag.clear()

    def test_process_bind_param_naive(self):
        if False:
            print('Hello World!')
        '\n        Check if naive datetimes are prevented from saving to the db\n        '
        dag_id = 'test_process_bind_param_naive'
        start_date = datetime.datetime.now()
        dag = DAG(dag_id=dag_id, start_date=start_date)
        dag.clear()
        with pytest.raises((ValueError, StatementError)):
            dag.create_dagrun(run_id=start_date.isoformat, state=State.NONE, execution_date=start_date, start_date=start_date, session=self.session)
        dag.clear()

    @pytest.mark.parametrize('dialect, supports_for_update_of, expected_return_value', [('postgresql', True, {'skip_locked': True}), ('mysql', False, {}), ('mysql', True, {'skip_locked': True}), ('sqlite', False, {'skip_locked': True})])
    def test_skip_locked(self, dialect, supports_for_update_of, expected_return_value):
        if False:
            print('Hello World!')
        session = mock.Mock()
        session.bind.dialect.name = dialect
        session.bind.dialect.supports_for_update_of = supports_for_update_of
        assert skip_locked(session=session) == expected_return_value

    @pytest.mark.parametrize('dialect, supports_for_update_of, expected_return_value', [('postgresql', True, {'nowait': True}), ('mysql', False, {}), ('mysql', True, {'nowait': True}), ('sqlite', False, {'nowait': True})])
    def test_nowait(self, dialect, supports_for_update_of, expected_return_value):
        if False:
            print('Hello World!')
        session = mock.Mock()
        session.bind.dialect.name = dialect
        session.bind.dialect.supports_for_update_of = supports_for_update_of
        assert nowait(session=session) == expected_return_value

    @pytest.mark.parametrize('dialect, supports_for_update_of, use_row_level_lock_conf, expected_use_row_level_lock', [('postgresql', True, True, True), ('postgresql', True, False, False), ('mysql', False, True, False), ('mysql', False, False, False), ('mysql', True, True, True), ('mysql', True, False, False), ('sqlite', False, True, True)])
    def test_with_row_locks(self, dialect, supports_for_update_of, use_row_level_lock_conf, expected_use_row_level_lock):
        if False:
            for i in range(10):
                print('nop')
        query = mock.Mock()
        session = mock.Mock()
        session.bind.dialect.name = dialect
        session.bind.dialect.supports_for_update_of = supports_for_update_of
        with mock.patch('airflow.utils.sqlalchemy.USE_ROW_LEVEL_LOCKING', use_row_level_lock_conf):
            returned_value = with_row_locks(query=query, session=session, nowait=True)
        if expected_use_row_level_lock:
            query.with_for_update.assert_called_once_with(nowait=True)
        else:
            assert returned_value == query
            query.with_for_update.assert_not_called()

    def test_prohibit_commit(self):
        if False:
            while True:
                i = 10
        with prohibit_commit(self.session) as guard:
            self.session.execute(text('SELECT 1'))
            with pytest.raises(RuntimeError):
                self.session.commit()
            self.session.rollback()
            self.session.execute(text('SELECT 1'))
            guard.commit()
            with pytest.raises(RuntimeError):
                self.session.execute(text('SELECT 1'))
                self.session.commit()

    def test_prohibit_commit_specific_session_only(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that "prohibit_commit" applies only to the given session object,\n        not any other session objects that may be used\n        '
        other_session = Session.session_factory()
        assert other_session is not self.session
        with prohibit_commit(self.session):
            self.session.execute(text('SELECT 1'))
            with pytest.raises(RuntimeError):
                self.session.commit()
            self.session.rollback()
            other_session.execute(text('SELECT 1'))
            other_session.commit()

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.session.close()
        settings.engine.dispose()

class TestExecutorConfigType:

    @pytest.mark.parametrize('input, expected', [('anything', 'anything'), ({'pod_override': TEST_POD}, {'pod_override': {'__var': {'spec': {'containers': [{'name': 'base'}]}}, '__type': DagAttributeTypes.POD}})])
    def test_bind_processor(self, input, expected):
        if False:
            return 10
        '\n        The returned bind processor should pickle the object as is, unless it is a dictionary with\n        a pod_override node, in which case it should run it through BaseSerialization.\n        '
        config_type = ExecutorConfigType()
        mock_dialect = MagicMock()
        mock_dialect.dbapi = None
        process = config_type.bind_processor(mock_dialect)
        assert pickle.loads(process(input)) == expected
        assert pickle.loads(process(input)) == expected, 'should not mutate variable'

    @pytest.mark.parametrize('input', [pytest.param(pickle.dumps('anything'), id='anything'), pytest.param(pickle.dumps({'pod_override': BaseSerialization.serialize(TEST_POD)}), id='serialized_pod'), pytest.param(pickle.dumps({'pod_override': TEST_POD}), id='old_pickled_raw_pod'), pytest.param(pickle.dumps({'pod_override': {'name': 'hi'}}), id='arbitrary_dict')])
    def test_result_processor(self, input):
        if False:
            print('Hello World!')
        '\n        The returned bind processor should pickle the object as is, unless it is a dictionary with\n        a pod_override node whose value was serialized with BaseSerialization.\n        '
        config_type = ExecutorConfigType()
        mock_dialect = MagicMock()
        mock_dialect.dbapi = None
        process = config_type.result_processor(mock_dialect, None)
        result = process(input)
        expected = pickle.loads(input)
        pod_override = isinstance(expected, dict) and expected.get('pod_override')
        if pod_override and isinstance(pod_override, dict) and pod_override.get(Encoding.TYPE):
            expected['pod_override'] = BaseSerialization.deserialize(expected['pod_override'])
        assert result == expected

    def test_compare_values(self):
        if False:
            print('Hello World!')
        '\n        When comparison raises AttributeError, return False.\n        This can happen when executor config contains kubernetes objects pickled\n        under older kubernetes library version.\n        '

        class MockAttrError:

            def __eq__(self, other):
                if False:
                    print('Hello World!')
                raise AttributeError('hello')
        a = MockAttrError()
        with pytest.raises(AttributeError):
            assert a == a
        instance = ExecutorConfigType()
        assert instance.compare_values(a, a) is False
        assert instance.compare_values('a', 'a') is True

    def test_result_processor_bad_pickled_obj(self):
        if False:
            print('Hello World!')
        '\n        If unpickled obj is missing attrs that curr lib expects\n        '
        test_container = k8s.V1Container(name='base')
        test_pod = k8s.V1Pod(spec=k8s.V1PodSpec(containers=[test_container]))
        copy_of_test_pod = deepcopy(test_pod)
        assert 'tty' in test_container.openapi_types
        assert hasattr(test_container, '_tty')
        del test_container._tty
        with pytest.raises(AttributeError):
            test_pod.to_dict()
        assert copy_of_test_pod.to_dict()
        fixed_pod = ensure_pod_is_valid_after_unpickling(test_pod)
        assert fixed_pod.to_dict() == copy_of_test_pod.to_dict()
        with pytest.raises(AttributeError):
            test_pod.to_dict()
        input = pickle.dumps({'pod_override': TEST_POD})
        config_type = ExecutorConfigType()
        mock_dialect = MagicMock()
        mock_dialect.dbapi = None
        process = config_type.result_processor(mock_dialect, None)
        result = process(input)
        assert result['pod_override'].to_dict() == copy_of_test_pod.to_dict()