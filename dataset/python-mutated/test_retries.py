from __future__ import annotations
import logging
from unittest import mock
import pytest
from sqlalchemy.exc import OperationalError
from airflow.utils.retries import retry_db_transaction

class TestRetries:

    def test_retry_db_transaction_with_passing_retries(self):
        if False:
            i = 10
            return i + 15
        'Test that retries can be passed to decorator'
        mock_obj = mock.MagicMock()
        mock_session = mock.MagicMock()
        op_error = OperationalError(statement=mock.ANY, params=mock.ANY, orig=mock.ANY)

        @retry_db_transaction(retries=2)
        def test_function(session):
            if False:
                print('Hello World!')
            session.execute('select 1')
            mock_obj(2)
            raise op_error
        with pytest.raises(OperationalError):
            test_function(session=mock_session)
        assert mock_obj.call_count == 2

    @pytest.mark.db_test
    def test_retry_db_transaction_with_default_retries(self, caplog):
        if False:
            return 10
        'Test that by default 3 retries will be carried out'
        mock_obj = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_rollback = mock.MagicMock()
        mock_session.rollback = mock_rollback
        op_error = OperationalError(statement=mock.ANY, params=mock.ANY, orig=mock.ANY)

        @retry_db_transaction
        def test_function(session):
            if False:
                while True:
                    i = 10
            session.execute('select 1')
            mock_obj(2)
            raise op_error
        caplog.set_level(logging.DEBUG, logger=self.__module__)
        caplog.clear()
        with pytest.raises(OperationalError):
            test_function(session=mock_session)
        for try_no in range(1, 4):
            assert f'Running TestRetries.test_retry_db_transaction_with_default_retries.<locals>.test_function with retries. Try {try_no} of 3' in caplog.messages
        assert mock_session.execute.call_count == 3
        assert mock_rollback.call_count == 3
        mock_rollback.assert_has_calls([mock.call(), mock.call(), mock.call()])

    def test_retry_db_transaction_fails_when_used_in_function_without_retry(self):
        if False:
            i = 10
            return i + 15
        'Test that an error is raised when the decorator is used on a function without session arg'
        with pytest.raises(ValueError, match='has no `session` argument'):

            @retry_db_transaction
            def test_function():
                if False:
                    return 10
                print('hi')
                raise OperationalError(statement=mock.ANY, params=mock.ANY, orig=mock.ANY)

    def test_retry_db_transaction_fails_when_session_not_passed(self):
        if False:
            print('Hello World!')
        'Test that an error is raised when session is not passed to the function'

        @retry_db_transaction
        def test_function(session):
            if False:
                while True:
                    i = 10
            session.execute('select 1;')
            raise OperationalError(statement=mock.ANY, params=mock.ANY, orig=mock.ANY)
        error_message = f'session is a required argument for {test_function.__qualname__}'
        with pytest.raises(TypeError, match=error_message):
            test_function()