from __future__ import absolute_import
import mongoengine
import unittest2
from tooz import coordination
from st2common.exceptions import db as db_exc
from st2common.exceptions import workflow as wf_exc
from st2common.models.db import workflow as wf_db_models

class WorkflowExceptionTest(unittest2.TestCase):

    def test_retry_on_transient_db_errors(self):
        if False:
            return 10
        instance = wf_db_models.WorkflowExecutionDB()
        exc = db_exc.StackStormDBObjectWriteConflictError(instance)
        self.assertTrue(wf_exc.retry_on_transient_db_errors(exc))

    def test_do_not_retry_on_transient_db_errors(self):
        if False:
            while True:
                i = 10
        instance = wf_db_models.WorkflowExecutionDB()
        exc = db_exc.StackStormDBObjectConflictError('foobar', '1234', instance)
        self.assertFalse(wf_exc.retry_on_transient_db_errors(exc))
        self.assertFalse(wf_exc.retry_on_transient_db_errors(NotImplementedError()))
        self.assertFalse(wf_exc.retry_on_transient_db_errors(Exception()))

    def test_retry_on_connection_errors(self):
        if False:
            return 10
        exc = coordination.ToozConnectionError('foobar')
        self.assertTrue(wf_exc.retry_on_connection_errors(exc))
        exc = mongoengine.connection.ConnectionFailure()
        self.assertTrue(wf_exc.retry_on_connection_errors(exc))

    def test_do_not_retry_on_connection_errors(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(wf_exc.retry_on_connection_errors(NotImplementedError()))
        self.assertFalse(wf_exc.retry_on_connection_errors(Exception()))