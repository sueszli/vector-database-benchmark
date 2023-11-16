from __future__ import absolute_import
import os
import sys
import signal
import datetime
from st2common.util import concurrency
from st2common.constants import action as action_constants
from st2common.util import date as date_utils
from st2common.models.db.execution import ActionExecutionDB
from st2common.models.db.liveaction import LiveActionDB
from st2common.models.db.execution import ActionExecutionOutputDB
from st2common.models.system.common import ResourceReference
from st2common.persistence.liveaction import LiveAction
from st2common.persistence.execution import ActionExecution
from st2common.persistence.execution import ActionExecutionOutput
from st2common.services import executions
from st2tests.base import IntegrationTestCase
from st2tests.base import CleanDbTestCase
from st2tests.fixtures.generic.fixture import PACK_NAME as FIXTURES_PACK
from st2tests.fixturesloader import FixturesLoader
from six.moves import range
__all__ = ['GarbageCollectorServiceTestCase']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ST2_CONFIG_PATH = os.path.join(BASE_DIR, '../../../conf/st2.tests.conf')
ST2_CONFIG_PATH = os.path.abspath(ST2_CONFIG_PATH)
INQUIRY_CONFIG_PATH = os.path.join(BASE_DIR, '../../../conf/st2.tests2.conf')
INQUIRY_CONFIG_PATH = os.path.abspath(INQUIRY_CONFIG_PATH)
PYTHON_BINARY = sys.executable
BINARY = os.path.join(BASE_DIR, '../../../st2reactor/bin/st2garbagecollector')
BINARY = os.path.abspath(BINARY)
CMD = [PYTHON_BINARY, BINARY, '--config-file', ST2_CONFIG_PATH]
CMD_INQUIRY = [PYTHON_BINARY, BINARY, '--config-file', INQUIRY_CONFIG_PATH]
TEST_FIXTURES = {'runners': ['inquirer.yaml'], 'actions': ['ask.yaml']}

class GarbageCollectorServiceTestCase(IntegrationTestCase, CleanDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super(GarbageCollectorServiceTestCase, cls).setUpClass()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(GarbageCollectorServiceTestCase, self).setUp()
        self.models = FixturesLoader().save_fixtures_to_db(fixtures_pack=FIXTURES_PACK, fixtures_dict=TEST_FIXTURES)

    def test_garbage_collection(self):
        if False:
            for i in range(10):
                print('nop')
        now = date_utils.get_datetime_utc_now()
        status = action_constants.LIVEACTION_STATUS_SUCCEEDED
        old_executions_count = 15
        ttl_days = 30
        timestamp = now - datetime.timedelta(days=ttl_days)
        for index in range(0, old_executions_count):
            action_execution_db = ActionExecutionDB(start_timestamp=timestamp, end_timestamp=timestamp, status=status, action={'ref': 'core.local'}, runner={'name': 'local-shell-cmd'}, liveaction={'ref': 'foo'})
            ActionExecution.add_or_update(action_execution_db)
            stdout_db = ActionExecutionOutputDB(execution_id=str(action_execution_db.id), action_ref='core.local', runner_ref='dummy', timestamp=timestamp, output_type='stdout', data='stdout')
            ActionExecutionOutput.add_or_update(stdout_db)
            stderr_db = ActionExecutionOutputDB(execution_id=str(action_execution_db.id), action_ref='core.local', runner_ref='dummy', timestamp=timestamp, output_type='stderr', data='stderr')
            ActionExecutionOutput.add_or_update(stderr_db)
        new_executions_count = 5
        ttl_days = 2
        timestamp = now - datetime.timedelta(days=ttl_days)
        for index in range(0, new_executions_count):
            action_execution_db = ActionExecutionDB(start_timestamp=timestamp, end_timestamp=timestamp, status=status, action={'ref': 'core.local'}, runner={'name': 'local-shell-cmd'}, liveaction={'ref': 'foo'})
            ActionExecution.add_or_update(action_execution_db)
            stdout_db = ActionExecutionOutputDB(execution_id=str(action_execution_db.id), action_ref='core.local', runner_ref='dummy', timestamp=timestamp, output_type='stdout', data='stdout')
            ActionExecutionOutput.add_or_update(stdout_db)
            stderr_db = ActionExecutionOutputDB(execution_id=str(action_execution_db.id), action_ref='core.local', runner_ref='dummy', timestamp=timestamp, output_type='stderr', data='stderr')
            ActionExecutionOutput.add_or_update(stderr_db)
        new_output_count = 5
        ttl_days = 15
        timestamp = now - datetime.timedelta(days=ttl_days)
        for index in range(0, new_output_count):
            action_execution_db = ActionExecutionDB(start_timestamp=timestamp, end_timestamp=timestamp, status=status, action={'ref': 'core.local'}, runner={'name': 'local-shell-cmd'}, liveaction={'ref': 'foo'})
            ActionExecution.add_or_update(action_execution_db)
            stdout_db = ActionExecutionOutputDB(execution_id=str(action_execution_db.id), action_ref='core.local', runner_ref='dummy', timestamp=timestamp, output_type='stdout', data='stdout')
            ActionExecutionOutput.add_or_update(stdout_db)
            stderr_db = ActionExecutionOutputDB(execution_id=str(action_execution_db.id), action_ref='core.local', runner_ref='dummy', timestamp=timestamp, output_type='stderr', data='stderr')
            ActionExecutionOutput.add_or_update(stderr_db)
        execs = ActionExecution.get_all()
        self.assertEqual(len(execs), old_executions_count + new_executions_count + new_output_count)
        stdout_dbs = ActionExecutionOutput.query(output_type='stdout')
        self.assertEqual(len(stdout_dbs), old_executions_count + new_executions_count + new_output_count)
        stderr_dbs = ActionExecutionOutput.query(output_type='stderr')
        self.assertEqual(len(stderr_dbs), old_executions_count + new_executions_count + new_output_count)
        process = self._start_garbage_collector()
        concurrency.sleep(15)
        process.send_signal(signal.SIGKILL)
        self.remove_process(process=process)
        execs = ActionExecution.get_all()
        self.assertEqual(len(execs), new_executions_count + new_output_count)
        stdout_dbs = ActionExecutionOutput.query(output_type='stdout')
        self.assertEqual(len(stdout_dbs), new_executions_count)
        stderr_dbs = ActionExecutionOutput.query(output_type='stderr')
        self.assertEqual(len(stderr_dbs), new_executions_count)

    def test_inquiry_garbage_collection(self):
        if False:
            print('Hello World!')
        now = date_utils.get_datetime_utc_now()
        old_inquiry_count = 15
        timestamp = now - datetime.timedelta(minutes=3)
        for index in range(0, old_inquiry_count):
            self._create_inquiry(ttl=2, timestamp=timestamp)
        disabled_inquiry_count = 3
        timestamp = now - datetime.timedelta(minutes=3)
        for index in range(0, disabled_inquiry_count):
            self._create_inquiry(ttl=0, timestamp=timestamp)
        new_inquiry_count = 5
        timestamp = now - datetime.timedelta(minutes=3)
        for index in range(0, new_inquiry_count):
            self._create_inquiry(ttl=15, timestamp=timestamp)
        filters = {'status': action_constants.LIVEACTION_STATUS_PENDING}
        inquiries = list(ActionExecution.query(**filters))
        self.assertEqual(len(inquiries), old_inquiry_count + new_inquiry_count + disabled_inquiry_count)
        process = self._start_garbage_collector()
        concurrency.sleep(15)
        process.send_signal(signal.SIGKILL)
        self.remove_process(process=process)
        inquiries = list(ActionExecution.query(**filters))
        self.assertEqual(len(inquiries), new_inquiry_count + disabled_inquiry_count)

    def _create_inquiry(self, ttl, timestamp):
        if False:
            while True:
                i = 10
        action_db = self.models['actions']['ask.yaml']
        liveaction_db = LiveActionDB()
        liveaction_db.status = action_constants.LIVEACTION_STATUS_PENDING
        liveaction_db.start_timestamp = timestamp
        liveaction_db.action = ResourceReference(name=action_db.name, pack=action_db.pack).ref
        liveaction_db.result = {'ttl': ttl}
        liveaction_db = LiveAction.add_or_update(liveaction_db)
        executions.create_execution_object(liveaction_db)

    def _start_garbage_collector(self):
        if False:
            for i in range(10):
                print('nop')
        subprocess = concurrency.get_subprocess_module()
        process = subprocess.Popen(CMD_INQUIRY, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, preexec_fn=os.setsid)
        self.add_process(process=process)
        return process