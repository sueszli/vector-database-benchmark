from six.moves import http_client
from st2common.bootstrap import actionsregistrar
from st2common.bootstrap import runnersregistrar
import st2tests
from st2tests.api import FunctionalTest
from st2tests.fixtures.packs.core.fixture import PACK_PATH as CORE_PACK_PATH
from st2tests.fixtures.packs.orquesta_tests.fixture import PACK_PATH as TEST_PACK_PATH
PACKS = [TEST_PACK_PATH, CORE_PACK_PATH]

class WorkflowInspectionControllerTest(FunctionalTest, st2tests.WorkflowTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super(WorkflowInspectionControllerTest, cls).setUpClass()
        st2tests.WorkflowTestCase.setUpClass()
        runnersregistrar.register_runners()
        actions_registrar = actionsregistrar.ActionsRegistrar(use_pack_cache=False, fail_on_failure=True)
        for pack in PACKS:
            actions_registrar.register_from_pack(pack)

    def _do_post(self, wf_def, expect_errors=False):
        if False:
            i = 10
            return i + 15
        return self.app.post('/v1/workflows/inspect', wf_def, expect_errors=expect_errors, content_type='text/plain')

    def test_inspection(self):
        if False:
            while True:
                i = 10
        wf_file = 'sequential.yaml'
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, wf_file)
        wf_def = self.get_wf_def(TEST_PACK_PATH, wf_meta)
        expected_errors = []
        response = self._do_post(wf_def, expect_errors=False)
        self.assertEqual(http_client.OK, response.status_int)
        self.assertListEqual(response.json, expected_errors)

    def test_inspection_return_errors(self):
        if False:
            print('Hello World!')
        wf_file = 'fail-inspection.yaml'
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, wf_file)
        wf_def = self.get_wf_def(TEST_PACK_PATH, wf_meta)
        expected_errors = [{'type': 'content', 'message': 'The action "std.noop" is not registered in the database.', 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.action', 'spec_path': 'tasks.task3.action'}, {'type': 'context', 'language': 'yaql', 'expression': '<% ctx().foobar %>', 'message': 'Variable "foobar" is referenced before assignment.', 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.input', 'spec_path': 'tasks.task1.input'}, {'type': 'expression', 'language': 'yaql', 'expression': '<% <% succeeded() %>', 'message': "Parse error: unexpected '<' at position 0 of expression '<% succeeded()'", 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.next.items.properties.when', 'spec_path': 'tasks.task2.next[0].when'}, {'type': 'syntax', 'message': "[{'cmd': 'echo <% ctx().macro %>'}] is not valid under any of the given schemas", 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.input.oneOf', 'spec_path': 'tasks.task2.input'}]
        response = self._do_post(wf_def, expect_errors=False)
        self.assertEqual(http_client.OK, response.status_int)
        self.assertListEqual(response.json, expected_errors)