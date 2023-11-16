import unittest
from golem.core.variables import PROTOCOL_CONST
from .base import NodeTestBase, disable_key_reuse

class GolemNodeTest(NodeTestBase):

    def test_regular_task_run(self):
        if False:
            while True:
                i = 10
        '\n        runs a normal, successful task run between a single provider\n        and a single requestor.\n        '
        self._run_test('golem.regular_run')

    def test_regular_task_api_run(self):
        if False:
            while True:
                i = 10
        '\n        runs a normal, successful task run between a single provider\n        and a single requestor. On the new task_api task types\n        '
        self._run_test('golem.task_api')

    def test_concent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        runs a normal task between a provider and a requestor\n        with Concent enabled\n        '
        self._run_test('golem.concent')

    def test_rpc(self):
        if False:
            return 10
        self._run_test('golem.rpc_test')

    def test_rpc_concent(self):
        if False:
            while True:
                i = 10
        self._run_test('golem.rpc_test.concent')

    @disable_key_reuse
    def test_rpc_mainnet(self):
        if False:
            return 10
        self._run_test('golem.rpc_test.mainnet', '--mainnet')

    def test_task_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test('golem.task_timeout')

    def test_frame_restart(self):
        if False:
            return 10
        self._run_test('golem.restart_frame')

    def test_exr(self):
        if False:
            i = 10
            return i + 15
        '\n        verifies if Golem - when supplied with `EXR` as the format - will\n        render the output as EXR with the proper extension.\n        '
        self._run_test('golem.exr')

    def test_jpeg(self):
        if False:
            while True:
                i = 10
        '\n        verifies if Golem - when supplied with `JPEG` as the format - will\n        render the output as JPEG with the proper extension.\n        '
        self._run_test('golem.jpeg', **{'task-package': 'cube'})

    def test_jpg(self):
        if False:
            i = 10
            return i + 15
        "\n        verifies if Golem - when supplied with `JPG` as the format - will\n        still execute a task.\n\n        as the proper name of the format in Golem's internals is `JPEG`\n        the format is treated as an _unknown_ and thus, the default `PNG`\n        is used.\n        "
        self._run_test('golem.jpg')

    def test_nested(self):
        if False:
            return 10
        self._run_test('golem.regular_run_stop_on_reject', **{'task-package': 'nested'})

    def test_zero_price(self):
        if False:
            while True:
                i = 10
        self._run_test('golem.zero_price')

    def test_task_output_directory(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test('golem.task_output')

    def test_large_result(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test('golem.separate_hyperg', **{'task-package': 'cubes', 'task-settings': '3k-low-samples'})

    def test_restart_failed_subtasks(self):
        if False:
            print('Hello World!')
        self._run_test('golem.restart_failed_subtasks')

    def test_main_scene_file(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test('golem.nested_column')

    def test_multinode_regular_run(self):
        if False:
            print('Hello World!')
        self._run_test('golem.multinode_regular_run')

    def test_disabled_verification(self):
        if False:
            print('Hello World!')
        self._run_test('golem.disabled_verification')

    def test_lenient_verification(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test('golem.lenient_verification')

    def test_simultaneous_tasks(self):
        if False:
            return 10
        self._run_test('golem.simultaneous_tasks')

    def test_four_by_three(self):
        if False:
            i = 10
            return i + 15
        '\n        introduces an uneven division 400 pixels -> 3 subtasks\n        to test for the cropping regressions\n        '
        self._run_test('golem.regular_run_stop_on_reject', **{'task-settings': '4-by-3'})

    def test_concent_provider(self):
        if False:
            print('Hello World!')
        self._run_test('golem.concent_provider')

    def test_wasm_vbr_success(self):
        if False:
            for i in range(10):
                print('nop')
        self._run_test('golem.wasm_vbr_success')

    def test_wasm_vbr_single_failure(self):
        if False:
            return 10
        self._run_test('golem.wasm_vbr_single_failure')

    def test_wasm_vbr_crash_provider_side(self):
        if False:
            print('Hello World!')
        self._run_test('golem.wasm_vbr_crash_provider_side')