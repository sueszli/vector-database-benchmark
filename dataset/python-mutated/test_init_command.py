import os.path
import shutil
from pathlib import Path
from textwrap import dedent
from typing import List
from unittest import skipIf
from parameterized import parameterized
from samcli.commands.pipeline.bootstrap.cli import PIPELINE_CONFIG_DIR, PIPELINE_CONFIG_FILENAME
from samcli.commands.pipeline.init.interactive_init_flow import APP_PIPELINE_TEMPLATES_REPO_LOCAL_NAME
from samcli.cli.global_config import GlobalConfig
from tests.integration.pipeline.base import InitIntegBase, BootstrapIntegBase
from tests.integration.pipeline.test_bootstrap_command import SKIP_BOOTSTRAP_TESTS, CREDENTIAL_PROFILE
from tests.testing_utils import run_command_with_inputs, get_sam_command
QUICK_START_JENKINS_INPUTS_WITHOUT_AUTO_FILL = ['1', '1', '', 'credential-id', 'main', 'template.yaml', 'test', 'test-stack', 'test-pipeline-execution-role', 'test-cfn-execution-role', 'test-bucket', 'test-ecr', 'us-east-2', 'prod', 'prod-stack', 'prod-pipeline-execution-role', 'prod-cfn-execution-role', 'prod-bucket', 'prod-ecr', 'us-west-2']
SHARED_PATH: Path = GlobalConfig().config_dir
EXPECTED_JENKINS_FILE_PATH = Path(SHARED_PATH, APP_PIPELINE_TEMPLATES_REPO_LOCAL_NAME, 'tests', 'testfile_jenkins', 'expected')

class TestInit(InitIntegBase):
    """
    Here we use Jenkins template for testing
    """

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        pipelineconfig_file = Path(PIPELINE_CONFIG_DIR, PIPELINE_CONFIG_FILENAME)
        if pipelineconfig_file.exists():
            pipelineconfig_file.unlink()

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        shutil.rmtree(PIPELINE_CONFIG_DIR, ignore_errors=True)

    def test_quick_start(self):
        if False:
            return 10
        generated_jenkinsfile_path = Path('Jenkinsfile')
        self.generated_files.append(generated_jenkinsfile_path)
        init_command_list = self.get_init_command_list()
        init_process_execute = run_command_with_inputs(init_command_list, QUICK_START_JENKINS_INPUTS_WITHOUT_AUTO_FILL)
        self.assertEqual(init_process_execute.process.returncode, 0)
        self.assertTrue(Path('Jenkinsfile').exists())
        with open(EXPECTED_JENKINS_FILE_PATH, 'r') as expected, open(generated_jenkinsfile_path, 'r') as output:
            self.assertEqual(expected.read(), output.read())

    def test_failed_when_generated_file_already_exist_override(self):
        if False:
            return 10
        generated_jenkinsfile_path = Path('Jenkinsfile')
        generated_jenkinsfile_path.touch()
        self.generated_files.append(generated_jenkinsfile_path)
        init_command_list = self.get_init_command_list()
        init_process_execute = run_command_with_inputs(init_command_list, [*QUICK_START_JENKINS_INPUTS_WITHOUT_AUTO_FILL, 'y'])
        self.assertEqual(init_process_execute.process.returncode, 0)
        self.assertTrue(Path('Jenkinsfile').exists())
        with open(EXPECTED_JENKINS_FILE_PATH, 'r') as expected, open(generated_jenkinsfile_path, 'r') as output:
            self.assertEqual(expected.read(), output.read())

    def test_failed_when_generated_file_already_exist_not_override(self):
        if False:
            while True:
                i = 10
        generated_jenkinsfile_path = Path('Jenkinsfile')
        generated_jenkinsfile_path.touch()
        self.generated_files.append(generated_jenkinsfile_path)
        init_command_list = self.get_init_command_list()
        init_process_execute = run_command_with_inputs(init_command_list, [*QUICK_START_JENKINS_INPUTS_WITHOUT_AUTO_FILL, ''])
        self.assertEqual(init_process_execute.process.returncode, 0)
        with open(EXPECTED_JENKINS_FILE_PATH, 'r') as expected, open(os.path.join('.aws-sam', 'pipeline', 'generated-files', 'Jenkinsfile'), 'r') as output:
            self.assertEqual(expected.read(), output.read())
        self.assertEqual('', open('Jenkinsfile', 'r').read())

    def test_custom_template_with_manifest(self):
        if False:
            print('Hello World!')
        generated_file = Path('weather')
        self.generated_files.append(generated_file)
        custom_template_path = Path(__file__).parent.parent.joinpath(Path('testdata', 'pipeline', 'custom_template_with_manifest'))
        inputs = ['2', str(custom_template_path), '2', '', 'Rainy']
        init_command_list = self.get_init_command_list()
        init_process_execute = run_command_with_inputs(init_command_list, inputs)
        self.assertEqual(init_process_execute.process.returncode, 0)
        self.assertTrue(generated_file.exists())
        with open(generated_file, 'r') as f:
            self.assertEqual('Rainy\n', f.read())

    def test_custom_template_without_manifest(self):
        if False:
            i = 10
            return i + 15
        generated_file = Path('weather')
        self.generated_files.append(generated_file)
        custom_template_path = Path(__file__).parent.parent.joinpath(Path('testdata', 'pipeline', 'custom_template'))
        inputs = ['2', str(custom_template_path), '', 'Rainy']
        init_command_list = self.get_init_command_list()
        init_process_execute = run_command_with_inputs(init_command_list, inputs)
        self.assertEqual(init_process_execute.process.returncode, 0)
        self.assertTrue(generated_file.exists())
        with open(generated_file, 'r') as f:
            self.assertEqual('Rainy\n', f.read())

    @parameterized.expand([('with_bootstrap',), (False,)])
    def test_with_pipelineconfig_has_all_stage_values(self, with_bootstrap):
        if False:
            i = 10
            return i + 15
        generated_jenkinsfile_path = Path('Jenkinsfile')
        self.generated_files.append(generated_jenkinsfile_path)
        Path(PIPELINE_CONFIG_DIR).mkdir(parents=True, exist_ok=True)
        pipelineconfig_path = Path(PIPELINE_CONFIG_DIR, PIPELINE_CONFIG_FILENAME)
        with open(pipelineconfig_path, 'w') as f:
            f.write(dedent('            version = 0.1\n            [default]\n            [default.pipeline_bootstrap]\n            [default.pipeline_bootstrap.parameters]\n            pipeline_user = "arn:aws:iam::123:user/aws-sam-cli-managed-test-pipeline-res-PipelineUser-123"\n\n            [test]\n            [test.pipeline_bootstrap]\n            [test.pipeline_bootstrap.parameters]\n            pipeline_execution_role = "test-pipeline-execution-role"\n            cloudformation_execution_role = "test-cfn-execution-role"\n            artifacts_bucket = "test-bucket"\n            image_repository = "test-ecr"\n            region = "us-east-2"\n\n            [prod]\n            [prod.pipeline_bootstrap]\n            [prod.pipeline_bootstrap.parameters]\n            pipeline_execution_role = "prod-pipeline-execution-role"\n            cloudformation_execution_role = "prod-cfn-execution-role"\n            artifacts_bucket = "prod-bucket"\n            image_repository = "prod-ecr"\n            region = "us-west-2"\n            '))
        inputs = ['1', '1', 'credential-id', 'main', 'template.yaml', '1', 'test-stack', '2', 'prod-stack']
        init_command_list = self.get_init_command_list(with_bootstrap)
        init_process_execute = run_command_with_inputs(init_command_list, inputs)
        self.assertEqual(init_process_execute.process.returncode, 0)
        self.assertTrue(Path('Jenkinsfile').exists())
        with open(EXPECTED_JENKINS_FILE_PATH, 'r') as expected, open(generated_jenkinsfile_path, 'r') as output:
            self.assertEqual(expected.read(), output.read())

@skipIf(SKIP_BOOTSTRAP_TESTS, 'Skip bootstrap tests in CI/CD only')
class TestInitWithBootstrap(BootstrapIntegBase):
    generated_files: List[Path] = []

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.command_list = [get_sam_command(), 'pipeline', 'init', '--bootstrap']
        generated_jenkinsfile_path = Path('Jenkinsfile')
        self.generated_files.append(generated_jenkinsfile_path)

    def tearDown(self) -> None:
        if False:
            while True:
                i = 10
        for generated_file in self.generated_files:
            if generated_file.is_dir():
                shutil.rmtree(generated_file, ignore_errors=True)
            elif generated_file.exists():
                generated_file.unlink()
        super().tearDown()

    def test_without_stages_in_pipeline_config(self):
        if False:
            i = 10
            return i + 15
        stage_configuration_names = []
        for suffix in ['1', '2']:
            (stage_configuration_name, stack_name) = self._get_stage_and_stack_name(suffix)
            stage_configuration_names.append(stage_configuration_name)
            self.stack_names.append(stack_name)
        inputs = ['1', '1', 'y', stage_configuration_names[0], CREDENTIAL_PROFILE, self.region, '', '', '', '', 'N', '', 'y', 'y', stage_configuration_names[1], CREDENTIAL_PROFILE, self.region, '', '', '', '', 'N', '', 'y', 'credential-id', 'main', 'template.yaml', '1', 'test-stack', '2', 'prod-stack']
        init_process_execute = run_command_with_inputs(self.command_list, inputs)
        self.assertEqual(init_process_execute.process.returncode, 0)
        self.assertIn('Here are the stage configuration names detected', init_process_execute.stdout.decode())
        self.assertIn(stage_configuration_names[0], init_process_execute.stdout.decode())
        self.assertIn(stage_configuration_names[1], init_process_execute.stdout.decode())

    def test_with_one_stages_in_pipeline_config(self):
        if False:
            for i in range(10):
                print('nop')
        stage_configuration_names = []
        for suffix in ['1', '2']:
            (stage_configuration_name, stack_name) = self._get_stage_and_stack_name(suffix)
            stage_configuration_names.append(stage_configuration_name)
            self.stack_names.append(stack_name)
        bootstrap_command_list = self.get_bootstrap_command_list()
        inputs = [stage_configuration_names[0], CREDENTIAL_PROFILE, self.region, '1', '', '', '', '', 'N', '', 'y']
        bootstrap_process_execute = run_command_with_inputs(bootstrap_command_list, inputs)
        self.assertEqual(bootstrap_process_execute.process.returncode, 0)
        inputs = ['1', '1', 'y', stage_configuration_names[1], CREDENTIAL_PROFILE, self.region, '', '', '', 'N', '', 'y', 'credential-id', 'main', 'template.yaml', '1', 'test-stack', '2', 'prod-stack']
        init_process_execute = run_command_with_inputs(self.command_list, inputs)
        self.assertEqual(init_process_execute.process.returncode, 0)
        self.assertIn('Here are the stage configuration names detected', init_process_execute.stdout.decode())
        self.assertIn(stage_configuration_names[0], init_process_execute.stdout.decode())
        self.assertIn(stage_configuration_names[1], init_process_execute.stdout.decode())