import os
import tempfile
from subprocess import Popen, PIPE, TimeoutExpired
from unittest import skipIf
from urllib.parse import urlparse
import boto3
from parameterized import parameterized
import docker
from samcli.commands._utils.template import get_template_data
from .package_integ_base import PackageIntegBase
from tests.testing_utils import RUNNING_ON_CI, RUNNING_TEST_FOR_MASTER_ON_CI, RUN_BY_CANARY
SKIP_PACKAGE_TESTS = RUNNING_ON_CI and RUNNING_TEST_FOR_MASTER_ON_CI and (not RUN_BY_CANARY)
TIMEOUT = 300

@skipIf(SKIP_PACKAGE_TESTS, 'Skip package tests in CI/CD only')
class TestPackageImage(PackageIntegBase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.docker_client = docker.from_env()
        cls.local_images = [('public.ecr.aws/sam/emulation-python3.8', 'latest')]
        for (repo, tag) in cls.local_images:
            cls.docker_client.api.pull(repository=repo, tag=tag)
            cls.docker_client.api.tag(f'{repo}:{tag}', 'emulation-python3.8', tag='latest')
            cls.docker_client.api.tag(f'{repo}:{tag}', 'emulation-python3.8-2', tag='latest')
            cls.docker_client.api.tag(f'{repo}:{tag}', 'colorsrandomfunctionf61b9209', tag='latest')
        super(TestPackageImage, cls).setUpClass()

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestPackageImage, self).setUp()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super(TestPackageImage, self).tearDown()

    @parameterized.expand(['aws-serverless-function-image.yaml', 'aws-lambda-function-image.yaml', 'cdk_v1_synthesized_template_image_functions.json'])
    def test_package_template_without_image_repository(self, template_file):
        if False:
            print('Hello World!')
        template_path = self.test_data_path.joinpath(template_file)
        command_list = PackageIntegBase.get_command_list(template=template_path)
        process = Popen(command_list, stdout=PIPE, stderr=PIPE)
        try:
            (stdout, stderr) = process.communicate(timeout=TIMEOUT)
        except TimeoutExpired:
            process.kill()
            raise
        process_stderr = stderr.strip()
        self.assertIn("Error: Missing option '--image-repositories', '--image-repository'", process_stderr.decode('utf-8'))
        self.assertEqual(2, process.returncode)

    @parameterized.expand(['aws-serverless-function-image.yaml', 'aws-lambda-function-image.yaml', 'aws-lambda-function-image-and-api.yaml', 'cdk_v1_synthesized_template_image_functions.json'])
    def test_package_template_with_image_repository(self, template_file):
        if False:
            return 10
        template_path = self.test_data_path.joinpath(template_file)
        command_list = PackageIntegBase.get_command_list(image_repository=self.ecr_repo_name, template=template_path)
        process = Popen(command_list, stdout=PIPE)
        try:
            (stdout, _) = process.communicate(timeout=TIMEOUT)
        except TimeoutExpired:
            process.kill()
            raise
        process_stdout = stdout.strip()
        self.assertEqual(0, process.returncode)
        self.assertIn(f'{self.ecr_repo_name}', process_stdout.decode('utf-8'))

    @parameterized.expand([('Hello', 'aws-serverless-function-image.yaml'), ('MyLambdaFunction', 'aws-lambda-function-image.yaml'), ('ColorsRandomFunctionF61B9209', 'cdk_v1_synthesized_template_image_functions.json'), ('ColorsRandomFunction', 'cdk_v1_synthesized_template_image_functions.json')])
    def test_package_template_with_image_repositories(self, resource_id, template_file):
        if False:
            i = 10
            return i + 15
        template_path = self.test_data_path.joinpath(template_file)
        command_list = PackageIntegBase.get_command_list(image_repositories=f'{resource_id}={self.ecr_repo_name}', template=template_path)
        process = Popen(command_list, stdout=PIPE)
        try:
            (stdout, _) = process.communicate(timeout=TIMEOUT)
        except TimeoutExpired:
            process.kill()
            raise
        process_stdout = stdout.strip()
        self.assertIn(f'{self.ecr_repo_name}', process_stdout.decode('utf-8'))
        self.assertEqual(0, process.returncode)

    @parameterized.expand([('ColorsRandomFunctionF61B9209', 'cdk_v1_synthesized_template_Level2_nested_image_functions.json'), ('ColorsRandomFunction', 'cdk_v1_synthesized_template_Level2_nested_image_functions.json'), ('Level2Stack/ColorsRandomFunction', 'cdk_v1_synthesized_template_Level2_nested_image_functions.json'), ('ColorsRandomFunctionF61B9209', 'cdk_v1_synthesized_template_Level1_nested_image_functions.json'), ('ColorsRandomFunction', 'cdk_v1_synthesized_template_Level1_nested_image_functions.json'), ('Level1Stack/Level2Stack/ColorsRandomFunction', 'cdk_v1_synthesized_template_Level1_nested_image_functions.json')])
    def test_package_template_with_image_repositories_nested_stack(self, resource_id, template_file):
        if False:
            return 10
        template_path = self.test_data_path.joinpath(template_file)
        command_list = PackageIntegBase.get_command_list(image_repositories=f'{resource_id}={self.ecr_repo_name}', template=template_path, resolve_s3=True)
        process = Popen(command_list, stderr=PIPE)
        try:
            (_, stderr) = process.communicate(timeout=TIMEOUT)
        except TimeoutExpired:
            process.kill()
            raise
        process_stderr = stderr.strip()
        self.assertIn(f'{self.ecr_repo_name}', process_stderr.decode('utf-8'))
        self.assertEqual(0, process.returncode)

    @parameterized.expand(['aws-serverless-function-image.yaml', 'aws-lambda-function-image.yaml', 'cdk_v1_synthesized_template_image_functions.json'])
    def test_package_template_with_non_ecr_repo_uri_image_repository(self, template_file):
        if False:
            i = 10
            return i + 15
        template_path = self.test_data_path.joinpath(template_file)
        command_list = PackageIntegBase.get_command_list(image_repository='non-ecr-repo-uri', template=template_path, resolve_s3=True)
        process = Popen(command_list, stderr=PIPE)
        try:
            (_, stderr) = process.communicate(timeout=TIMEOUT)
        except TimeoutExpired:
            process.kill()
            raise
        process_stderr = stderr.strip()
        self.assertEqual(2, process.returncode)
        self.assertIn("Error: Invalid value for '--image-repository'", process_stderr.decode('utf-8'))

    @parameterized.expand(['aws-serverless-function-image.yaml', 'aws-lambda-function-image.yaml', 'cdk_v1_synthesized_template_image_functions.json'])
    def test_package_template_and_s3_bucket(self, template_file):
        if False:
            print('Hello World!')
        template_path = self.test_data_path.joinpath(template_file)
        command_list = PackageIntegBase.get_command_list(s3_bucket=self.s3_bucket, s3_prefix=self.s3_prefix, template=template_path)
        process = Popen(command_list, stdout=PIPE, stderr=PIPE)
        try:
            (_, stderr) = process.communicate(timeout=TIMEOUT)
        except TimeoutExpired:
            process.kill()
            raise
        process_stderr = stderr.strip()
        self.assertEqual(2, process.returncode)
        self.assertIn("Error: Missing option '--image-repositories', '--image-repository'", process_stderr.decode('utf-8'))

    @parameterized.expand(['aws-serverless-application-image.yaml'])
    def test_package_template_with_image_function_in_nested_application(self, template_file):
        if False:
            print('Hello World!')
        template_path = self.test_data_path.joinpath(template_file)
        with tempfile.NamedTemporaryFile() as packaged_file, tempfile.TemporaryFile() as packaged_nested_file:
            packaged_file.close()
            command_list = PackageIntegBase.get_command_list(image_repository=self.ecr_repo_name, template=template_path, resolve_s3=True, output_template_file=packaged_file.name)
            process = Popen(command_list, stdout=PIPE, stderr=PIPE)
            try:
                process.communicate(timeout=TIMEOUT)
            except TimeoutExpired:
                process.kill()
                raise
            self.assertEqual(0, process.returncode)
            template_dict = get_template_data(packaged_file.name)
            nested_app_template_uri = template_dict.get('Resources', {}).get('myApp', {}).get('Properties').get('Location')
            parsed = urlparse(nested_app_template_uri)
            (bucket_name, key) = parsed.path.lstrip('/').split('/')
            s3 = boto3.resource('s3')
            s3.Object(bucket_name, key).download_fileobj(packaged_nested_file)
            packaged_nested_file.seek(0)
            self.assertIn(f'{self.ecr_repo_name}', packaged_nested_file.read().decode())

    def test_package_with_deep_nested_template_image(self):
        if False:
            return 10
        '\n        this template contains two nested stacks:\n        - root\n          - FunctionA\n          - ChildStackX\n            - FunctionB\n            - ChildStackY\n              - FunctionA\n        '
        template_file = os.path.join('deep-nested-image', 'template.yaml')
        template_path = self.test_data_path.joinpath(template_file)
        command_list = PackageIntegBase.get_command_list(image_repository=self.ecr_repo_name, resolve_s3=True, template=template_path, force_upload=True)
        process = Popen(command_list, stdout=PIPE, stderr=PIPE)
        try:
            (_, stderr) = process.communicate(timeout=TIMEOUT)
        except TimeoutExpired:
            process.kill()
            raise
        process_stderr = stderr.strip().decode('utf-8')
        images = [('emulation-python3.8', 'latest'), ('emulation-python3.8-2', 'latest')]
        for (image, tag) in images:
            self.assertRegex(process_stderr, f'{image}-.+-{tag}: digest:')