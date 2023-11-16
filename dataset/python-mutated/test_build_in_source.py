import os
import shutil
import pytest
import logging
from unittest import skip
from parameterized import parameterized
from tests.integration.buildcmd.build_integ_base import BuildIntegProvidedBase, BuildIntegEsbuildBase
LOG = logging.getLogger(__name__)

@skip('Building in source option is not exposed yet. Stop skipping once it is.')
class TestBuildCommand_BuildInSource_Makefile(BuildIntegProvidedBase):
    template = 'template.yaml'
    is_nested_parent = False

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        cls.code_uri = 'provided_create_new_file'
        cls.code_uri_path = os.path.join(cls.test_data_path, cls.code_uri)
        cls.file_created_from_make_command = 'file-created-from-make-command.txt'

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        new_file_in_codeuri_path = os.path.join(self.code_uri_path, self.file_created_from_make_command)
        if os.path.isfile(new_file_in_codeuri_path):
            os.remove(new_file_in_codeuri_path)

    @parameterized.expand([(True, True), (False, False), (None, False)])
    @pytest.mark.flaky(reruns=3)
    def test_builds_successfully_with_makefile(self, build_in_source, new_file_should_be_in_codeuri):
        if False:
            while True:
                i = 10
        self._test_with_Makefile(runtime='provided.al2', use_container=False, manifest=None, code_uri=self.code_uri, build_in_source=build_in_source)
        self.assertEqual(self.file_created_from_make_command in os.listdir(self.code_uri_path), new_file_should_be_in_codeuri)

@skip('Building in source option is not exposed yet. Stop skipping once it is.')
class TestBuildCommand_BuildInSource_Esbuild(BuildIntegEsbuildBase):
    is_nested_parent = False

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.source_directories = []

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        for source in self.source_directories:
            shutil.rmtree(os.path.join(source, 'node_modules'), ignore_errors=True)

    @parameterized.expand([(True, True), (False, False), (None, False)])
    @pytest.mark.flaky(reruns=3)
    def test_builds_successfully_without_local_dependencies(self, build_in_source, dependencies_expected_in_source):
        if False:
            print('Hello World!')
        self.template_path = os.path.join(self.test_data_path, 'template_with_metadata.yaml')
        codeuri = os.path.join(self.test_data_path, 'Esbuild', 'Node')
        self.source_directories = [codeuri]
        self._test_with_default_package_json(build_in_source=build_in_source, runtime='nodejs16.x', code_uri=codeuri, handler='main.lambdaHandler', architecture='x86_64', use_container=False, expected_files={'main.js', 'main.js.map'})
        self.assertEqual(os.path.isdir(os.path.join(codeuri, 'node_modules')), dependencies_expected_in_source)

    @pytest.mark.flaky(reruns=3)
    def test_builds_successfully_with_local_dependency(self):
        if False:
            for i in range(10):
                print('nop')
        self.template_path = os.path.join(self.test_data_path, 'template_with_metadata.yaml')
        codeuri = os.path.join(self.test_data_path, 'Esbuild', 'NodeWithLocalDependency')
        self.source_directories = [codeuri]
        runtime = 'nodejs16.x'
        architecture = 'x86_64'
        self._test_with_default_package_json(build_in_source=True, runtime=runtime, code_uri=codeuri, handler='main.lambdaHandler', architecture=architecture, use_container=False, expected_files={'main.js', 'main.js.map'})
        self.assertEqual(os.path.isdir(os.path.join(codeuri, 'node_modules')), True)