from unittest import skipIf
from parameterized import parameterized
from .regression_package_base import PackageRegressionBase
from tests.testing_utils import RUNNING_ON_CI, RUNNING_TEST_FOR_MASTER_ON_CI, RUN_BY_CANARY
SKIP_PACKAGE_REGRESSION_TESTS = RUNNING_ON_CI and RUNNING_TEST_FOR_MASTER_ON_CI and (not RUN_BY_CANARY)

@skipIf(SKIP_PACKAGE_REGRESSION_TESTS, 'Skip package regression tests in CI/CD only')
class TestPackageRegression(PackageRegressionBase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()

    def tearDown(self):
        if False:
            return 10
        super().tearDown()

    @parameterized.expand([('aws-serverless-api.yaml', True), ('aws-appsync-graphqlschema.yaml', True), ('aws-appsync-resolver.yaml', True), ('aws-appsync-functionconfiguration.yaml', True), ('aws-apigateway-restapi.yaml', True), ('aws-elasticbeanstalk-applicationversion.yaml', True), ('aws-cloudformation-stack-regression.yaml', False), ('aws-cloudformation-stack-regression.yaml', False)])
    def test_package_with_output_template_file(self, template_file, skip_sam_metadata=False):
        if False:
            return 10
        arguments = {'s3_bucket': self.s3_bucket.name, 'template_file': self.test_data_path.joinpath(template_file)}
        self.regression_check(arguments, skip_sam_metadata)

    @parameterized.expand([('aws-serverless-api.yaml', True), ('aws-appsync-graphqlschema.yaml', True), ('aws-appsync-resolver.yaml', True), ('aws-appsync-functionconfiguration.yaml', True), ('aws-apigateway-restapi.yaml', True), ('aws-elasticbeanstalk-applicationversion.yaml', True), ('aws-cloudformation-stack-regression.yaml', False), ('aws-cloudformation-stack-regression.yaml', False)])
    def test_package_with_output_template_file_and_prefix(self, template_file, skip_sam_metadata=False):
        if False:
            i = 10
            return i + 15
        arguments = {'s3_bucket': self.s3_bucket.name, 'template_file': self.test_data_path.joinpath(template_file), 's3_prefix': 'regression/tests'}
        self.regression_check(arguments, skip_sam_metadata)

    @parameterized.expand([('aws-serverless-api.yaml', True), ('aws-appsync-graphqlschema.yaml', True), ('aws-appsync-resolver.yaml', True), ('aws-appsync-functionconfiguration.yaml', True), ('aws-apigateway-restapi.yaml', True), ('aws-elasticbeanstalk-applicationversion.yaml', True), ('aws-cloudformation-stack-regression.yaml', False), ('aws-cloudformation-stack-regression.yaml', False)])
    def test_package_with_output_template_file_json_and_prefix(self, template_file, skip_sam_metadata=False):
        if False:
            for i in range(10):
                print('nop')
        arguments = {'s3_bucket': self.s3_bucket.name, 'template_file': self.test_data_path.joinpath(template_file), 's3_prefix': 'regression/tests', 'use_json': True}
        self.regression_check(arguments, skip_sam_metadata)