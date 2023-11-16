from unittest import mock
from moto.core import DEFAULT_ACCOUNT_ID
from prowler.providers.aws.services.codeartifact.codeartifact_service import LatestPackageVersion, LatestPackageVersionStatus, OriginConfiguration, OriginInformation, OriginInformationValues, Package, Repository, Restrictions, RestrictionValues
AWS_REGION = 'eu-west-1'

class Test_codeartifact_packages_external_public_publishing_disabled:

    def test_no_repositories(self):
        if False:
            for i in range(10):
                print('nop')
        codeartifact_client = mock.MagicMock
        codeartifact_client.repositories = {}
        with mock.patch('prowler.providers.aws.services.codeartifact.codeartifact_service.CodeArtifact', new=codeartifact_client):
            from prowler.providers.aws.services.codeartifact.codeartifact_packages_external_public_publishing_disabled.codeartifact_packages_external_public_publishing_disabled import codeartifact_packages_external_public_publishing_disabled
            check = codeartifact_packages_external_public_publishing_disabled()
            result = check.execute()
            assert len(result) == 0

    def test_repository_without_packages(self):
        if False:
            return 10
        codeartifact_client = mock.MagicMock
        codeartifact_client.repositories = {'test-repository': Repository(name='test-repository', arn='', domain_name='', domain_owner='', region=AWS_REGION, packages=[])}
        with mock.patch('prowler.providers.aws.services.codeartifact.codeartifact_service.CodeArtifact', new=codeartifact_client):
            from prowler.providers.aws.services.codeartifact.codeartifact_packages_external_public_publishing_disabled.codeartifact_packages_external_public_publishing_disabled import codeartifact_packages_external_public_publishing_disabled
            check = codeartifact_packages_external_public_publishing_disabled()
            result = check.execute()
            assert len(result) == 0

    def test_repository_package_public_publishing_origin_internal(self):
        if False:
            i = 10
            return i + 15
        codeartifact_client = mock.MagicMock
        package_name = 'test-package'
        package_namespace = 'test-namespace'
        repository_arn = f'arn:aws:codebuild:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:repository/test-repository'
        codeartifact_client.repositories = {'test-repository': Repository(name='test-repository', arn=repository_arn, domain_name='', domain_owner='', region=AWS_REGION, packages=[Package(name=package_name, namespace=package_namespace, format='pypi', origin_configuration=OriginConfiguration(restrictions=Restrictions(publish=RestrictionValues.ALLOW, upstream=RestrictionValues.ALLOW)), latest_version=LatestPackageVersion(version='latest', status=LatestPackageVersionStatus.Published, origin=OriginInformation(origin_type=OriginInformationValues.INTERNAL)))])}
        with mock.patch('prowler.providers.aws.services.codeartifact.codeartifact_service.CodeArtifact', new=codeartifact_client):
            from prowler.providers.aws.services.codeartifact.codeartifact_packages_external_public_publishing_disabled.codeartifact_packages_external_public_publishing_disabled import codeartifact_packages_external_public_publishing_disabled
            check = codeartifact_packages_external_public_publishing_disabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == AWS_REGION
            assert result[0].resource_id == 'test-package'
            assert result[0].resource_arn == repository_arn
            assert result[0].resource_tags == []
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'Internal package {package_name} is vulnerable to dependency confusion in repository {repository_arn}.'

    def test_repository_package_private_publishing_origin_internal(self):
        if False:
            while True:
                i = 10
        codeartifact_client = mock.MagicMock
        package_name = 'test-package'
        package_namespace = 'test-namespace'
        repository_arn = f'arn:aws:codebuild:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:repository/test-repository'
        codeartifact_client.repositories = {'test-repository': Repository(name='test-repository', arn=repository_arn, domain_name='', domain_owner='', region=AWS_REGION, packages=[Package(name=package_name, namespace=package_namespace, format='pypi', origin_configuration=OriginConfiguration(restrictions=Restrictions(publish=RestrictionValues.BLOCK, upstream=RestrictionValues.BLOCK)), latest_version=LatestPackageVersion(version='latest', status=LatestPackageVersionStatus.Published, origin=OriginInformation(origin_type=OriginInformationValues.INTERNAL)))])}
        with mock.patch('prowler.providers.aws.services.codeartifact.codeartifact_service.CodeArtifact', new=codeartifact_client):
            from prowler.providers.aws.services.codeartifact.codeartifact_packages_external_public_publishing_disabled.codeartifact_packages_external_public_publishing_disabled import codeartifact_packages_external_public_publishing_disabled
            check = codeartifact_packages_external_public_publishing_disabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == AWS_REGION
            assert result[0].resource_id == 'test-package'
            assert result[0].resource_arn == repository_arn
            assert result[0].resource_tags == []
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'Internal package {package_name} is not vulnerable to dependency confusion in repository {repository_arn}.'