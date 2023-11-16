from enum import Enum
from typing import Optional
from botocore.exceptions import ClientError
from pydantic import BaseModel
from prowler.lib.logger import logger
from prowler.lib.scan_filters.scan_filters import is_resource_filtered
from prowler.providers.aws.lib.service.service import AWSService

class CodeArtifact(AWSService):

    def __init__(self, audit_info):
        if False:
            return 10
        super().__init__(__class__.__name__, audit_info)
        self.repositories = {}
        self.__threading_call__(self.__list_repositories__)
        self.__threading_call__(self.__list_packages__)
        self.__list_tags_for_resource__()

    def __list_repositories__(self, regional_client):
        if False:
            return 10
        logger.info('CodeArtifact - Listing Repositories...')
        try:
            list_repositories_paginator = regional_client.get_paginator('list_repositories')
            for page in list_repositories_paginator.paginate():
                for repository in page['repositories']:
                    if not self.audit_resources or is_resource_filtered(repository['arn'], self.audit_resources):
                        package_name = repository['name']
                        package_domain_name = repository['domainName']
                        package_domain_owner = repository['domainOwner']
                        package_arn = repository['arn']
                        self.repositories[package_arn] = Repository(name=package_name, arn=package_arn, domain_name=package_domain_name, domain_owner=package_domain_owner, region=regional_client.region)
        except Exception as error:
            logger.error(f'{regional_client.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}')

    def __list_packages__(self, regional_client):
        if False:
            return 10
        logger.info('CodeArtifact - Listing Packages and retrieving information...')
        for repository in self.repositories:
            try:
                if self.repositories[repository].region == regional_client.region:
                    list_packages_paginator = regional_client.get_paginator('list_packages')
                    list_packages_parameters = {'domain': self.repositories[repository].domain_name, 'domainOwner': self.repositories[repository].domain_owner, 'repository': repository}
                    packages = []
                    for page in list_packages_paginator.paginate(**list_packages_parameters):
                        for package in page['packages']:
                            package_format = package['format']
                            package_namespace = package.get('namespace')
                            package_name = package['package']
                            package_origin_configuration_restrictions_publish = package['originConfiguration']['restrictions']['publish']
                            package_origin_configuration_restrictions_upstream = package['originConfiguration']['restrictions']['upstream']
                            latest_version_information = regional_client.list_package_versions(domain=self.repositories[repository].domain_name, domainOwner=self.repositories[repository].domain_owner, repository=repository, format=package_format, package=package_name, sortBy='PUBLISHED_TIME')
                            latest_version = ''
                            latest_origin_type = 'UNKNOWN'
                            latest_status = 'Published'
                            if latest_version_information.get('versions'):
                                latest_version = latest_version_information['versions'][0].get('version')
                                latest_origin_type = latest_version_information['versions'][0].get('origin', {}).get('originType', 'UNKNOWN')
                                latest_status = latest_version_information['versions'][0].get('status', 'Published')
                            packages.append(Package(name=package_name, namespace=package_namespace, format=package_format, origin_configuration=OriginConfiguration(restrictions=Restrictions(publish=package_origin_configuration_restrictions_publish, upstream=package_origin_configuration_restrictions_upstream)), latest_version=LatestPackageVersion(version=latest_version, status=latest_status, origin=OriginInformation(origin_type=latest_origin_type))))
                    self.repositories[repository].packages = packages
            except ClientError as error:
                if error.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.warning(f'{regional_client.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}')
                    continue
            except Exception as error:
                logger.error(f'{regional_client.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}')

    def __list_tags_for_resource__(self):
        if False:
            while True:
                i = 10
        logger.info('CodeArtifact - List Tags...')
        try:
            for repository in self.repositories.values():
                regional_client = self.regional_clients[repository.region]
                response = regional_client.list_tags_for_resource(resourceArn=repository.arn)['tags']
                repository.tags = response
        except Exception as error:
            logger.error(f'{regional_client.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}')

class RestrictionValues(Enum):
    """Possible values for the package origin restriction"""
    ALLOW = 'ALLOW'
    BLOCK = 'BLOCK'

class Restrictions(BaseModel):
    """Information about the upstream and publish package origin restrictions"""
    publish: RestrictionValues
    upstream: RestrictionValues

class OriginConfiguration(BaseModel):
    """Details about the package origin configuration of a package"""
    restrictions: Restrictions

class OriginInformationValues(Enum):
    """Possible values for the OriginInformation"""
    INTERNAL = 'INTERNAL'
    EXTERNAL = 'EXTERNAL'
    UNKNOWN = 'UNKNOWN'

class OriginInformation(BaseModel):
    """
    Describes how the package version was originally added to the domain.

    An INTERNAL origin type means the package version was published directly to a repository in the domain.

    An EXTERNAL origin type means the package version was ingested from an external connection.
    """
    origin_type: OriginInformationValues

class LatestPackageVersionStatus(Enum):
    """Possible values for the package status"""
    Published = 'Published'
    Unfinished = 'Unfinished'
    Unlisted = 'Unlisted'
    Archived = 'Archived'
    Disposed = 'Disposed'
    Deleted = 'Deleted'

class LatestPackageVersion(BaseModel):
    """Details of the latest package version"""
    version: str
    status: LatestPackageVersionStatus
    origin: OriginInformation

class Package(BaseModel):
    """Details of a package"""
    name: str
    namespace: Optional[str]
    format: str
    origin_configuration: OriginConfiguration
    latest_version: LatestPackageVersion

class Repository(BaseModel):
    """Information about a Repository"""
    name: str
    arn: str
    domain_name: str
    domain_owner: str
    packages: list[Package] = []
    region: str
    tags: Optional[list] = []