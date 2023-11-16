from botocore.client import ClientError
from pydantic import BaseModel
from prowler.lib.logger import logger
from prowler.lib.scan_filters.scan_filters import is_resource_filtered
from prowler.providers.aws.lib.service.service import AWSService

class SSMIncidents(AWSService):

    def __init__(self, audit_info):
        if False:
            print('Hello World!')
        super().__init__('ssm-incidents', audit_info)
        self.replication_set = []
        self.__list_replication_sets__()
        self.__get_replication_set__()
        self.response_plans = []
        self.__threading_call__(self.__list_response_plans__)
        self.__list_tags_for_resource__()

    def __list_replication_sets__(self):
        if False:
            i = 10
            return i + 15
        logger.info('SSMIncidents - Listing Replication Sets...')
        try:
            if self.regional_clients:
                regional_client = self.regional_clients[list(self.regional_clients.keys())[0]]
                list_replication_sets = regional_client.list_replication_sets().get('replicationSetArns')
                if list_replication_sets:
                    replication_set = list_replication_sets[0]
                    if not self.audit_resources or is_resource_filtered(replication_set, self.audit_resources):
                        self.replication_set = [ReplicationSet(arn=replication_set)]
        except Exception as error:
            logger.error(f'{error.__class__.__name__}:{error.__traceback__.tb_lineno} -- {error}')

    def __get_replication_set__(self):
        if False:
            i = 10
            return i + 15
        logger.info('SSMIncidents - Getting Replication Sets...')
        try:
            if not self.replication_set:
                return
            replication_set = self.replication_set[0]
            for regional_client in self.regional_clients.values():
                try:
                    get_replication_set = regional_client.get_replication_set(arn=replication_set.arn)['replicationSet']
                    replication_set.status = get_replication_set['status']
                    for region in get_replication_set['regionMap']:
                        replication_set.region_map.append(RegionMap(status=get_replication_set['regionMap'][region]['status'], region=region, sse_kms_id=get_replication_set['regionMap'][region]['sseKmsKeyId']))
                    break
                except ClientError as error:
                    if error.response['Error']['Code'] == 'ResourceNotFoundException':
                        continue
                    else:
                        logger.error(f'{self.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}')
        except Exception as error:
            logger.error(f'{error.__class__.__name__}:{error.__traceback__.tb_lineno} -- {error}')

    def __list_response_plans__(self, regional_client):
        if False:
            print('Hello World!')
        logger.info('SSMIncidents - Listing Response Plans...')
        try:
            list_response_plans_paginator = regional_client.get_paginator('list_response_plans')
            for page in list_response_plans_paginator.paginate():
                for response_plan in page['responsePlanSummaries']:
                    self.response_plans.append(ResponsePlan(arn=response_plan.get('Arn', ''), region=regional_client.region, name=response_plan.get('Name', '')))
        except Exception as error:
            logger.error(f'{error.__class__.__name__}:{error.__traceback__.tb_lineno} -- {error}')

    def __list_tags_for_resource__(self):
        if False:
            return 10
        logger.info('SSMIncidents - List Tags...')
        try:
            for response_plan in self.response_plans:
                regional_client = self.regional_clients[response_plan.region]
                response = regional_client.list_tags_for_resource(resourceArn=response_plan.arn)['tags']
                response_plan.tags = response
        except Exception as error:
            logger.error(f'{error.__class__.__name__}:{error.__traceback__.tb_lineno} -- {error}')

class RegionMap(BaseModel):
    status: str
    region: str
    sse_kms_id: str

class ReplicationSet(BaseModel):
    arn: str
    status: str = None
    region_map: list[RegionMap] = []

class ResponsePlan(BaseModel):
    arn: str
    name: str
    region: str
    tags: list = None