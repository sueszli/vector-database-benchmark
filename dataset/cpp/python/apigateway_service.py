from typing import Optional

from pydantic import BaseModel

from prowler.lib.logger import logger
from prowler.lib.scan_filters.scan_filters import is_resource_filtered
from prowler.providers.aws.lib.service.service import AWSService


################## APIGateway
class APIGateway(AWSService):
    def __init__(self, audit_info):
        # Call AWSService's __init__
        super().__init__(__class__.__name__, audit_info)
        self.rest_apis = []
        self.__threading_call__(self.__get_rest_apis__)
        self.__get_authorizers__()
        self.__get_rest_api__()
        self.__get_stages__()

    def __get_rest_apis__(self, regional_client):
        logger.info("APIGateway - Getting Rest APIs...")
        try:
            get_rest_apis_paginator = regional_client.get_paginator("get_rest_apis")
            for page in get_rest_apis_paginator.paginate():
                for apigw in page["items"]:
                    arn = f"arn:{self.audited_partition}:apigateway:{regional_client.region}::/restapis/{apigw['id']}"
                    if not self.audit_resources or (
                        is_resource_filtered(arn, self.audit_resources)
                    ):
                        self.rest_apis.append(
                            RestAPI(
                                id=apigw["id"],
                                arn=arn,
                                region=regional_client.region,
                                name=apigw.get("name", ""),
                                tags=[apigw.get("tags")],
                            )
                        )
        except Exception as error:
            logger.error(
                f"{regional_client.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}"
            )

    def __get_authorizers__(self):
        logger.info("APIGateway - Getting Rest APIs authorizer...")
        try:
            for rest_api in self.rest_apis:
                regional_client = self.regional_clients[rest_api.region]
                authorizers = regional_client.get_authorizers(restApiId=rest_api.id)[
                    "items"
                ]
                if authorizers:
                    rest_api.authorizer = True
        except Exception as error:
            logger.error(f"{error.__class__.__name__}: {error}")

    def __get_rest_api__(self):
        logger.info("APIGateway - Describing Rest API...")
        try:
            for rest_api in self.rest_apis:
                regional_client = self.regional_clients[rest_api.region]
                rest_api_info = regional_client.get_rest_api(restApiId=rest_api.id)
                if rest_api_info["endpointConfiguration"]["types"] == ["PRIVATE"]:
                    rest_api.public_endpoint = False
        except Exception as error:
            logger.error(f"{error.__class__.__name__}: {error}")

    def __get_stages__(self):
        logger.info("APIGateway - Getting stages for Rest APIs...")
        try:
            for rest_api in self.rest_apis:
                regional_client = self.regional_clients[rest_api.region]
                stages = regional_client.get_stages(restApiId=rest_api.id)
                for stage in stages["item"]:
                    waf = None
                    logging = False
                    client_certificate = False
                    if "webAclArn" in stage:
                        waf = stage["webAclArn"]
                    if "methodSettings" in stage:
                        if stage["methodSettings"]:
                            logging = True
                    if "clientCertificateId" in stage:
                        client_certificate = True
                    arn = f"arn:{self.audited_partition}:apigateway:{regional_client.region}::/restapis/{rest_api.id}/stages/{stage['stageName']}"
                    rest_api.stages.append(
                        Stage(
                            name=stage["stageName"],
                            arn=arn,
                            logging=logging,
                            client_certificate=client_certificate,
                            waf=waf,
                            tags=[stage.get("tags")],
                        )
                    )
        except Exception as error:
            logger.error(f"{error.__class__.__name__}: {error}")


class Stage(BaseModel):
    name: str
    arn: str
    logging: bool
    client_certificate: bool
    waf: Optional[str]
    tags: Optional[list] = []


class RestAPI(BaseModel):
    id: str
    arn: str
    region: str
    name: str
    authorizer: bool = False
    public_endpoint: bool = True
    stages: list[Stage] = []
    tags: Optional[list] = []
