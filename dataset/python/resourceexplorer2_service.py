from pydantic import BaseModel

from prowler.lib.logger import logger
from prowler.lib.scan_filters.scan_filters import is_resource_filtered
from prowler.providers.aws.lib.service.service import AWSService


################################ ResourceExplorer2
class ResourceExplorer2(AWSService):
    def __init__(self, audit_info):
        # Call AWSService's __init__
        super().__init__("resource-explorer-2", audit_info)
        self.indexes = []
        self.__threading_call__(self.__list_indexes__)

    def __list_indexes__(self, regional_client):
        logger.info("ResourceExplorer - list indexes...")
        try:
            list_indexes_paginator = regional_client.get_paginator("list_indexes")
            for page in list_indexes_paginator.paginate():
                for index in page.get("Indexes"):
                    if not self.audit_resources or (
                        is_resource_filtered(index["Arn"], self.audit_resources)
                    ):
                        self.indexes.append(
                            Indexes(
                                arn=index["Arn"],
                                region=index["Region"],
                                type=index["Type"],
                            )
                        )
        except Exception as error:
            logger.error(
                f"{regional_client.region} -- {error.__class__.__name__}[{error.__traceback__.tb_lineno}]: {error}"
            )


class Indexes(BaseModel):
    arn: str
    region: str
    type: str
