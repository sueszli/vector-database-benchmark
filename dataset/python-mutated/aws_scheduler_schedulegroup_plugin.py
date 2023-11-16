from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class SchedulerScheduleGroupProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::Scheduler::ScheduleGroup'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            i = 10
            return i + 15
        from localstack.services.scheduler.resource_providers.aws_scheduler_schedulegroup import SchedulerScheduleGroupProvider
        self.factory = SchedulerScheduleGroupProvider