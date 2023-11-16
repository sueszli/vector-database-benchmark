from typing import Optional, Type
from localstack.services.cloudformation.resource_provider import CloudFormationResourceProviderPlugin, ResourceProvider

class SchedulerScheduleProviderPlugin(CloudFormationResourceProviderPlugin):
    name = 'AWS::Scheduler::Schedule'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.factory: Optional[Type[ResourceProvider]] = None

    def load(self):
        if False:
            while True:
                i = 10
        from localstack.services.scheduler.resource_providers.aws_scheduler_schedule import SchedulerScheduleProvider
        self.factory = SchedulerScheduleProvider