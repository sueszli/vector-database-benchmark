from ..base import ModelDeletionTask, ModelRelation

class MonitorEnvironmentDeletionTask(ModelDeletionTask):

    def get_child_relations(self, instance):
        if False:
            while True:
                i = 10
        from sentry.monitors import models
        return [ModelRelation(models.MonitorCheckIn, {'monitor_environment_id': instance.id}, ModelDeletionTask)]