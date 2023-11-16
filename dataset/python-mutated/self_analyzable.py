from typing import Any, Optional, Tuple
from django.core.files import File
from api_app.pivots_manager.classes import Pivot
from api_app.pivots_manager.models import PivotConfig

class SelfAnalyzable(Pivot):

    def should_run(self) -> Tuple[bool, Optional[str]]:
        if False:
            return 10
        self._config: PivotConfig
        (to_run, motivation) = super().should_run()
        if to_run:
            related_config_class = self.related_configs.model
            related_configs_pk = set(self.related_configs.values_list('pk', flat=True))
            playbook_configs = set(related_config_class.objects.filter(playbooks=self._config.playbook_to_execute).values_list('pk', flat=True))
            if related_configs_pk.issubset(playbook_configs):
                return (False, f'Found infinite loop in {self._config.name}.')
        return (to_run, motivation)

    def get_value_to_pivot_to(self) -> Any:
        if False:
            return 10
        if self._job.is_sample:
            return File(self._job.analyzed_object, name=self._job.analyzed_object_name)
        else:
            return self._job.analyzed_object_name