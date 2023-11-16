from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.models.baseoperator import BaseOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class SmoothOperator(BaseOperator):
    """Operator that does nothing, it logs a YouTube link to Sade song "Smooth Operator"."""
    ui_color = '#e8f7e4'
    yt_link: str = 'https://www.youtube.com/watch?v=4TYv2PhG89A'

    def __init__(self, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)

    def execute(self, context: Context):
        if False:
            i = 10
            return i + 15
        self.log.info('Enjoy Sade - Smooth Operator: %s', self.yt_link)