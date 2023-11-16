"""Custom logging formatter for Airflow."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from airflow.configuration import conf
from airflow.utils.helpers import parse_template_string, render_template_to_string
if TYPE_CHECKING:
    from jinja2 import Template
    from airflow.models.taskinstance import TaskInstance

class TaskHandlerWithCustomFormatter(logging.StreamHandler):
    """Custom implementation of StreamHandler, a class which writes logging records for Airflow."""
    prefix_jinja_template: Template | None = None

    def set_context(self, ti) -> None:
        if False:
            print('Hello World!')
        '\n        Accept the run-time context (i.e. the current task) and configure the formatter accordingly.\n\n        :param ti:\n        :return:\n        '
        if ti.raw or self.formatter is None:
            return
        prefix = conf.get('logging', 'task_log_prefix_template')
        if prefix:
            (_, self.prefix_jinja_template) = parse_template_string(prefix)
            rendered_prefix = self._render_prefix(ti)
        else:
            rendered_prefix = ''
        formatter = logging.Formatter(f'{rendered_prefix}:{self.formatter._fmt}')
        self.setFormatter(formatter)
        self.setLevel(self.level)

    def _render_prefix(self, ti: TaskInstance) -> str:
        if False:
            while True:
                i = 10
        if self.prefix_jinja_template:
            jinja_context = ti.get_template_context()
            return render_template_to_string(self.prefix_jinja_template, jinja_context)
        logging.warning("'task_log_prefix_template' is in invalid format, ignoring the variable value")
        return ''