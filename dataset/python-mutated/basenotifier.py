from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING, Sequence
from airflow.template.templater import Templater
from airflow.utils.context import context_merge
if TYPE_CHECKING:
    import jinja2
    from airflow import DAG
    from airflow.utils.context import Context

class BaseNotifier(Templater):
    """BaseNotifier class for sending notifications."""
    template_fields: Sequence[str] = ()
    template_ext: Sequence[str] = ()

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.resolve_template_files()

    def _update_context(self, context: Context) -> Context:
        if False:
            i = 10
            return i + 15
        '\n        Add additional context to the context.\n\n        :param context: The airflow context\n        '
        additional_context = ((f, getattr(self, f)) for f in self.template_fields)
        context_merge(context, additional_context)
        return context

    def _render(self, template, context, dag: DAG | None=None):
        if False:
            i = 10
            return i + 15
        dag = dag or context['dag']
        return super()._render(template, context, dag)

    def render_template_fields(self, context: Context, jinja_env: jinja2.Environment | None=None) -> None:
        if False:
            print('Hello World!')
        'Template all attributes listed in *self.template_fields*.\n\n        This mutates the attributes in-place and is irreversible.\n\n        :param context: Context dict with values to apply on content.\n        :param jinja_env: Jinja environment to use for rendering.\n        '
        dag = context['dag']
        if not jinja_env:
            jinja_env = self.get_template_env(dag=dag)
        self._do_render_template_fields(self, self.template_fields, context, jinja_env, set())

    @abstractmethod
    def notify(self, context: Context) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Send a notification.\n\n        :param context: The airflow context\n        '
        ...

    def __call__(self, *args) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Send a notification.\n\n        :param context: The airflow context\n        '
        if len(args) == 1:
            context = args[0]
        else:
            context = {'dag': args[0], 'task_list': args[1], 'blocking_task_list': args[2], 'slas': args[3], 'blocking_tis': args[4]}
        self._update_context(context)
        self.render_template_fields(context)
        try:
            self.notify(context)
        except Exception as e:
            self.log.exception('Failed to send notification: %s', e)