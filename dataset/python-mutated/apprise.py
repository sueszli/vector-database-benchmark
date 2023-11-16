from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Iterable
from airflow.exceptions import AirflowOptionalProviderFeatureException
try:
    from airflow.notifications.basenotifier import BaseNotifier
except ImportError:
    raise AirflowOptionalProviderFeatureException('Failed to import BaseNotifier. This feature is only available in Airflow versions >= 2.6.0')
from airflow.providers.apprise.hooks.apprise import AppriseHook
if TYPE_CHECKING:
    from apprise import AppriseConfig, NotifyFormat, NotifyType

class AppriseNotifier(BaseNotifier):
    """
    Apprise BaseNotifier.

    :param body: Specify the message body
    :param title: Specify the message title. This field is complete optional
    :param notify_type: Specify the message type (default=info). Possible values are "info",
        "success", "failure", and "warning"
    :param body_format: Specify the input message format (default=text). Possible values are "text",
        "html", and "markdown".
    :param tag: Specify one or more tags to filter which services to notify
    :param attach: Specify one or more file attachment locations
    :param interpret_escapes: Enable interpretation of backslash escapes. For example, this would convert
        sequences such as \\n and \\r to their respected ascii new-line and carriage
    :param config: Specify one or more configuration
    :param apprise_conn_id: connection that has Apprise configs setup
    """
    template_fields = ('body', 'title', 'tag', 'attach')

    def __init__(self, *, body: str, title: str | None=None, notify_type: NotifyType | None=None, body_format: NotifyFormat | None=None, tag: str | Iterable[str] | None=None, attach: str | None=None, interpret_escapes: bool | None=None, config: AppriseConfig | None=None, apprise_conn_id: str=AppriseHook.default_conn_name):
        if False:
            return 10
        super().__init__()
        self.apprise_conn_id = apprise_conn_id
        self.body = body
        self.title = title
        self.notify_type = notify_type
        self.body_format = body_format
        self.tag = tag
        self.attach = attach
        self.interpret_escapes = interpret_escapes
        self.config = config

    @cached_property
    def hook(self) -> AppriseHook:
        if False:
            print('Hello World!')
        'Apprise Hook.'
        return AppriseHook(apprise_conn_id=self.apprise_conn_id)

    def notify(self, context):
        if False:
            i = 10
            return i + 15
        'Send a alert to a apprise configured service.'
        self.hook.notify(body=self.body, title=self.title, notify_type=self.notify_type, body_format=self.body_format, tag=self.tag, attach=self.attach, interpret_escapes=self.interpret_escapes, config=self.config)
send_apprise_notification = AppriseNotifier