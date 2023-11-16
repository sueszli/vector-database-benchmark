from __future__ import annotations
from typing import TYPE_CHECKING, Sequence
from airflow.models import BaseOperator
from airflow.providers.dingding.hooks.dingding import DingdingHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class DingdingOperator(BaseOperator):
    """
    This operator allows to send DingTalk message using Custom Robot API.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:DingdingOperator`

    :param dingding_conn_id: Dingding connection id that has access token in the password field,
        and optional host name in host field, if host not set than default
        ``https://oapi.dingtalk.com`` will use.
    :param message_type: Message type you want to send to Dingding, support five type so far
        including ``text``, ``link``, ``markdown``, ``actionCard``, ``feedCard``.
    :param message: The message send to chat group
    :param at_mobiles: Remind specific users with this message
    :param at_all: Remind all people in group or not. If True, will overwrite ``at_mobiles``
    """
    template_fields: Sequence[str] = ('message',)
    ui_color = '#4ea4d4'

    def __init__(self, *, dingding_conn_id: str='dingding_default', message_type: str='text', message: str | dict | None=None, at_mobiles: list[str] | None=None, at_all: bool=False, **kwargs) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.dingding_conn_id = dingding_conn_id
        self.message_type = message_type
        self.message = message
        self.at_mobiles = at_mobiles
        self.at_all = at_all

    def execute(self, context: Context) -> None:
        if False:
            i = 10
            return i + 15
        self.log.info('Sending Dingding message.')
        hook = DingdingHook(self.dingding_conn_id, self.message_type, self.message, self.at_mobiles, self.at_all)
        hook.send()