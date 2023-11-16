"""
A service that checks for flow run notifications and sends them.
"""
import asyncio
from uuid import UUID
import sqlalchemy as sa
from prefect.server import models, schemas
from prefect.server.database.dependencies import inject_db
from prefect.server.database.interface import PrefectDBInterface
from prefect.server.services.loop_service import LoopService
from prefect.settings import PREFECT_UI_URL

class FlowRunNotifications(LoopService):
    """
    A loop service that checks for flow run notifications that need to be sent.

    Notifications are queued, and this service pulls them off the queue and
    actually sends the notification.
    """
    loop_seconds: int = 4

    @inject_db
    async def run_once(self, db: PrefectDBInterface):
        while True:
            async with db.session_context(begin_transaction=True) as session:
                notifications = await db.get_flow_run_notifications_from_queue(session=session, limit=1)
                self.logger.debug(f'Got {len(notifications)} notifications from queue.')
                if not notifications:
                    break
                assert len(notifications) == 1, 'Expected one notification; query limit not respected.'
                try:
                    await self.send_flow_run_notification(session=session, db=db, notification=notifications[0])
                finally:
                    connection = await session.connection()
                    if connection.invalidated:
                        await session.rollback()
                        assert not connection.invalidated

    @inject_db
    async def send_flow_run_notification(self, session: sa.orm.session, db: PrefectDBInterface, notification):
        try:
            orm_block_document = await session.get(db.BlockDocument, notification.block_document_id)
            if orm_block_document is None:
                self.logger.error(f'Missing block document {notification.block_document_id} from policy {notification.flow_run_notification_policy_id}')
                return
            from prefect.blocks.core import Block
            block = Block._from_block_document(await schemas.core.BlockDocument.from_orm_model(session=session, orm_block_document=orm_block_document, include_secrets=True))
            message = self.construct_notification_message(notification=notification)
            await block.notify(subject='Prefect flow run notification', body=message)
            self.logger.debug(f'Successfully sent notification for flow run {notification.flow_run_id} from policy {notification.flow_run_notification_policy_id}')
        except Exception:
            self.logger.error(f'Error sending notification for policy {notification.flow_run_notification_policy_id} on flow run {notification.flow_run_id}', exc_info=True)

    def construct_notification_message(self, notification) -> str:
        if False:
            return 10
        '\n        Construct the message for a flow run notification, including\n        templating any variables.\n        '
        message_template = notification.flow_run_notification_policy_message_template or models.flow_run_notification_policies.DEFAULT_MESSAGE_TEMPLATE
        notification_dict = dict(notification._mapping)
        notification_dict['flow_run_url'] = self.get_ui_url_for_flow_run_id(flow_run_id=notification_dict['flow_run_id'])
        message = message_template.format(**{k: notification_dict[k] for k in schemas.core.FLOW_RUN_NOTIFICATION_TEMPLATE_KWARGS})
        return message

    def get_ui_url_for_flow_run_id(self, flow_run_id: UUID) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Returns a link to the flow run view of the given flow run id.\n\n        Args:\n            flow_run_id: the flow run id.\n        '
        ui_url = PREFECT_UI_URL.value() or 'http://ephemeral-prefect/api'
        return f'{ui_url}/flow-runs/flow-run/{flow_run_id}'
if __name__ == '__main__':
    asyncio.run(FlowRunNotifications().start())