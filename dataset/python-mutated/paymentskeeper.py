import datetime
import logging
from typing import Iterable, List, Optional
from golem import model
from golem.core.common import to_unicode, datetime_to_timestamp_utc
logger = logging.getLogger(__name__)

class PaymentsDatabase(object):
    """Save and retrieve from database information
       about payments that this node has to make / made
    """

    @staticmethod
    def get_payment_value(subtask_id: str):
        if False:
            print('Hello World!')
        'Returns value of a payment\n           that was done to the same node and for the same\n           task as payment for payment_info\n        '
        return PaymentsDatabase.get_payment_for_subtask(subtask_id)

    @staticmethod
    def get_payment_for_subtask(subtask_id):
        if False:
            print('Hello World!')
        try:
            return model.TaskPayment.get(model.TaskPayment.subtask == subtask_id).wallet_operation.amount
        except model.TaskPayment.DoesNotExist:
            logger.debug("Can't get payment value - payment does not exist")
            return 0

    @staticmethod
    def get_subtasks_payments(subtask_ids: Iterable[str]) -> List[model.TaskPayment]:
        if False:
            for i in range(10):
                print('nop')
        return list(model.TaskPayment.payments().where(model.TaskPayment.subtask.in_(subtask_ids)))

    @staticmethod
    def get_newest_payment(num: Optional[int]=None, interval: Optional[datetime.timedelta]=None):
        if False:
            while True:
                i = 10
        ' Return specific number of recently modified payments\n        :param num: Number of payments to return. Unlimited if None.\n        :param interval: Return payments from last interval of time. Unlimited\n                         if None.\n        :return:\n        '
        query = model.TaskPayment.payments().order_by(model.WalletOperation.modified_date.desc())
        if interval is not None:
            then = datetime.datetime.now(tz=datetime.timezone.utc) - interval
            query = query.where(model.WalletOperation.modified_date >= then)
        if num is not None:
            query = query.limit(num)
        return query.execute()

class PaymentsKeeper:
    """Keeps information about outgoing payments
       that should be processed and send or received.
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        ' Create new payments keeper instance'
        self.db = PaymentsDatabase()

    def get_list_of_all_payments(self, num: Optional[int]=None, interval: Optional[datetime.timedelta]=None):
        if False:
            while True:
                i = 10
        return [{'subtask': to_unicode(payment.subtask), 'payee': to_unicode(payment.wallet_operation.recipient_address), 'value': to_unicode(payment.wallet_operation.amount), 'status': to_unicode(payment.wallet_operation.status.name), 'fee': to_unicode(payment.wallet_operation.gas_cost), 'block_number': '', 'transaction': to_unicode(payment.wallet_operation.tx_hash), 'node': payment.node, 'created': datetime_to_timestamp_utc(payment.created_date), 'modified': datetime_to_timestamp_utc(payment.wallet_operation.modified_date)} for payment in self.db.get_newest_payment(num, interval)]

    def get_payment(self, subtask_id):
        if False:
            return 10
        '\n        Get cost of subtasks defined by @subtask_id\n        :param subtask_id: Subtask ID\n        :return: Cost of the @subtask_id\n        '
        return self.db.get_payment_for_subtask(subtask_id)

    def get_subtasks_payments(self, subtask_ids: Iterable[str]) -> List[model.TaskPayment]:
        if False:
            for i in range(10):
                print('nop')
        return self.db.get_subtasks_payments(subtask_ids)

    @staticmethod
    def confirmed_transfer(tx_hash: str, successful: bool, gas_cost: int) -> None:
        if False:
            i = 10
            return i + 15
        try:
            operation = model.WalletOperation.select().where(model.WalletOperation.tx_hash == tx_hash).get()
        except model.WalletOperation.DoesNotExist:
            logger.warning('Got confirmation of unknown transfer. tx_hash=%s', tx_hash)
            return
        if not successful:
            logger.error('Failed transaction. tx_hash=%s', tx_hash)
            operation.on_failed(gas_cost=gas_cost)
            operation.save()
            return
        operation.on_confirmed(gas_cost=gas_cost)
        operation.save()