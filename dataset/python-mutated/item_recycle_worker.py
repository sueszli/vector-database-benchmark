import time
from pokemongo_bot.worker_result import WorkerResult
from pokemongo_bot.base_task import BaseTask
from pokemongo_bot.tree_config_builder import ConfigException
RECYCLE_REQUEST_RESPONSE_SUCCESS = 1

class ItemRecycler(BaseTask):
    """
    This class contains details of recycling process.
    """
    SUPPORTED_TASK_API_VERSION = 1

    def __init__(self, bot, item_to_recycle, amount_to_recycle):
        if False:
            return 10
        '\n        Initialise an instance of ItemRecycler\n        :param bot: The instance of the Bot\n        :type bot: pokemongo_bot.PokemonGoBot\n        :param item_to_recycle: The item to recycle\n        :type item_to_recycle: inventory.Item\n        :param amount_to_recycle: The amount to recycle\n        :type amount_to_recycle: int\n        :return: Nothing.\n        :rtype: None\n        '
        self.bot = bot
        self.item_to_recycle = item_to_recycle
        self.amount_to_recycle = amount_to_recycle
        self.recycle_item_request_result = None
        self.last_log_time = time.time()

    def work(self):
        if False:
            while True:
                i = 10
        '\n        Start the recycling process\n        :return: Returns whether or not the task went well\n        :rtype: WorkerResult\n        '
        if self.should_run():
            self._request_recycle()
            if self.is_recycling_success():
                self._emit_recycle_succeed()
                return WorkerResult.SUCCESS
            else:
                self._emit_recycle_failed()
                return WorkerResult.ERROR

    def should_run(self):
        if False:
            return 10
        '\n        Returns a value indicating whether or not the recycler should be run.\n        :return: True if the recycler should be run; otherwise, False.\n        :rtype: bool\n        '
        if self.amount_to_recycle > 0 and self.item_to_recycle is not None:
            return True
        return False

    def _request_recycle(self):
        if False:
            while True:
                i = 10
        "\n        Request recycling of the item and store api call response's result.\n        :return: Nothing.\n        :rtype: None\n        "
        response = self.bot.api.recycle_inventory_item(item_id=self.item_to_recycle.id, count=self.amount_to_recycle)
        self.recycle_item_request_result = response.get('responses', {}).get('RECYCLE_INVENTORY_ITEM', {}).get('result', 0)

    def is_recycling_success(self):
        if False:
            while True:
                i = 10
        '\n        Returns a value indicating whether or not the item has been successfully recycled.\n        :return: True if the item has been successfully recycled; otherwise, False.\n        :rtype: bool\n        '
        return self.recycle_item_request_result == RECYCLE_REQUEST_RESPONSE_SUCCESS

    def _emit_recycle_succeed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Emits recycle succeed event in logs\n        :return: Nothing.\n        :rtype: None\n        '
        self.emit_event('item_discarded', formatted='Discarded {amount}x {item}.', data={'amount': str(self.amount_to_recycle), 'item': self.item_to_recycle.name})

    def _emit_recycle_failed(self):
        if False:
            print('Hello World!')
        '\n        Emits recycle failed event in logs\n        :return: Nothing.\n        :rtype: None\n        '
        self.emit_event('item_discard_fail', formatted='Failed to discard {item}', data={'item': self.item_to_recycle.name})