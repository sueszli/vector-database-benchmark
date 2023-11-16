""" Bitpanda exchange subclass """
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from freqtrade.exchange import Exchange
logger = logging.getLogger(__name__)

class Bitpanda(Exchange):
    """
    Bitpanda exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    def get_trades_for_order(self, order_id: str, pair: str, since: datetime, params: Optional[Dict]=None) -> List:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fetch Orders using the "fetch_my_trades" endpoint and filter them by order-id.\n        The "since" argument passed in is coming from the database and is in UTC,\n        as timezone-native datetime object.\n        From the python documentation:\n            > Naive datetime instances are assumed to represent local time\n        Therefore, calling "since.timestamp()" will get the UTC timestamp, after applying the\n        transformation from local timezone to UTC.\n        This works for timezones UTC+ since then the result will contain trades from a few hours\n        instead of from the last 5 seconds, however fails for UTC- timezones,\n        since we\'re then asking for trades with a "since" argument in the future.\n\n        :param order_id order_id: Order-id as given when creating the order\n        :param pair: Pair the order is for\n        :param since: datetime object of the order creation time. Assumes object is in UTC.\n        '
        params = {'to': int(datetime.now(timezone.utc).timestamp() * 1000)}
        return super().get_trades_for_order(order_id, pair, since, params)