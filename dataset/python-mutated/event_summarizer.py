import time
from threading import RLock
from typing import Any, Callable, Dict, List

class EventSummarizer:
    """Utility that aggregates related log messages to reduce log spam."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.events_by_key: Dict[str, int] = {}
        self.messages_to_send: List[str] = []
        self.throttled_messages: Dict[str, float] = {}
        self.lock = RLock()

    def add(self, template: str, *, quantity: Any, aggregate: Callable[[Any, Any], Any]) -> None:
        if False:
            i = 10
            return i + 15
        'Add a log message, which will be combined by template.\n\n        Args:\n            template: Format string with one placeholder for quantity.\n            quantity: Quantity to aggregate.\n            aggregate: Aggregation function used to combine the\n                quantities. The result is inserted into the template to\n                produce the final log message.\n        '
        with self.lock:
            if not template.endswith('.'):
                template += '.'
            if template in self.events_by_key:
                self.events_by_key[template] = aggregate(self.events_by_key[template], quantity)
            else:
                self.events_by_key[template] = quantity

    def add_once_per_interval(self, message: str, key: str, interval_s: int):
        if False:
            for i in range(10):
                print('nop')
        'Add a log message, which is throttled once per interval by a key.\n\n        Args:\n            message: The message to log.\n            key: The key to use to deduplicate the message.\n            interval_s: Throttling interval in seconds.\n        '
        with self.lock:
            if key not in self.throttled_messages:
                self.throttled_messages[key] = time.time() + interval_s
                self.messages_to_send.append(message)

    def summary(self) -> List[str]:
        if False:
            print('Hello World!')
        'Generate the aggregated log summary of all added events.'
        with self.lock:
            out = []
            for (template, quantity) in self.events_by_key.items():
                out.append(template.format(quantity))
            out.extend(self.messages_to_send)
        return out

    def clear(self) -> None:
        if False:
            return 10
        'Clear the events added.'
        with self.lock:
            self.events_by_key.clear()
            self.messages_to_send.clear()
            for (k, t) in list(self.throttled_messages.items()):
                if time.time() > t:
                    del self.throttled_messages[k]