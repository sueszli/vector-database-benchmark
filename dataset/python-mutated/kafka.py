from __future__ import annotations
from typing import Any, Callable, Sequence
from airflow.models import BaseOperator
from airflow.providers.apache.kafka.triggers.await_message import AwaitMessageTrigger
VALID_COMMIT_CADENCE = {'never', 'end_of_batch', 'end_of_operator'}

class AwaitMessageSensor(BaseOperator):
    """An Airflow sensor that defers until a specific message is published to Kafka.

    The sensor creates a consumer that reads the Kafka log until it encounters a positive event.

    The behavior of the consumer for this trigger is as follows:
    - poll the Kafka topics for a message
    - if no message returned, sleep
    - process the message with provided callable and commit the message offset
    - if callable returns any data, raise a TriggerEvent with the return data
    - else continue to next message
    - return event (as default xcom or specific xcom key)


    :param kafka_config_id: The connection object to use, defaults to "kafka_default"
    :param topics: Topics (or topic regex) to use for reading from
    :param apply_function: The function to apply to messages to determine if an event occurred. As a dot
        notation string.
    :param apply_function_args: Arguments to be applied to the processing function,
        defaults to None
    :param apply_function_kwargs: Key word arguments to be applied to the processing function,
        defaults to None
    :param poll_timeout: How long the kafka consumer should wait for a message to arrive from the kafka
        cluster,defaults to 1
    :param poll_interval: How long the kafka consumer should sleep after reaching the end of the Kafka log,
        defaults to 5
    :param xcom_push_key: the name of a key to push the returned message to, defaults to None


    """
    BLUE = '#ffefeb'
    ui_color = BLUE
    template_fields = ('topics', 'apply_function', 'apply_function_args', 'apply_function_kwargs', 'kafka_config_id')

    def __init__(self, topics: Sequence[str], apply_function: str, kafka_config_id: str='kafka_default', apply_function_args: Sequence[Any] | None=None, apply_function_kwargs: dict[Any, Any] | None=None, poll_timeout: float=1, poll_interval: float=5, xcom_push_key=None, **kwargs: Any) -> None:
        if False:
            return 10
        super().__init__(**kwargs)
        self.topics = topics
        self.apply_function = apply_function
        self.apply_function_args = apply_function_args
        self.apply_function_kwargs = apply_function_kwargs
        self.kafka_config_id = kafka_config_id
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.xcom_push_key = xcom_push_key

    def execute(self, context) -> Any:
        if False:
            print('Hello World!')
        self.defer(trigger=AwaitMessageTrigger(topics=self.topics, apply_function=self.apply_function, apply_function_args=self.apply_function_args, apply_function_kwargs=self.apply_function_kwargs, kafka_config_id=self.kafka_config_id, poll_timeout=self.poll_timeout, poll_interval=self.poll_interval), method_name='execute_complete')

    def execute_complete(self, context, event=None):
        if False:
            for i in range(10):
                print('nop')
        if self.xcom_push_key:
            self.xcom_push(context, key=self.xcom_push_key, value=event)
        return event

class AwaitMessageTriggerFunctionSensor(BaseOperator):
    """
    Defer until a specific message is published to Kafka, trigger a registered function, then resume waiting.

    The behavior of the consumer for this trigger is as follows:
    - poll the Kafka topics for a message
    - if no message returned, sleep
    - process the message with provided callable and commit the message offset
    - if callable returns any data, raise a TriggerEvent with the return data
    - else continue to next message
    - return event (as default xcom or specific xcom key)


    :param kafka_config_id: The connection object to use, defaults to "kafka_default"
    :param topics: Topics (or topic regex) to use for reading from
    :param apply_function: The function to apply to messages to determine if an event occurred. As a dot
        notation string.
    :param event_triggered_function: The callable to trigger once the apply_function encounters a
        positive event.
    :param apply_function_args: Arguments to be applied to the processing function, defaults to None
    :param apply_function_kwargs: Key word arguments to be applied to the processing function,
        defaults to None
    :param poll_timeout: How long the kafka consumer should wait for a message to arrive from the kafka
        cluster, defaults to 1
    :param poll_interval: How long the kafka consumer should sleep after reaching the end of the Kafka log,
        defaults to 5


    """
    BLUE = '#ffefeb'
    ui_color = BLUE
    template_fields = ('topics', 'apply_function', 'apply_function_args', 'apply_function_kwargs', 'kafka_config_id')

    def __init__(self, topics: Sequence[str], apply_function: str, event_triggered_function: Callable, kafka_config_id: str='kafka_default', apply_function_args: Sequence[Any] | None=None, apply_function_kwargs: dict[Any, Any] | None=None, poll_timeout: float=1, poll_interval: float=5, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.topics = topics
        self.apply_function = apply_function
        self.apply_function_args = apply_function_args
        self.apply_function_kwargs = apply_function_kwargs
        self.kafka_config_id = kafka_config_id
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval
        self.event_triggered_function = event_triggered_function
        if not callable(self.event_triggered_function):
            raise TypeError(f'parameter event_triggered_function is expected to be of type callable,got {type(event_triggered_function)}')

    def execute(self, context, event=None) -> Any:
        if False:
            for i in range(10):
                print('nop')
        self.defer(trigger=AwaitMessageTrigger(topics=self.topics, apply_function=self.apply_function, apply_function_args=self.apply_function_args, apply_function_kwargs=self.apply_function_kwargs, kafka_config_id=self.kafka_config_id, poll_timeout=self.poll_timeout, poll_interval=self.poll_interval), method_name='execute_complete')
        return event

    def execute_complete(self, context, event=None):
        if False:
            return 10
        self.event_triggered_function(event, **context)
        self.defer(trigger=AwaitMessageTrigger(topics=self.topics, apply_function=self.apply_function, apply_function_args=self.apply_function_args, apply_function_kwargs=self.apply_function_kwargs, kafka_config_id=self.kafka_config_id, poll_timeout=self.poll_timeout, poll_interval=self.poll_interval), method_name='execute_complete')