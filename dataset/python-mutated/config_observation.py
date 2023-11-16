from __future__ import annotations
import time
from typing import Any, List, MutableMapping
from airbyte_cdk.models import AirbyteControlConnectorConfigMessage, AirbyteControlMessage, AirbyteMessage, OrchestratorType, Type

class ObservedDict(dict):

    def __init__(self, non_observed_mapping: MutableMapping, observer: ConfigObserver, update_on_unchanged_value=True) -> None:
        if False:
            print('Hello World!')
        non_observed_mapping = non_observed_mapping.copy()
        self.observer = observer
        self.update_on_unchanged_value = update_on_unchanged_value
        for (item, value) in non_observed_mapping.items():
            if isinstance(value, MutableMapping):
                non_observed_mapping[item] = ObservedDict(value, observer)
            if isinstance(value, List):
                for (i, sub_value) in enumerate(value):
                    if isinstance(sub_value, MutableMapping):
                        value[i] = ObservedDict(sub_value, observer)
        super().__init__(non_observed_mapping)

    def __setitem__(self, item: Any, value: Any):
        if False:
            while True:
                i = 10
        'Override dict.__setitem__ by:\n        1. Observing the new value if it is a dict\n        2. Call observer update if the new value is different from the previous one\n        '
        previous_value = self.get(item)
        if isinstance(value, MutableMapping):
            value = ObservedDict(value, self.observer)
        if isinstance(value, List):
            for (i, sub_value) in enumerate(value):
                if isinstance(sub_value, MutableMapping):
                    value[i] = ObservedDict(sub_value, self.observer)
        super(ObservedDict, self).__setitem__(item, value)
        if self.update_on_unchanged_value or value != previous_value:
            self.observer.update()

class ConfigObserver:
    """This class is made to track mutations on ObservedDict config.
    When update is called a CONNECTOR_CONFIG control message is emitted on stdout.
    """

    def set_config(self, config: ObservedDict) -> None:
        if False:
            return 10
        self.config = config

    def update(self) -> None:
        if False:
            while True:
                i = 10
        emit_configuration_as_airbyte_control_message(self.config)

def observe_connector_config(non_observed_connector_config: MutableMapping[str, Any]):
    if False:
        return 10
    if isinstance(non_observed_connector_config, ObservedDict):
        raise ValueError('This connector configuration is already observed')
    connector_config_observer = ConfigObserver()
    observed_connector_config = ObservedDict(non_observed_connector_config, connector_config_observer)
    connector_config_observer.set_config(observed_connector_config)
    return observed_connector_config

def emit_configuration_as_airbyte_control_message(config: MutableMapping):
    if False:
        i = 10
        return i + 15
    '\n    WARNING: deprecated - emit_configuration_as_airbyte_control_message is being deprecated in favor of the MessageRepository mechanism.\n    See the airbyte_cdk.sources.message package\n    '
    airbyte_message = create_connector_config_control_message(config)
    print(airbyte_message.json(exclude_unset=True))

def create_connector_config_control_message(config):
    if False:
        print('Hello World!')
    control_message = AirbyteControlMessage(type=OrchestratorType.CONNECTOR_CONFIG, emitted_at=time.time() * 1000, connectorConfig=AirbyteControlConnectorConfigMessage(config=config))
    return AirbyteMessage(type=Type.CONTROL, control=control_message)