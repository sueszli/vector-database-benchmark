import logging
from typing import Any, List, Mapping
from airbyte_cdk.config_observation import create_connector_config_control_message
from airbyte_cdk.entrypoint import AirbyteEntrypoint
from airbyte_cdk.sources import Source
from airbyte_cdk.sources.message import InMemoryMessageRepository, MessageRepository
logger = logging.getLogger('airbyte_logger')

class MigrateStartDate:
    """
    This class stands for migrating the config at runtime,
    while providing the backward compatibility when falling back to the previous source version.

    Delete start_date field if it set to None or an empty string ("").
    """
    message_repository: MessageRepository = InMemoryMessageRepository()
    key: str = 'start_date'

    @classmethod
    def should_migrate(cls, config: Mapping[str, Any]) -> bool:
        if False:
            return 10
        '\n        Determines if a configuration requires migration.\n\n        Args:\n        - config (Mapping[str, Any]): The configuration data to check.\n\n        Returns:\n        - True: If the configuration requires migration.\n        - False: Otherwise.\n        '
        return not config.get(cls.key, 'skip_if_start_date_in_config')

    @classmethod
    def delete_from_config(cls, config: Mapping[str, Any], source: Source=None) -> Mapping[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Removes the specified key from the configuration.\n\n        Args:\n        - config (Mapping[str, Any]): The configuration from which the key should be removed.\n        - source (Source, optional): The data source. Defaults to None.\n\n        Returns:\n        - Mapping[str, Any]: The configuration after removing the key.\n        '
        config.pop(cls.key, None)
        return config

    @classmethod
    def modify_and_save(cls, config_path: str, source: Source, config: Mapping[str, Any]) -> Mapping[str, Any]:
        if False:
            return 10
        '\n        Modifies the configuration and then saves it back to the source.\n\n        Args:\n        - config_path (str): The path where the configuration is stored.\n        - source (Source): The data source.\n        - config (Mapping[str, Any]): The current configuration.\n\n        Returns:\n        - Mapping[str, Any]: The updated configuration.\n        '
        migrated_config = cls.delete_from_config(config, source)
        source.write_config(migrated_config, config_path)
        return migrated_config

    @classmethod
    def emit_control_message(cls, migrated_config: Mapping[str, Any]) -> None:
        if False:
            while True:
                i = 10
        '\n        Emits the control messages related to configuration migration.\n\n        Args:\n        - migrated_config (Mapping[str, Any]): The migrated configuration.\n        '
        cls.message_repository.emit_message(create_connector_config_control_message(migrated_config))
        for message in cls.message_repository._message_queue:
            print(message.json(exclude_unset=True))

    @classmethod
    def migrate(cls, args: List[str], source: Source) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Orchestrates the configuration migration process.\n\n        It first checks if the `--config` argument is provided, and if so,\n        determines whether migration is needed, and then performs the migration\n        if required.\n\n        Args:\n        - args (List[str]): List of command-line arguments.\n        - source (Source): The data source.\n        '
        config_path = AirbyteEntrypoint(source).extract_config(args)
        if config_path:
            config = source.read_config(config_path)
            if cls.should_migrate(config):
                cls.emit_control_message(cls.modify_and_save(config_path, source, config))