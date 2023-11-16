import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple, Union
from airbyte_cdk.models import AirbyteCatalog, AirbyteConnectionStatus, AirbyteMessage, AirbyteStateMessage, AirbyteStreamStatus, ConfiguredAirbyteCatalog, ConfiguredAirbyteStream, Status, SyncMode
from airbyte_cdk.models import Type as MessageType
from airbyte_cdk.sources.connector_state_manager import ConnectorStateManager
from airbyte_cdk.sources.message import MessageRepository
from airbyte_cdk.sources.source import Source
from airbyte_cdk.sources.streams import Stream
from airbyte_cdk.sources.streams.core import StreamData
from airbyte_cdk.sources.streams.http.http import HttpStream
from airbyte_cdk.sources.utils.record_helper import stream_data_to_airbyte_message
from airbyte_cdk.sources.utils.schema_helpers import InternalConfig, split_config
from airbyte_cdk.sources.utils.slice_logger import DebugSliceLogger, SliceLogger
from airbyte_cdk.utils.event_timing import create_timer
from airbyte_cdk.utils.stream_status_utils import as_airbyte_message as stream_status_as_airbyte_message
from airbyte_cdk.utils.traced_exception import AirbyteTracedException

class AbstractSource(Source, ABC):
    """
    Abstract base class for an Airbyte Source. Consumers should implement any abstract methods
    in this class to create an Airbyte Specification compliant Source.
    """

    @abstractmethod
    def check_connection(self, logger: logging.Logger, config: Mapping[str, Any]) -> Tuple[bool, Optional[Any]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        :param logger: source logger\n        :param config: The user-provided configuration as specified by the source\'s spec.\n          This usually contains information required to check connection e.g. tokens, secrets and keys etc.\n        :return: A tuple of (boolean, error). If boolean is true, then the connection check is successful\n          and we can connect to the underlying data source using the provided configuration.\n          Otherwise, the input config cannot be used to connect to the underlying data source,\n          and the "error" object should describe what went wrong.\n          The error object will be cast to string to display the problem to the user.\n        '

    @abstractmethod
    def streams(self, config: Mapping[str, Any]) -> List[Stream]:
        if False:
            return 10
        "\n        :param config: The user-provided configuration as specified by the source's spec.\n        Any stream construction related operation should happen here.\n        :return: A list of the streams in this source connector.\n        "
    _stream_to_instance_map: Dict[str, Stream] = {}
    _slice_logger: SliceLogger = DebugSliceLogger()

    @property
    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Source name'
        return self.__class__.__name__

    def discover(self, logger: logging.Logger, config: Mapping[str, Any]) -> AirbyteCatalog:
        if False:
            return 10
        'Implements the Discover operation from the Airbyte Specification.\n        See https://docs.airbyte.com/understanding-airbyte/airbyte-protocol/#discover.\n        '
        streams = [stream.as_airbyte_stream() for stream in self.streams(config=config)]
        return AirbyteCatalog(streams=streams)

    def check(self, logger: logging.Logger, config: Mapping[str, Any]) -> AirbyteConnectionStatus:
        if False:
            print('Hello World!')
        'Implements the Check Connection operation from the Airbyte Specification.\n        See https://docs.airbyte.com/understanding-airbyte/airbyte-protocol/#check.\n        '
        (check_succeeded, error) = self.check_connection(logger, config)
        if not check_succeeded:
            return AirbyteConnectionStatus(status=Status.FAILED, message=repr(error))
        return AirbyteConnectionStatus(status=Status.SUCCEEDED)

    def read(self, logger: logging.Logger, config: Mapping[str, Any], catalog: ConfiguredAirbyteCatalog, state: Optional[Union[List[AirbyteStateMessage], MutableMapping[str, Any]]]=None) -> Iterator[AirbyteMessage]:
        if False:
            while True:
                i = 10
        'Implements the Read operation from the Airbyte Specification. See https://docs.airbyte.com/understanding-airbyte/airbyte-protocol/.'
        logger.info(f'Starting syncing {self.name}')
        (config, internal_config) = split_config(config)
        stream_instances = {s.name: s for s in self.streams(config)}
        state_manager = ConnectorStateManager(stream_instance_map=stream_instances, state=state)
        self._stream_to_instance_map = stream_instances
        with create_timer(self.name) as timer:
            for configured_stream in catalog.streams:
                stream_instance = stream_instances.get(configured_stream.stream.name)
                if not stream_instance:
                    if not self.raise_exception_on_missing_stream:
                        continue
                    raise KeyError(f'The stream {configured_stream.stream.name} no longer exists in the configuration. Refresh the schema in replication settings and remove this stream from future sync attempts.')
                try:
                    timer.start_event(f'Syncing stream {configured_stream.stream.name}')
                    (stream_is_available, reason) = stream_instance.check_availability(logger, self)
                    if not stream_is_available:
                        logger.warning(f"Skipped syncing stream '{stream_instance.name}' because it was unavailable. {reason}")
                        continue
                    logger.info(f'Marking stream {configured_stream.stream.name} as STARTED')
                    yield stream_status_as_airbyte_message(configured_stream.stream, AirbyteStreamStatus.STARTED)
                    yield from self._read_stream(logger=logger, stream_instance=stream_instance, configured_stream=configured_stream, state_manager=state_manager, internal_config=internal_config)
                    logger.info(f'Marking stream {configured_stream.stream.name} as STOPPED')
                    yield stream_status_as_airbyte_message(configured_stream.stream, AirbyteStreamStatus.COMPLETE)
                except AirbyteTracedException as e:
                    yield stream_status_as_airbyte_message(configured_stream.stream, AirbyteStreamStatus.INCOMPLETE)
                    raise e
                except Exception as e:
                    yield from self._emit_queued_messages()
                    logger.exception(f'Encountered an exception while reading stream {configured_stream.stream.name}')
                    logger.info(f'Marking stream {configured_stream.stream.name} as STOPPED')
                    yield stream_status_as_airbyte_message(configured_stream.stream, AirbyteStreamStatus.INCOMPLETE)
                    display_message = stream_instance.get_error_display_message(e)
                    if display_message:
                        raise AirbyteTracedException.from_exception(e, message=display_message) from e
                    raise e
                finally:
                    timer.finish_event()
                    logger.info(f'Finished syncing {configured_stream.stream.name}')
                    logger.info(timer.report())
        logger.info(f'Finished syncing {self.name}')

    @property
    def raise_exception_on_missing_stream(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    @property
    def per_stream_state_enabled(self) -> bool:
        if False:
            while True:
                i = 10
        return True

    def _read_stream(self, logger: logging.Logger, stream_instance: Stream, configured_stream: ConfiguredAirbyteStream, state_manager: ConnectorStateManager, internal_config: InternalConfig) -> Iterator[AirbyteMessage]:
        if False:
            i = 10
            return i + 15
        if internal_config.page_size and isinstance(stream_instance, HttpStream):
            logger.info(f'Setting page size for {stream_instance.name} to {internal_config.page_size}')
            stream_instance.page_size = internal_config.page_size
        logger.debug(f'Syncing configured stream: {configured_stream.stream.name}', extra={'sync_mode': configured_stream.sync_mode, 'primary_key': configured_stream.primary_key, 'cursor_field': configured_stream.cursor_field})
        stream_instance.log_stream_sync_configuration()
        use_incremental = configured_stream.sync_mode == SyncMode.incremental and stream_instance.supports_incremental
        if use_incremental:
            record_iterator = self._read_incremental(logger, stream_instance, configured_stream, state_manager, internal_config)
        else:
            record_iterator = self._read_full_refresh(logger, stream_instance, configured_stream, internal_config)
        record_counter = 0
        stream_name = configured_stream.stream.name
        logger.info(f'Syncing stream: {stream_name} ')
        for record in record_iterator:
            if record.type == MessageType.RECORD:
                record_counter += 1
                if record_counter == 1:
                    logger.info(f'Marking stream {stream_name} as RUNNING')
                    yield stream_status_as_airbyte_message(configured_stream.stream, AirbyteStreamStatus.RUNNING)
            yield from self._emit_queued_messages()
            yield record
        logger.info(f'Read {record_counter} records from {stream_name} stream')

    def _read_incremental(self, logger: logging.Logger, stream_instance: Stream, configured_stream: ConfiguredAirbyteStream, state_manager: ConnectorStateManager, internal_config: InternalConfig) -> Iterator[AirbyteMessage]:
        if False:
            i = 10
            return i + 15
        'Read stream using incremental algorithm\n\n        :param logger:\n        :param stream_instance:\n        :param configured_stream:\n        :param state_manager:\n        :param internal_config:\n        :return:\n        '
        stream_name = configured_stream.stream.name
        stream_state = state_manager.get_stream_state(stream_name, stream_instance.namespace)
        if stream_state and 'state' in dir(stream_instance):
            stream_instance.state = stream_state
            logger.info(f'Setting state of {self.name} stream to {stream_state}')
        for record_data_or_message in stream_instance.read_incremental(configured_stream.cursor_field, logger, self._slice_logger, stream_state, state_manager, self.per_stream_state_enabled, internal_config):
            yield self._get_message(record_data_or_message, stream_instance)

    def _emit_queued_messages(self) -> Iterable[AirbyteMessage]:
        if False:
            while True:
                i = 10
        if self.message_repository:
            yield from self.message_repository.consume_queue()
        return

    def _read_full_refresh(self, logger: logging.Logger, stream_instance: Stream, configured_stream: ConfiguredAirbyteStream, internal_config: InternalConfig) -> Iterator[AirbyteMessage]:
        if False:
            for i in range(10):
                print('nop')
        total_records_counter = 0
        for record_data_or_message in stream_instance.read_full_refresh(configured_stream.cursor_field, logger, self._slice_logger):
            message = self._get_message(record_data_or_message, stream_instance)
            yield message
            if message.type == MessageType.RECORD:
                total_records_counter += 1
                if internal_config.is_limit_reached(total_records_counter):
                    return

    def _get_message(self, record_data_or_message: Union[StreamData, AirbyteMessage], stream: Stream) -> AirbyteMessage:
        if False:
            i = 10
            return i + 15
        '\n        Converts the input to an AirbyteMessage if it is a StreamData. Returns the input as is if it is already an AirbyteMessage\n        '
        if isinstance(record_data_or_message, AirbyteMessage):
            return record_data_or_message
        else:
            return stream_data_to_airbyte_message(stream.name, record_data_or_message, stream.transformer, stream.get_json_schema())

    @property
    def message_repository(self) -> Union[None, MessageRepository]:
        if False:
            return 10
        return None