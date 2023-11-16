import asyncio
import copy
import logging
import os
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Dict, List, Union
import yaml
from jinja2 import Template
from mage_ai.data_preparation.executors.pipeline_executor import PipelineExecutor
from mage_ai.data_preparation.logging.logger import DictLogger
from mage_ai.data_preparation.models.constants import BlockType
from mage_ai.data_preparation.models.pipeline import Pipeline
from mage_ai.data_preparation.shared.stream import StreamToLogger
from mage_ai.data_preparation.shared.utils import get_template_vars
from mage_ai.shared.hash import merge_dict

class StreamingPipelineExecutor(PipelineExecutor):

    def __init__(self, pipeline: Pipeline, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(pipeline, **kwargs)
        self.parse_and_validate_blocks()

    def parse_and_validate_blocks(self):
        if False:
            while True:
                i = 10
        '\n        Find the first valid streaming pipeline is in the structure:\n        source -> transformer1 -> sink2\n               -> transformer2 -> sink2\n               -> transformer3 -> sink3\n        '
        blocks = self.pipeline.blocks_by_uuid.values()
        source_blocks = []
        sink_blocks = []
        transformer_blocks = []
        for b in blocks:
            if b.type == BlockType.DATA_LOADER:
                if len(b.upstream_blocks or []) > 0:
                    raise Exception(f"Data loader {b.uuid} can't have upstream blocks.")
                if len(b.downstream_blocks or []) < 1:
                    raise Exception(f'Data loader {b.uuid} must have at least one transformer or data exporter as the downstream block.')
                source_blocks.append(b)
            if b.type == BlockType.DATA_EXPORTER:
                if len(b.downstream_blocks or []) > 0:
                    raise Exception(f"Data expoter {b.uuid} can't have downstream blocks.")
                if len(b.upstream_blocks or []) != 1:
                    raise Exception(f'Data exporter {b.uuid} must have a transformer or data loader as the upstream block.')
                sink_blocks.append(b)
            if b.type == BlockType.TRANSFORMER:
                if len(b.upstream_blocks or []) != 1:
                    raise Exception(f'Transformer {b.uuid} should (only) have one upstream block.')
                transformer_blocks.append(b)
        if len(source_blocks) != 1:
            raise Exception('Please provide (only) one data loader block as the source.')
        self.source_block = source_blocks[0]
        self.sink_blocks = sink_blocks

    def execute(self, build_block_output_stdout: Callable[..., object]=None, global_vars: Dict=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        tags = self.build_tags(**kwargs)
        if build_block_output_stdout:
            stdout_logger = logging.getLogger('streaming_pipeline_executor')
            self.logger = DictLogger(stdout_logger)
            stdout = build_block_output_stdout(self.pipeline.uuid)
        else:
            self.logger = DictLogger(self.logger_manager.logger, logging_tags=tags)
            stdout = StreamToLogger(self.logger, logging_tags=tags)
        try:
            with redirect_stdout(stdout):
                with redirect_stderr(stdout):
                    self.__execute_in_python(build_block_output_stdout=build_block_output_stdout, global_vars=global_vars)
        except Exception as e:
            if not build_block_output_stdout:
                self.logger.exception(f'Failed to execute streaming pipeline {self.pipeline.uuid}', **merge_dict(dict(error=e), tags))
            raise e

    def __execute_in_python(self, build_block_output_stdout: Callable[..., object]=None, global_vars: Dict=None):
        if False:
            return 10
        from mage_ai.streaming.sinks.sink_factory import SinkFactory
        from mage_ai.streaming.sources.base import SourceConsumeMethod
        from mage_ai.streaming.sources.source_factory import SourceFactory
        if global_vars is None:
            global_vars = dict()
        source_config = self.__interpolate_vars(self.source_block.content, global_vars=global_vars)
        source = SourceFactory.get_source(source_config, checkpoint_path=os.path.join(self.pipeline.pipeline_variables_dir, 'streaming_checkpoint'))
        sinks_by_uuid = dict()
        for sink_block in self.sink_blocks:
            sinks_by_uuid[sink_block.uuid] = SinkFactory.get_sink(self.__interpolate_vars(sink_block.content, global_vars=global_vars), buffer_path=os.path.join(self.pipeline.pipeline_variables_dir, 'buffer'))

        def __deepcopy(data):
            if False:
                print('Hello World!')
            if data is None:
                return data
            if type(data) is list:
                data_copy = []
                for item in data:
                    data_copy.append(__deepcopy(item))
                return data_copy
            try:
                return copy.deepcopy(data)
            except Exception:
                return copy.copy(data)

        def handle_batch_events_recursively(curr_block, outputs_by_block: Dict, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            curr_block_output = outputs_by_block[curr_block.uuid]
            for downstream_block in curr_block.downstream_blocks:
                if downstream_block.type == BlockType.TRANSFORMER:
                    execute_block_kwargs = dict(global_vars=kwargs, input_args=[__deepcopy(curr_block_output)], logger=self.logger)
                    if build_block_output_stdout:
                        execute_block_kwargs['build_block_output_stdout'] = build_block_output_stdout
                    outputs_by_block[downstream_block.uuid] = downstream_block.execute_block(**execute_block_kwargs)['output']
                elif downstream_block.type == BlockType.DATA_EXPORTER:
                    sinks_by_uuid[downstream_block.uuid].batch_write(__deepcopy(curr_block_output))
                if downstream_block.downstream_blocks:
                    handle_batch_events_recursively(downstream_block, outputs_by_block, **kwargs)

        def handle_batch_events(messages: List[Union[Dict, str]], **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            outputs_by_block = dict()
            outputs_by_block[self.source_block.uuid] = messages
            handle_batch_events_recursively(self.source_block, outputs_by_block, **merge_dict(global_vars, kwargs))

        async def handle_event_async(message, **kwargs):
            outputs_by_block = dict()
            outputs_by_block[self.source_block.uuid] = [message]
            handle_batch_events_recursively(self.source_block, outputs_by_block, **merge_dict(global_vars, kwargs))
        try:
            if source.consume_method == SourceConsumeMethod.BATCH_READ:
                source.batch_read(handler=handle_batch_events)
            elif source.consume_method == SourceConsumeMethod.READ_ASYNC:
                loop = asyncio.get_event_loop()
                if loop is not None:
                    loop.run_until_complete(source.read_async(handler=handle_event_async))
                else:
                    asyncio.run(source.read_async(handler=handle_event_async))
        finally:
            source.destroy()
            for sink in sinks_by_uuid.values():
                sink.destroy()

    def __execute_in_flink(self):
        if False:
            return 10
        '\n        TODO: Implement this method\n        '
        pass

    def __interpolate_vars(self, content: str, global_vars: Dict=None):
        if False:
            return 10
        if global_vars is None:
            global_vars = dict()
        config_file = Template(content).render(variables=lambda x: global_vars.get(x) if global_vars else None, **get_template_vars())
        return yaml.safe_load(config_file)