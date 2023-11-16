import json
import traceback
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Union
import pytz
import requests
from dateutil.relativedelta import relativedelta
from mage_ai.data_integrations.utils.scheduler import build_block_run_metadata, get_extra_variables
from mage_ai.data_preparation.logging.logger import DictLogger
from mage_ai.data_preparation.logging.logger_manager_factory import LoggerManagerFactory
from mage_ai.data_preparation.models.block.data_integration.utils import destination_module_file_path, get_selected_streams, get_streams_from_catalog, get_streams_from_output_directory, source_module_file_path
from mage_ai.data_preparation.models.block.utils import create_block_runs_from_dynamic_block
from mage_ai.data_preparation.models.block.utils import dynamic_block_uuid as dynamic_block_uuid_func
from mage_ai.data_preparation.models.block.utils import dynamic_block_values_and_metadata, is_dynamic_block, is_dynamic_block_child, should_reduce_output
from mage_ai.data_preparation.models.constants import BlockLanguage, BlockType, PipelineType
from mage_ai.data_preparation.models.project import Project
from mage_ai.data_preparation.models.project.constants import FeatureUUID
from mage_ai.data_preparation.models.triggers import ScheduleInterval, ScheduleType
from mage_ai.data_preparation.shared.retry import RetryConfig
from mage_ai.orchestration.db.models.schedules import BlockRun, PipelineRun
from mage_ai.shared.hash import merge_dict
from mage_ai.shared.utils import clean_name

class BlockExecutor:
    """
    Executor for a block in a pipeline.
    """
    RETRYABLE = True

    def __init__(self, pipeline, block_uuid, execution_partition=None):
        if False:
            while True:
                i = 10
        '\n        Initialize the BlockExecutor.\n\n        Args:\n            pipeline: The pipeline object.\n            block_uuid: The UUID of the block.\n            execution_partition: The execution partition of the block.\n        '
        self.pipeline = pipeline
        self.block_uuid = block_uuid
        self.block = self.pipeline.get_block(self.block_uuid, check_template=True)
        self.execution_partition = execution_partition
        self.logger_manager = LoggerManagerFactory.get_logger_manager(pipeline_uuid=self.pipeline.uuid, block_uuid=clean_name(self.block_uuid), partition=self.execution_partition, repo_config=self.pipeline.repo_config)
        self.logger = DictLogger(self.logger_manager.logger)
        self.project = Project(self.pipeline.repo_config)

    def execute(self, analyze_outputs: bool=False, block_run_id: int=None, callback_url: Union[str, None]=None, global_vars: Union[Dict, None]=None, input_from_output: Union[Dict, None]=None, on_complete: Union[Callable[[str], None], None]=None, on_failure: Union[Callable[[str, Dict], None], None]=None, on_start: Union[Callable[[str], None], None]=None, pipeline_run_id: int=None, retry_config: Dict=None, runtime_arguments: Union[Dict, None]=None, template_runtime_configuration: Union[Dict, None]=None, update_status: bool=False, verify_output: bool=True, block_run_dicts: List[str]=None, **kwargs) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute the block.\n\n        Args:\n            analyze_outputs: Whether to analyze the outputs of the block.\n            block_run_id: The ID of the block run.\n            callback_url: The URL for the callback.\n            global_vars: Global variables for the block execution.\n            input_from_output: Input from the output of a previous block.\n            on_complete: Callback function called when the block execution is complete.\n            on_failure: Callback function called when the block execution fails.\n            on_start: Callback function called when the block execution starts.\n            pipeline_run_id: The ID of the pipeline run.\n            retry_config: Configuration for retrying the block execution.\n            runtime_arguments: Runtime arguments for the block execution.\n            template_runtime_configuration: Template runtime configuration for the block execution.\n            update_status: Whether to update the status of the block in pipeline metadata.yaml file.\n            verify_output: Whether to verify the output of the block.\n            **kwargs: Additional keyword arguments.\n\n        Returns:\n            The result of the block execution.\n        '
        if template_runtime_configuration:
            self.block.template_runtime_configuration = template_runtime_configuration
        try:
            result = dict()
            tags = self.build_tags(block_run_id=block_run_id, pipeline_run_id=pipeline_run_id, **kwargs)
            self.logger.logging_tags = tags
            if on_start is not None:
                on_start(self.block_uuid)
            block_run = BlockRun.query.get(block_run_id) if block_run_id else None
            pipeline_run = PipelineRun.query.get(pipeline_run_id) if pipeline_run_id else None
            is_original_block = self.block.uuid == self.block_uuid
            is_data_integration_child = False
            is_data_integration_controller = False
            data_integration_metadata = None
            run_in_parallel = False
            upstream_block_uuids = None
            is_data_integration = self.block.is_data_integration()
            if is_data_integration and pipeline_run:
                if not runtime_arguments:
                    runtime_arguments = {}
                pipeline_schedule = pipeline_run.pipeline_schedule
                schedule_interval = pipeline_schedule.schedule_interval
                if ScheduleType.API == pipeline_schedule.schedule_type:
                    execution_date = datetime.utcnow()
                else:
                    execution_date = pipeline_schedule.current_execution_date()
                end_date = None
                start_date = None
                date_diff = None
                variables = pipeline_run.get_variables(extra_variables=get_extra_variables(self.pipeline))
                if variables:
                    if global_vars:
                        global_vars.update(variables)
                    else:
                        global_vars = variables
                if ScheduleInterval.ONCE == schedule_interval:
                    end_date = variables.get('_end_date')
                    start_date = variables.get('_start_date')
                elif ScheduleInterval.HOURLY == schedule_interval:
                    date_diff = timedelta(hours=1)
                elif ScheduleInterval.DAILY == schedule_interval:
                    date_diff = timedelta(days=1)
                elif ScheduleInterval.WEEKLY == schedule_interval:
                    date_diff = timedelta(weeks=1)
                elif ScheduleInterval.MONTHLY == schedule_interval:
                    date_diff = relativedelta(months=1)
                if date_diff is not None:
                    end_date = execution_date.isoformat()
                    start_date = (execution_date - date_diff).isoformat()
                runtime_arguments.update(dict(_end_date=end_date, _execution_date=execution_date.isoformat(), _execution_partition=pipeline_run.execution_partition, _start_date=start_date))
            if block_run and block_run.metrics and is_data_integration:
                data_integration_metadata = block_run.metrics
                run_in_parallel = int(data_integration_metadata.get('run_in_parallel') or 0) == 1
                upstream_block_uuids = data_integration_metadata.get('upstream_block_uuids')
                is_data_integration_child = data_integration_metadata.get('child', False)
                is_data_integration_controller = data_integration_metadata.get('controller', False)
                stream = data_integration_metadata.get('stream')
                if stream and is_data_integration_child and (not is_data_integration_controller):
                    if not self.block.template_runtime_configuration:
                        self.block.template_runtime_configuration = {}
                    self.block.template_runtime_configuration['selected_streams'] = [stream]
                    for key in ['index', 'parent_stream']:
                        if key in data_integration_metadata:
                            self.block.template_runtime_configuration[key] = data_integration_metadata.get(key)
            if not is_data_integration_controller or is_data_integration_child:
                self.logger.info(f'Start executing block with {self.__class__.__name__}.', **tags)
            if block_run:
                block_run_data = block_run.metrics or {}
                dynamic_block_index = block_run_data.get('dynamic_block_index', None)
                dynamic_upstream_block_uuids = block_run_data.get('dynamic_upstream_block_uuids', None)
            else:
                dynamic_block_index = None
                dynamic_upstream_block_uuids = None
            if dynamic_upstream_block_uuids:
                dynamic_upstream_block_uuids_reduce = []
                dynamic_upstream_block_uuids_no_reduce = []
                for upstream_block_uuid in dynamic_upstream_block_uuids:
                    upstream_block = self.pipeline.get_block(upstream_block_uuid)
                    if not should_reduce_output(upstream_block):
                        dynamic_upstream_block_uuids_no_reduce.append(upstream_block_uuid)
                        continue
                    parts = upstream_block_uuid.split(':')
                    suffix = None
                    if len(parts) >= 3:
                        suffix = ':'.join(parts[2:])
                    for block_grandparent in list(filter(lambda x: is_dynamic_block(x), upstream_block.upstream_blocks)):
                        block_grandparent_uuid = block_grandparent.uuid
                        if suffix and is_dynamic_block_child(block_grandparent):
                            block_grandparent_uuid = f'{block_grandparent_uuid}:{suffix}'
                        (values, block_metadata) = dynamic_block_values_and_metadata(block_grandparent, self.execution_partition, block_grandparent_uuid)
                        for (idx, _) in enumerate(values):
                            if idx < len(block_metadata):
                                metadata = block_metadata[idx].copy()
                            else:
                                metadata = {}
                            dynamic_upstream_block_uuids_reduce.append(dynamic_block_uuid_func(upstream_block.uuid, metadata, idx, upstream_block_uuid=block_grandparent_uuid))
                dynamic_upstream_block_uuids = dynamic_upstream_block_uuids_reduce + dynamic_upstream_block_uuids_no_reduce
            conditional_result = self._execute_conditional(dynamic_block_index=dynamic_block_index, dynamic_upstream_block_uuids=dynamic_upstream_block_uuids, global_vars=global_vars, logging_tags=tags, pipeline_run=pipeline_run)
            if not conditional_result:
                self.logger.info(f'Conditional block(s) returned false for {self.block.uuid}. This block run and downstream blocks will be set as CONDITION_FAILED.', **merge_dict(tags, dict(block_type=self.block.type, block_uuid=self.block.uuid)))
                if is_data_integration:

                    def __update_condition_failed(block_run_id_init: int, block_run_block_uuid_init: str, block_init, block_run_dicts=block_run_dicts, tags=tags):
                        if False:
                            return 10
                        self.__update_block_run_status(BlockRun.BlockRunStatus.CONDITION_FAILED, block_run_id=block_run_id_init, tags=tags)
                        downstream_block_uuids = block_init.downstream_block_uuids
                        for block_run_dict in block_run_dicts:
                            block_run_block_uuid = block_run_dict.get('block_uuid')
                            block_run_id2 = block_run_dict.get('id')
                            if block_run_block_uuid_init == block_run_block_uuid:
                                continue
                            block = self.pipeline.get_block(block_run_block_uuid)
                            if block.uuid in downstream_block_uuids:
                                __update_condition_failed(block_run_id2, block_run_block_uuid, block)
                            metrics = block_run_dict.get('metrics')
                            original_block_uuid = metrics.get('original_block_uuid')
                            if block_init == original_block_uuid or block_init.uuid == block.uuid:
                                self.__update_block_run_status(BlockRun.BlockRunStatus.CONDITION_FAILED, block_run_id=block_run_id2, tags=tags)
                    __update_condition_failed(block_run_id, self.block_uuid, self.block)
                else:
                    self.__update_block_run_status(BlockRun.BlockRunStatus.CONDITION_FAILED, block_run_id=block_run_id, callback_url=callback_url, tags=tags)
                return dict(output=[])
            should_execute = True
            should_finish = False
            if data_integration_metadata and is_original_block:
                controller_block_uuid = data_integration_metadata.get('controller_block_uuid')
                if on_complete and controller_block_uuid:
                    on_complete(controller_block_uuid)
                    self.logger.info(f'All child block runs completed, updating controller block run for block {controller_block_uuid} to complete.', **merge_dict(tags, dict(block_uuid=self.block.uuid, controller_block_uuid=controller_block_uuid)))
                should_execute = False
            elif is_data_integration_controller and is_data_integration_child and (not run_in_parallel):
                children = []
                status_count = {}
                block_run_dicts_mapping = {}
                for block_run_dict in block_run_dicts:
                    block_run_dicts_mapping[block_run_dict['block_uuid']] = block_run_dict
                    metrics = block_run_dict.get('metrics') or {}
                    controller_block_uuid = metrics.get('controller_block_uuid')
                    if controller_block_uuid == self.block_uuid:
                        children.append(block_run_dict)
                    status = block_run_dict.get('status')
                    if status not in status_count:
                        status_count[status] = 0
                    status_count[status] += 1
                children_length = len(children)
                should_finish = children_length >= 1 and status_count.get(BlockRun.BlockRunStatus.COMPLETED.value, 0) >= children_length
                if upstream_block_uuids:
                    statuses_completed = []
                    for up_block_uuid in upstream_block_uuids:
                        block_run_dict = block_run_dicts_mapping.get(up_block_uuid)
                        if block_run_dict:
                            statuses_completed.append(BlockRun.BlockRunStatus.COMPLETED.value == block_run_dict.get('status'))
                        else:
                            statuses_completed.append(False)
                    should_execute = all(statuses_completed)
                else:
                    should_execute = True
            elif is_data_integration_child and (not is_data_integration_controller):
                index = int(data_integration_metadata.get('index') or 0)
                if index >= 1:
                    controller_block_uuid = data_integration_metadata.get('controller_block_uuid')
                    block_run_dict_previous = None
                    for block_run_dict in block_run_dicts:
                        if block_run_dict_previous:
                            break
                        metrics = block_run_dict.get('metrics')
                        if not metrics:
                            continue
                        if controller_block_uuid == metrics.get('controller_block_uuid') and index - 1 == int(metrics.get('index') or 0):
                            block_run_dict_previous = block_run_dict
                    if block_run_dict_previous:
                        should_execute = BlockRun.BlockRunStatus.COMPLETED.value == block_run_dict_previous.get('status')
                        if not should_execute:
                            stream = data_integration_metadata.get('stream')
                            self.logger.info(f'Block run ({block_run_id}) {self.block_uuid} for stream {stream} and batch {index} is waiting for batch {index - 1} to complete.', **merge_dict(tags, dict(batch=index - 1, block_uuid=self.block.uuid, controller_block_uuid=controller_block_uuid, index=index)))
                            return
                else:
                    should_execute = True
            if should_execute:
                try:
                    from mage_ai.shared.retry import retry
                    if retry_config is None:
                        if self.RETRYABLE:
                            retry_config = merge_dict(self.pipeline.repo_config.retry_config or dict(), self.block.retry_config or dict())
                        else:
                            retry_config = dict()
                    if type(retry_config) is not RetryConfig:
                        retry_config = RetryConfig.load(config=retry_config)

                    @retry(retries=retry_config.retries if self.RETRYABLE else 0, delay=retry_config.delay, max_delay=retry_config.max_delay, exponential_backoff=retry_config.exponential_backoff, logger=self.logger, logging_tags=tags)
                    def __execute_with_retry():
                        if False:
                            return 10
                        return self._execute(analyze_outputs=analyze_outputs, block_run_id=block_run_id, callback_url=callback_url, global_vars=global_vars, update_status=update_status, input_from_output=input_from_output, logging_tags=tags, pipeline_run_id=pipeline_run_id, verify_output=verify_output, runtime_arguments=runtime_arguments, template_runtime_configuration=template_runtime_configuration, dynamic_block_index=dynamic_block_index, dynamic_block_uuid=None if dynamic_block_index is None else block_run.block_uuid, dynamic_upstream_block_uuids=dynamic_upstream_block_uuids, data_integration_metadata=data_integration_metadata, pipeline_run=pipeline_run, block_run_dicts=block_run_dicts, **kwargs)
                    result = __execute_with_retry()
                except Exception as error:
                    self.logger.exception(f'Failed to execute block {self.block.uuid}', **merge_dict(tags, dict(error=error)))
                    if on_failure is not None:
                        on_failure(self.block_uuid, error=dict(error=error, errors=traceback.format_stack(), message=traceback.format_exc()))
                    else:
                        self.__update_block_run_status(BlockRun.BlockRunStatus.FAILED, block_run_id=block_run_id, callback_url=callback_url, tags=tags)
                    self._execute_callback('on_failure', callback_kwargs=dict(__error=error), dynamic_block_index=dynamic_block_index, dynamic_upstream_block_uuids=dynamic_upstream_block_uuids, global_vars=global_vars, logging_tags=tags, pipeline_run=pipeline_run)
                    raise error
            if not should_finish:
                should_finish = not is_data_integration_controller or (is_data_integration_child and run_in_parallel)
            if not should_finish:
                should_finish = is_data_integration_controller and is_data_integration_child and self.block.is_destination()
            if should_finish:
                self.logger.info(f'Finish executing block with {self.__class__.__name__}.', **tags)
                if on_complete is not None:
                    on_complete(self.block_uuid)
                else:
                    self.__update_block_run_status(BlockRun.BlockRunStatus.COMPLETED, block_run_id=block_run_id, callback_url=callback_url, pipeline_run=pipeline_run, tags=tags)
            if not data_integration_metadata or is_original_block:
                self._execute_callback('on_success', dynamic_block_index=dynamic_block_index, dynamic_upstream_block_uuids=dynamic_upstream_block_uuids, global_vars=global_vars, logging_tags=tags, pipeline_run=pipeline_run)
            return result
        finally:
            self.logger_manager.output_logs_to_destination()

    def _execute(self, analyze_outputs: bool=False, block_run_id: int=None, callback_url: Union[str, None]=None, global_vars: Union[Dict, None]=None, update_status: bool=False, input_from_output: Union[Dict, None]=None, logging_tags: Dict=None, verify_output: bool=True, runtime_arguments: Union[Dict, None]=None, dynamic_block_index: Union[int, None]=None, dynamic_block_uuid: Union[str, None]=None, dynamic_upstream_block_uuids: Union[List[str], None]=None, data_integration_metadata: Dict=None, pipeline_run: PipelineRun=None, block_run_dicts: List[str]=None, **kwargs) -> Dict:
        if False:
            return 10
        '\n        Execute the block.\n\n        Args:\n            analyze_outputs: Whether to analyze the outputs of the block.\n            callback_url: The URL for the callback.\n            global_vars: Global variables for the block execution.\n            update_status: Whether to update the status of the block execution.\n            input_from_output: Input from the output of a previous block.\n            logging_tags: Tags for logging.\n            verify_output: Whether to verify the output of the block.\n            runtime_arguments: Runtime arguments for the block execution.\n            dynamic_block_index: Index of the dynamic block.\n            dynamic_block_uuid: UUID of the dynamic block.\n            dynamic_upstream_block_uuids: List of UUIDs of the dynamic upstream blocks.\n            **kwargs: Additional keyword arguments.\n\n        Returns:\n            The result of the block execution.\n        '
        if logging_tags is None:
            logging_tags = dict()
        extra_options = {}
        store_variables = True
        is_data_integration = False
        if self.project.is_feature_enabled(FeatureUUID.DATA_INTEGRATION_IN_BATCH_PIPELINE):
            is_data_integration = self.block.is_data_integration()
            di_settings = None
            blocks = [self.block]
            if self.block.upstream_blocks:
                blocks += self.block.upstream_blocks
            for block in blocks:
                is_current_block_run_block = block.uuid == self.block.uuid
                data_integration_settings = block.get_data_integration_settings(dynamic_block_index=dynamic_block_index, dynamic_upstream_block_uuids=dynamic_upstream_block_uuids, from_notebook=False, global_vars=global_vars, partition=self.execution_partition)
                try:
                    if data_integration_settings:
                        if is_current_block_run_block:
                            di_settings = data_integration_settings
                        data_integration_uuid = data_integration_settings.get('data_integration_uuid')
                        if data_integration_uuid:
                            if 'data_integration_runtime_settings' not in extra_options:
                                extra_options['data_integration_runtime_settings'] = {}
                            if 'module_file_paths' not in extra_options['data_integration_runtime_settings']:
                                extra_options['data_integration_runtime_settings']['module_file_paths'] = dict(destinations={}, sources={})
                            if self.block.is_source():
                                key = 'sources'
                                file_path_func = source_module_file_path
                            else:
                                key = 'destinations'
                                file_path_func = destination_module_file_path
                            if data_integration_uuid not in extra_options['data_integration_runtime_settings']['module_file_paths'][key]:
                                extra_options['data_integration_runtime_settings']['module_file_paths'][key][data_integration_uuid] = file_path_func(data_integration_uuid)
                            if is_current_block_run_block:
                                store_variables = False
                except Exception as err:
                    print(f'[WARNING] BlockExecutor._execute: {err}')
            if di_settings and data_integration_metadata and data_integration_metadata.get('controller') and data_integration_metadata.get('original_block_uuid'):
                original_block_uuid = data_integration_metadata.get('original_block_uuid')
                if is_data_integration:
                    arr = []
                    is_source = self.block.is_source()
                    data_integration_uuid = di_settings.get('data_integration_uuid')
                    catalog = di_settings.get('catalog', [])
                    block_run_block_uuids = []
                    if block_run_dicts:
                        block_run_block_uuids += [br.get('block_uuid') for br in block_run_dicts]
                    if data_integration_metadata.get('child'):
                        stream = data_integration_metadata.get('stream')
                        block_run_metadata = build_block_run_metadata(self.block, self.logger, data_integration_settings=di_settings, dynamic_block_index=dynamic_block_index, dynamic_upstream_block_uuids=dynamic_upstream_block_uuids, global_vars=global_vars, logging_tags=logging_tags, parent_stream=data_integration_metadata.get('parent_stream'), partition=self.execution_partition, selected_streams=[stream])
                        for br_metadata in block_run_metadata:
                            index = br_metadata.get('index') or 0
                            number_of_batches = br_metadata.get('number_of_batches') or 0
                            block_run_block_uuid = f'{original_block_uuid}:{data_integration_uuid}:{stream}:{index}'
                            if block_run_block_uuid not in block_run_block_uuids:
                                br = pipeline_run.create_block_run(block_run_block_uuid, metrics=merge_dict(dict(child=1, controller_block_uuid=self.block_uuid, is_last_block_run=index == number_of_batches - 1, original_block_uuid=original_block_uuid, stream=stream), br_metadata))
                                self.logger.info(f'Created block run {br.id} for block {br.block_uuid} in batch index {index} ({index + 1} out of {number_of_batches}).', **merge_dict(logging_tags, dict(data_integration_uuid=data_integration_uuid, index=index, number_of_batches=number_of_batches, original_block_uuid=original_block_uuid, stream=stream)))
                                arr.append(br)
                    else:
                        block_run_dicts = []

                        def _build_controller_block_run_dict(stream, block_run_block_uuids=block_run_block_uuids, controller_block_uuid=self.block_uuid, data_integration_uuid=data_integration_uuid, metrics: Dict=None, original_block_uuid=original_block_uuid, run_in_parallel: bool=False):
                            if False:
                                while True:
                                    i = 10
                            block_run_block_uuid = ':'.join([original_block_uuid, data_integration_uuid, stream, 'controller'])
                            if block_run_block_uuid not in block_run_block_uuids:
                                return dict(block_uuid=block_run_block_uuid, metrics=merge_dict(dict(child=1, controller=1, controller_block_uuid=controller_block_uuid, original_block_uuid=original_block_uuid, run_in_parallel=1 if run_in_parallel else 0, stream=stream), metrics or {}))
                        if is_source:
                            for stream_dict in get_selected_streams(catalog):
                                stream = stream_dict.get('tap_stream_id')
                                run_in_parallel = stream_dict.get('run_in_parallel', False)
                                block_dict = _build_controller_block_run_dict(stream, run_in_parallel=run_in_parallel)
                                if block_dict:
                                    block_run_dicts.append(block_dict)
                        else:
                            uuids_to_remove = self.block.inputs_only_uuids
                            up_uuids = self.block.upstream_block_uuids
                            if dynamic_upstream_block_uuids:
                                up_uuids += dynamic_upstream_block_uuids
                                for up_uuid in dynamic_upstream_block_uuids:
                                    up_block = self.pipeline.get_block(up_uuid)
                                    if up_block:
                                        uuids_to_remove.append(up_block.uuid)
                            up_uuids = [i for i in up_uuids if i not in uuids_to_remove]
                            for up_uuid in up_uuids:
                                run_in_parallel = False
                                up_block = self.pipeline.get_block(up_uuid)
                                if up_block.is_source():
                                    output_file_path_by_stream = get_streams_from_output_directory(up_block, execution_partition=self.execution_partition)
                                    for stream_id in output_file_path_by_stream.keys():
                                        stream_dict = get_streams_from_catalog(catalog, [stream_id])
                                        if stream_dict:
                                            run_in_parallel = stream_dict[0].get('run_in_parallel') or False
                                        block_dict = _build_controller_block_run_dict(stream_id, metrics=dict(parent_stream=up_uuid, run_in_parallel=run_in_parallel))
                                        if block_dict:
                                            block_run_dicts.append(block_dict)
                                else:
                                    stream_dict = get_streams_from_catalog(catalog, [up_uuid])
                                    if stream_dict:
                                        run_in_parallel = stream_dict[0].get('run_in_parallel') or False
                                    block_dict = _build_controller_block_run_dict(up_uuid, metrics=dict(run_in_parallel=run_in_parallel))
                                    if block_dict:
                                        block_run_dicts.append(block_dict)
                        block_run_dicts_length = len(block_run_dicts)
                        for (idx, block_run_dict) in enumerate(block_run_dicts):
                            metrics = block_run_dict['metrics']
                            stream = metrics['stream']
                            run_in_parallel = metrics.get('run_in_parallel') or 0
                            if not run_in_parallel or run_in_parallel == 0:
                                if idx >= 1:
                                    block_run_previous = block_run_dicts[idx - 1]
                                    metrics['upstream_block_uuids'] = [block_run_previous['block_uuid']]
                                if idx < block_run_dicts_length - 1:
                                    block_run_next = block_run_dicts[idx + 1]
                                    metrics['downstream_block_uuids'] = [block_run_next['block_uuid']]
                            br = pipeline_run.create_block_run(block_run_dict['block_uuid'], metrics=metrics)
                            self.logger.info(f'Created block run {br.id} for block {br.block_uuid} for stream {stream} {metrics}.', **merge_dict(logging_tags, dict(data_integration_uuid=data_integration_uuid, original_block_uuid=original_block_uuid, stream=stream)))
                            arr.append(br)
                    return arr
        result = self.block.execute_sync(analyze_outputs=analyze_outputs, execution_partition=self.execution_partition, global_vars=global_vars, logger=self.logger, logging_tags=logging_tags, run_all_blocks=True, update_status=update_status, input_from_output=input_from_output, verify_output=verify_output, runtime_arguments=runtime_arguments, dynamic_block_index=dynamic_block_index, dynamic_block_uuid=dynamic_block_uuid, dynamic_upstream_block_uuids=dynamic_upstream_block_uuids, store_variables=store_variables, **extra_options)
        if BlockType.DBT == self.block.type:
            self.block.run_tests(block=self.block, global_vars=global_vars, logger=self.logger, logging_tags=logging_tags)
        elif PipelineType.INTEGRATION != self.pipeline.type and (not is_data_integration or BlockLanguage.PYTHON == self.block.language):
            self.block.run_tests(execution_partition=self.execution_partition, global_vars=global_vars, logger=self.logger, logging_tags=logging_tags, update_tests=False, dynamic_block_uuid=dynamic_block_uuid)
        return result

    def _execute_conditional(self, global_vars: Dict, logging_tags: Dict, pipeline_run: PipelineRun, dynamic_block_index: Union[int, None]=None, dynamic_upstream_block_uuids: Union[List[str], None]=None) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Execute the conditional blocks.\n\n        Args:\n            global_vars: Global variables for the block execution.\n            logging_tags: Tags for logging.\n            pipeline_run: The pipeline run object.\n            dynamic_block_index: Index of the dynamic block.\n            dynamic_upstream_block_uuids: List of UUIDs of the dynamic upstream blocks.\n\n        Returns:\n            True if all conditional blocks evaluate to True, False otherwise.\n        '
        result = True
        for conditional_block in self.block.conditional_blocks:
            try:
                block_result = conditional_block.execute_conditional(self.block, dynamic_block_index=dynamic_block_index, dynamic_upstream_block_uuids=dynamic_upstream_block_uuids, execution_partition=self.execution_partition, global_vars=global_vars, logger=self.logger, logging_tags=logging_tags, pipeline_run=pipeline_run)
                if not block_result:
                    self.logger.info(f'Conditional block {conditional_block.uuid} evaluated as False for block {self.block.uuid}', **logging_tags)
                result = result and block_result
            except Exception as conditional_err:
                self.logger.exception(f'Failed to execute conditional block {conditional_block.uuid} for block {self.block.uuid}.', **merge_dict(logging_tags, dict(error=conditional_err)))
                result = False
        return result

    def _execute_callback(self, callback: str, global_vars: Dict, logging_tags: Dict, pipeline_run: PipelineRun, callback_kwargs: Dict=None, dynamic_block_index: Union[int, None]=None, dynamic_upstream_block_uuids: Union[List[str], None]=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Execute the callback blocks.\n\n        Args:\n            callback: The callback type ('on_success' or 'on_failure').\n            global_vars: Global variables for the block execution.\n            logging_tags: Tags for logging.\n            pipeline_run: The pipeline run object.\n            dynamic_block_index: Index of the dynamic block.\n            dynamic_upstream_block_uuids: List of UUIDs of the dynamic upstream blocks.\n        "
        arr = []
        if self.block.callback_block:
            arr.append(self.block.callback_block)
        if self.block.callback_blocks:
            arr += self.block.callback_blocks
        for callback_block in arr:
            try:
                callback_block.execute_callback(callback, callback_kwargs=callback_kwargs, dynamic_block_index=dynamic_block_index, dynamic_upstream_block_uuids=dynamic_upstream_block_uuids, execution_partition=self.execution_partition, global_vars=global_vars, logger=self.logger, logging_tags=logging_tags, parent_block=self.block, pipeline_run=pipeline_run)
            except Exception as callback_err:
                self.logger.exception(f'Failed to execute {callback} callback block {callback_block.uuid} for block {self.block.uuid}.', **merge_dict(logging_tags, dict(error=callback_err)))

    def _run_commands(self, block_run_id: int=None, global_vars: Dict=None, pipeline_run_id: int=None, **kwargs) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Run the commands for the block.\n\n        Args:\n            block_run_id: The ID of the block run.\n            global_vars: Global variables for the block execution.\n            **kwargs: Additional keyword arguments.\n\n        Returns:\n            A list of command arguments.\n        '
        cmd = f'/app/run_app.sh mage run {self.pipeline.repo_config.repo_path} {self.pipeline.uuid}'
        options = ['--block-uuid', self.block_uuid, '--executor-type', 'local_python']
        if self.execution_partition is not None:
            options += ['--execution-partition', self.execution_partition]
        if block_run_id is not None:
            options += ['--block-run-id', f'{block_run_id}']
        if pipeline_run_id:
            options += ['--pipeline-run-id', f'{pipeline_run_id}']
        if kwargs.get('template_runtime_configuration'):
            template_run_configuration = kwargs.get('template_runtime_configuration')
            options += ['--template-runtime-configuration', json.dumps(template_run_configuration)]
        return cmd.split(' ') + options

    def __update_block_run_status(self, status: BlockRun.BlockRunStatus, block_run_id: int=None, callback_url: str=None, pipeline_run: PipelineRun=None, tags: Dict=None):
        if False:
            i = 10
            return i + 15
        "\n        Update the status of block run by either updating the BlockRun db object or making\n        API call\n\n        Args:\n            status (str): 'completed' or 'failed'\n            block_run_id (int): the id of the block run\n            callback_url (str): with format http(s)://[host]:[port]/api/block_runs/[block_run_id]\n            tags (dict): tags used in logging\n        "
        if tags is None:
            tags = dict()
        if not block_run_id and (not callback_url):
            return
        try:
            if not block_run_id:
                block_run_id = int(callback_url.split('/')[-1])
            try:
                if status == BlockRun.BlockRunStatus.COMPLETED and pipeline_run is not None and is_dynamic_block(self.block):
                    create_block_runs_from_dynamic_block(self.block, pipeline_run, block_uuid=self.block.uuid if self.block.replicated_block else self.block_uuid)
            except Exception as err1:
                self.logger.exception(f'Failed to create block runs for dynamic block {self.block.uuid}.', **merge_dict(tags, dict(error=err1)))
            block_run = BlockRun.query.get(block_run_id)
            update_kwargs = dict(status=status)
            if status == BlockRun.BlockRunStatus.COMPLETED:
                update_kwargs['completed_at'] = datetime.now(tz=pytz.UTC)
            block_run.update(**update_kwargs)
            return
        except Exception as err2:
            self.logger.exception(f'Failed to update block run status to {status} for block {self.block.uuid}.', **merge_dict(tags, dict(error=err2)))
        response = requests.put(callback_url, data=json.dumps({'block_run': {'status': status}}), headers={'Content-Type': 'application/json'})
        self.logger.info(f'Callback response: {response.text}', **tags)

    def build_tags(self, **kwargs):
        if False:
            return 10
        '\n        Build tags for logging.\n\n        Args:\n            **kwargs: Additional keyword arguments.\n\n        Returns:\n            The built tags.\n        '
        default_tags = dict(block_type=self.block.type, block_uuid=self.block_uuid, pipeline_uuid=self.pipeline.uuid)
        if kwargs.get('block_run_id'):
            default_tags['block_run_id'] = kwargs.get('block_run_id')
        if kwargs.get('pipeline_run_id'):
            default_tags['pipeline_run_id'] = kwargs.get('pipeline_run_id')
        return merge_dict(kwargs.get('tags', {}), default_tags)