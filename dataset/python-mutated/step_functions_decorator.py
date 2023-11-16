import os
import json
import time
from metaflow.decorators import StepDecorator
from metaflow.metadata import MetaDatum
from .dynamo_db_client import DynamoDbClient

class StepFunctionsInternalDecorator(StepDecorator):
    name = 'step_functions_internal'

    def task_pre_step(self, step_name, task_datastore, metadata, run_id, task_id, flow, graph, retry_count, max_user_code_retries, ubf_context, inputs):
        if False:
            for i in range(10):
                print('nop')
        meta = {}
        meta['aws-step-functions-execution'] = os.environ['METAFLOW_RUN_ID']
        meta['aws-step-functions-state-machine'] = os.environ['SFN_STATE_MACHINE']
        entries = [MetaDatum(field=k, value=v, type=k, tags=['attempt_id:{0}'.format(retry_count)]) for (k, v) in meta.items()]
        metadata.register_metadata(run_id, step_name, task_id, entries)

    def task_finished(self, step_name, flow, graph, is_task_ok, retry_count, max_user_code_retries):
        if False:
            print('Hello World!')
        if not is_task_ok:
            return
        if graph[step_name].type == 'foreach':
            self._save_foreach_cardinality(os.environ['AWS_BATCH_JOB_ID'], flow._foreach_num_splits, self._ttl())
        elif graph[step_name].is_inside_foreach and any((graph[n].type == 'join' and graph[graph[n].split_parents[-1]].type == 'foreach' for n in graph[step_name].out_funcs)):
            self._save_parent_task_id_for_foreach_join(os.environ['METAFLOW_SPLIT_PARENT_TASK_ID_FOR_FOREACH_JOIN'], os.environ['AWS_BATCH_JOB_ID'])

    def _save_foreach_cardinality(self, foreach_split_task_id, for_each_cardinality, ttl):
        if False:
            return 10
        DynamoDbClient().save_foreach_cardinality(foreach_split_task_id, for_each_cardinality, ttl)

    def _save_parent_task_id_for_foreach_join(self, foreach_split_task_id, foreach_join_parent_task_id):
        if False:
            print('Hello World!')
        DynamoDbClient().save_parent_task_id_for_foreach_join(foreach_split_task_id, foreach_join_parent_task_id)

    def _ttl(self):
        if False:
            i = 10
            return i + 15
        delta = 366 * 24 * 60 * 60
        delta = int(os.environ.get('METAFLOW_SFN_WORKFLOW_TIMEOUT', delta))
        return delta + 90 * 24 * 60 * 60 + int(time.time())