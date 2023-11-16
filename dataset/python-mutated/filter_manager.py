import logging
from typing import Optional
import ray
from ray.rllib.utils.annotations import DeveloperAPI
logger = logging.getLogger(__name__)

@DeveloperAPI
class FilterManager:
    """Manages filters and coordination across remote evaluators that expose
    `get_filters` and `sync_filters`.
    """

    @staticmethod
    @DeveloperAPI
    def synchronize(local_filters, worker_set, update_remote=True, timeout_seconds: Optional[float]=None, use_remote_data_for_update: bool=True):
        if False:
            return 10
        "Aggregates filters from remote workers (if use_remote_data_for_update=True).\n\n        Local copy is updated and then broadcasted to all remote evaluators\n        (if `update_remote` is True).\n\n        Args:\n            local_filters: Filters to be synchronized.\n            remotes: Remote evaluators with filters.\n            update_remote: Whether to push updates from the local filters to the remote\n                workers' filters.\n            timeout_seconds: How long to wait for filter to get or set filters\n            use_remote_data_for_update: Whether to use the `worker_set`'s remote workers\n                to update the local filters. If False, stats from the remote workers\n                will not be used and discarded.\n        "
        if not (update_remote or use_remote_data_for_update):
            return
        logger.debug(f'Synchronizing filters: {local_filters}')
        remote_filters = worker_set.foreach_worker(func=lambda worker: worker.get_filters(flush_after=True), local_worker=False, timeout_seconds=timeout_seconds)
        if len(remote_filters) != worker_set.num_healthy_remote_workers():
            logger.error('Failed to get remote filters from a rollout worker in FilterManager! Filtered metrics may be computed, but filtered wrong.')
        if use_remote_data_for_update:
            for rf in remote_filters:
                for k in local_filters:
                    local_filters[k].apply_changes(rf[k], with_buffer=False)
        if update_remote:
            copies = {k: v.as_serializable() for (k, v) in local_filters.items()}
            remote_copy = ray.put(copies)
            logger.debug('Updating remote filters ...')
            results = worker_set.foreach_worker(func=lambda worker: worker.sync_filters(ray.get(remote_copy)), local_worker=False, timeout_seconds=timeout_seconds)
            if len(results) != worker_set.num_healthy_remote_workers():
                logger.error('Failed to set remote filters to a rollout worker in FilterManager. Filtered metrics may be computed, but filtered wrong.')