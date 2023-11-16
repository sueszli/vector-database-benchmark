import time
from collections import defaultdict
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import ray
from ray.air.execution.resources.request import ResourceRequest, AcquiredResources, RemoteRayEntity
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
from ray.util.placement_group import PlacementGroup, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

@DeveloperAPI
@dataclass
class PlacementGroupAcquiredResources(AcquiredResources):
    placement_group: PlacementGroup

    def _annotate_remote_entity(self, entity: RemoteRayEntity, bundle: Dict[str, float], bundle_index: int) -> RemoteRayEntity:
        if False:
            return 10
        bundle = bundle.copy()
        num_cpus = bundle.pop('CPU', 0)
        num_gpus = bundle.pop('GPU', 0)
        memory = bundle.pop('memory', 0.0)
        return entity.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=self.placement_group, placement_group_bundle_index=bundle_index, placement_group_capture_child_tasks=True), num_cpus=num_cpus, num_gpus=num_gpus, memory=memory, resources=bundle)

@DeveloperAPI
class PlacementGroupResourceManager(ResourceManager):
    """Resource manager using placement groups as the resource backend.

    This manager will use placement groups to fulfill resource requests. Requesting
    a resource will schedule the placement group. Acquiring a resource will
    return a ``PlacementGroupAcquiredResources`` that can be used to schedule
    Ray tasks and actors on the placement group. Freeing an acquired resource
    will destroy the associated placement group.

    Ray core does not emit events when resources are available. Instead, the
    scheduling state has to be periodically updated.

    Per default, placement group scheduling state is refreshed every time when
    resource state is inquired, but not more often than once every ``update_interval_s``
    seconds. Alternatively, staging futures can be retrieved (and awaited) with
    ``get_resource_futures()`` and state update can be force with ``update_state()``.

    Args:
        update_interval_s: Minimum interval in seconds between updating scheduling
            state of placement groups.

    """
    _resource_cls: AcquiredResources = PlacementGroupAcquiredResources

    def __init__(self, update_interval_s: float=0.1):
        if False:
            return 10
        self._pg_to_request: Dict[PlacementGroup, ResourceRequest] = {}
        self._request_to_staged_pgs: Dict[ResourceRequest, Set[PlacementGroup]] = defaultdict(set)
        self._request_to_ready_pgs: Dict[ResourceRequest, Set[PlacementGroup]] = defaultdict(set)
        self._staging_future_to_pg: Dict[ray.ObjectRef, PlacementGroup] = dict()
        self._pg_to_staging_future: Dict[PlacementGroup, ray.ObjectRef] = dict()
        self._acquired_pgs: Set[PlacementGroup] = set()
        self.update_interval_s = update_interval_s
        self._last_update = time.monotonic() - self.update_interval_s - 1

    def get_resource_futures(self) -> List[ray.ObjectRef]:
        if False:
            while True:
                i = 10
        return list(self._staging_future_to_pg.keys())

    def _maybe_update_state(self):
        if False:
            return 10
        now = time.monotonic()
        if now > self._last_update + self.update_interval_s:
            self.update_state()

    def update_state(self):
        if False:
            return 10
        (ready, not_ready) = ray.wait(list(self._staging_future_to_pg.keys()), num_returns=len(self._staging_future_to_pg), timeout=0)
        for future in ready:
            pg = self._staging_future_to_pg.pop(future)
            self._pg_to_staging_future.pop(pg)
            request = self._pg_to_request[pg]
            self._request_to_staged_pgs[request].remove(pg)
            self._request_to_ready_pgs[request].add(pg)
        self._last_update = time.monotonic()

    def request_resources(self, resource_request: ResourceRequest):
        if False:
            while True:
                i = 10
        pg = resource_request.to_placement_group()
        self._pg_to_request[pg] = resource_request
        self._request_to_staged_pgs[resource_request].add(pg)
        future = pg.ready()
        self._staging_future_to_pg[future] = pg
        self._pg_to_staging_future[pg] = future

    def cancel_resource_request(self, resource_request: ResourceRequest):
        if False:
            while True:
                i = 10
        if self._request_to_staged_pgs[resource_request]:
            pg = self._request_to_staged_pgs[resource_request].pop()
            future = self._pg_to_staging_future.pop(pg)
            self._staging_future_to_pg.pop(future)
            ray.cancel(future)
        else:
            pg = self._request_to_ready_pgs[resource_request].pop()
            if not pg:
                raise RuntimeError(f"Cannot cancel resource request: No placement group was staged or is ready. Make sure to not cancel more resource requests than you've created. Request: {resource_request}")
        self._pg_to_request.pop(pg)
        ray.util.remove_placement_group(pg)

    def has_resources_ready(self, resource_request: ResourceRequest) -> bool:
        if False:
            while True:
                i = 10
        if not bool(len(self._request_to_ready_pgs[resource_request])):
            self._maybe_update_state()
        return bool(len(self._request_to_ready_pgs[resource_request]))

    def acquire_resources(self, resource_request: ResourceRequest) -> Optional[PlacementGroupAcquiredResources]:
        if False:
            i = 10
            return i + 15
        if not self.has_resources_ready(resource_request):
            return None
        pg = self._request_to_ready_pgs[resource_request].pop()
        self._acquired_pgs.add(pg)
        return self._resource_cls(placement_group=pg, resource_request=resource_request)

    def free_resources(self, acquired_resource: PlacementGroupAcquiredResources):
        if False:
            i = 10
            return i + 15
        pg = acquired_resource.placement_group
        self._acquired_pgs.remove(pg)
        remove_placement_group(pg)
        self._pg_to_request.pop(pg)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        if not ray.is_initialized():
            return
        for staged_pgs in self._request_to_staged_pgs.values():
            for staged_pg in staged_pgs:
                remove_placement_group(staged_pg)
        for ready_pgs in self._request_to_ready_pgs.values():
            for ready_pg in ready_pgs:
                remove_placement_group(ready_pg)
        for acquired_pg in self._acquired_pgs:
            remove_placement_group(acquired_pg)
        self.__init__(update_interval_s=self.update_interval_s)

    def __del__(self):
        if False:
            return 10
        self.clear()