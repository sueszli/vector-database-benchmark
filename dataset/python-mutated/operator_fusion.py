from typing import List, Optional, Tuple
from ray.data._internal.compute import get_compute, is_task_compute
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle, TaskContext
from ray.data._internal.execution.operators.actor_pool_map_operator import ActorPoolMapOperator
from ray.data._internal.execution.operators.base_physical_operator import AllToAllOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.task_pool_map_operator import TaskPoolMapOperator
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators.all_to_all_operator import AbstractAllToAll, RandomShuffle, Repartition
from ray.data._internal.logical.operators.map_operator import AbstractUDFMap
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
INHERITABLE_REMOTE_ARGS = ['scheduling_strategy']

class OperatorFusionRule(Rule):
    """Fuses linear chains of compatible physical operators."""

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        if False:
            print('Hello World!')
        self._op_map = plan.op_map.copy()
        fused_dag = self._fuse_map_operators_in_dag(plan.dag)
        fused_dag = self._fuse_all_to_all_operators_in_dag(fused_dag)
        self._remove_output_depes(fused_dag)
        self._update_output_depes(fused_dag)
        return PhysicalPlan(fused_dag, self._op_map)

    def _remove_output_depes(self, op: PhysicalOperator) -> None:
        if False:
            while True:
                i = 10
        for input in op._input_dependencies:
            input._output_dependencies = []
            self._remove_output_depes(input)

    def _update_output_depes(self, op: PhysicalOperator) -> None:
        if False:
            while True:
                i = 10
        for input in op._input_dependencies:
            input._output_dependencies.append(op)
            self._update_output_depes(input)

    def _fuse_map_operators_in_dag(self, dag: PhysicalOperator) -> MapOperator:
        if False:
            i = 10
            return i + 15
        'Starting at the given operator, traverses up the DAG of operators\n        and recursively fuses compatible MapOperator -> MapOperator pairs.\n        Returns the current (root) operator after completing upstream operator fusions.\n        '
        upstream_ops = dag.input_dependencies
        while len(upstream_ops) == 1 and isinstance(dag, MapOperator) and isinstance(upstream_ops[0], MapOperator) and self._can_fuse(dag, upstream_ops[0]):
            dag = self._get_fused_map_operator(dag, upstream_ops[0])
            upstream_ops = dag.input_dependencies
        self._propagate_target_max_block_size_to_input(dag)
        dag._input_dependencies = [self._fuse_map_operators_in_dag(upstream_op) for upstream_op in upstream_ops]
        return dag

    def _fuse_all_to_all_operators_in_dag(self, dag: AllToAllOperator) -> AllToAllOperator:
        if False:
            print('Hello World!')
        'Starting at the given operator, traverses up the DAG of operators\n        and recursively fuses compatible MapOperator -> AllToAllOperator pairs.\n\n        Also, sets the target block size of the immediately upstream map op to\n        match the shuffle block size. We use a larger block size for shuffles\n        because tiny blocks are bad for I/O performance.\n\n        Returns the current (root) operator after completing upstream operator fusions.\n        '
        upstream_ops = dag.input_dependencies
        while len(upstream_ops) == 1 and isinstance(dag, AllToAllOperator) and isinstance(upstream_ops[0], MapOperator):
            if self._can_fuse(dag, upstream_ops[0]):
                dag = self._get_fused_all_to_all_operator(dag, upstream_ops[0])
                upstream_ops = dag.input_dependencies
            else:
                map_op = upstream_ops[0]
                map_op._target_max_block_size = self._get_merged_target_max_block_size(upstream_ops[0].target_max_block_size, dag.target_max_block_size)
                break
        self._propagate_target_max_block_size_to_input(dag)
        dag._input_dependencies = [self._fuse_all_to_all_operators_in_dag(upstream_op) for upstream_op in upstream_ops]
        return dag

    def _can_fuse(self, down_op: PhysicalOperator, up_op: PhysicalOperator) -> bool:
        if False:
            print('Hello World!')
        'Returns whether the provided downstream operator can be fused with the given\n        upstream operator.\n\n        We currently support fusing two operators if the following are all true:\n            * We are fusing either MapOperator -> MapOperator or\n              MapOperator -> AllToAllOperator.\n            * They either use the same compute configuration, or the upstream operator\n              uses a task pool while the downstream operator uses an actor pool.\n            * If both operators involve callable classes, the callable classes are\n              the same class AND constructor args are the same for both.\n            * They have compatible remote arguments.\n        '
        from ray.data._internal.logical.operators.map_operator import AbstractMap, AbstractUDFMap
        if not (isinstance(up_op, TaskPoolMapOperator) and isinstance(down_op, (TaskPoolMapOperator, ActorPoolMapOperator)) or (isinstance(up_op, TaskPoolMapOperator) and isinstance(down_op, AllToAllOperator))):
            return False
        down_logical_op = self._op_map[down_op]
        up_logical_op = self._op_map[up_op]
        if up_op.get_additional_split_factor() > 1:
            return False
        if not down_logical_op._input_dependencies:
            return False
        if not (isinstance(up_logical_op, AbstractMap) and isinstance(down_logical_op, AbstractMap) or (isinstance(up_logical_op, AbstractMap) and isinstance(down_logical_op, RandomShuffle)) or (isinstance(up_logical_op, AbstractMap) and isinstance(down_logical_op, Repartition))):
            return False
        if isinstance(down_logical_op, Repartition) and (not down_logical_op._shuffle):
            return False
        if isinstance(down_logical_op, AbstractUDFMap) and isinstance(up_logical_op, AbstractUDFMap):
            if is_task_compute(down_logical_op._compute) and get_compute(up_logical_op._compute) != get_compute(down_logical_op._compute):
                return False
        if not _are_remote_args_compatible(getattr(up_logical_op, '_ray_remote_args', {}), getattr(down_logical_op, '_ray_remote_args', {})):
            return False
        if not self._can_merge_target_max_block_size(up_op.target_max_block_size, down_op.target_max_block_size):
            return False
        return True

    def _can_merge_target_max_block_size(self, up_target_max_block_size: Optional[int], down_target_max_block_size: Optional[int]):
        if False:
            for i in range(10):
                print('nop')
        if up_target_max_block_size is not None:
            if down_target_max_block_size is None:
                down_target_max_block_size = DataContext.get_current().target_max_block_size
            if up_target_max_block_size != down_target_max_block_size:
                return False
        return True

    def _get_merged_target_max_block_size(self, up_target_max_block_size: Optional[int], down_target_max_block_size: Optional[int]):
        if False:
            for i in range(10):
                print('nop')
        if up_target_max_block_size is not None:
            assert down_target_max_block_size is None or down_target_max_block_size == up_target_max_block_size
            return up_target_max_block_size
        else:
            return down_target_max_block_size

    def _propagate_target_max_block_size_to_input(self, dag):
        if False:
            i = 10
            return i + 15
        upstream_ops = dag.input_dependencies
        if len(upstream_ops) == 1 and isinstance(upstream_ops[0], InputDataBuffer) and self._can_merge_target_max_block_size(upstream_ops[0].target_max_block_size, dag.target_max_block_size):
            upstream_ops[0]._target_max_block_size = self._get_merged_target_max_block_size(upstream_ops[0].target_max_block_size, dag.target_max_block_size)

    def _get_fused_map_operator(self, down_op: MapOperator, up_op: MapOperator) -> MapOperator:
        if False:
            print('Hello World!')
        assert self._can_fuse(down_op, up_op), f'Current rule supports fusing MapOperator->MapOperator, but received: {type(up_op).__name__} -> {type(down_op).__name__}'
        name = up_op.name + '->' + down_op.name
        down_logical_op = self._op_map.pop(down_op)
        up_logical_op = self._op_map.pop(up_op)
        down_min_rows_per_block = down_logical_op._min_rows_per_block if isinstance(down_logical_op, AbstractUDFMap) else None
        up_min_rows_per_block = up_logical_op._min_rows_per_block if isinstance(up_logical_op, AbstractUDFMap) else None
        if down_min_rows_per_block is not None and up_min_rows_per_block is not None:
            min_rows_per_block = max(down_min_rows_per_block, up_min_rows_per_block)
        elif up_min_rows_per_block is not None:
            min_rows_per_block = up_min_rows_per_block
        else:
            min_rows_per_block = down_min_rows_per_block
        target_max_block_size = self._get_merged_target_max_block_size(up_op.target_max_block_size, down_op.target_max_block_size)
        compute = None
        if isinstance(down_logical_op, AbstractUDFMap):
            compute = get_compute(down_logical_op._compute)
        ray_remote_args = up_logical_op._ray_remote_args
        input_deps = up_op.input_dependencies
        assert len(input_deps) == 1
        input_op = input_deps[0]
        op = MapOperator.create(up_op.get_map_transformer().fuse(down_op.get_map_transformer()), input_op, target_max_block_size=target_max_block_size, name=name, compute_strategy=compute, min_rows_per_bundle=min_rows_per_block, ray_remote_args=ray_remote_args)
        if isinstance(up_logical_op, AbstractUDFMap):
            input_op = up_logical_op.input_dependency
        else:
            input_op = up_logical_op
        if isinstance(down_logical_op, AbstractUDFMap):
            logical_op = AbstractUDFMap(name, input_op, down_logical_op._fn, down_logical_op._fn_args, down_logical_op._fn_kwargs, down_logical_op._fn_constructor_args, down_logical_op._fn_constructor_kwargs, min_rows_per_block, compute, ray_remote_args)
        else:
            from ray.data._internal.logical.operators.map_operator import AbstractMap
            logical_op = AbstractMap(name, input_op, ray_remote_args)
        self._op_map[op] = logical_op
        return op

    def _get_fused_all_to_all_operator(self, down_op: AllToAllOperator, up_op: MapOperator) -> AllToAllOperator:
        if False:
            print('Hello World!')
        assert self._can_fuse(down_op, up_op), f'Current rule supports fusing MapOperator -> AllToAllOperator, but received: {type(up_op).__name__} -> {type(down_op).__name__}'
        name = up_op.name + '->' + down_op.name
        down_logical_op: AbstractAllToAll = self._op_map.pop(down_op)
        up_logical_op: AbstractUDFMap = self._op_map.pop(up_op)
        ray_remote_args = up_logical_op._ray_remote_args
        down_transform_fn = down_op.get_transformation_fn()
        up_map_transformer = up_op.get_map_transformer()

        def fused_all_to_all_transform_fn(blocks: List[RefBundle], ctx: TaskContext) -> Tuple[List[RefBundle], StatsDict]:
            if False:
                print('Hello World!')
            "To fuse MapOperator->AllToAllOperator, we store the map function\n            in the TaskContext so that it may be used by the downstream\n            AllToAllOperator's transform function."
            ctx.upstream_map_transformer = up_map_transformer
            ctx.upstream_map_ray_remote_args = ray_remote_args
            return down_transform_fn(blocks, ctx)
        input_deps = up_op.input_dependencies
        assert len(input_deps) == 1
        input_op = input_deps[0]
        target_max_block_size = self._get_merged_target_max_block_size(up_op.target_max_block_size, down_op.target_max_block_size)
        op = AllToAllOperator(fused_all_to_all_transform_fn, input_op, target_max_block_size=target_max_block_size, num_outputs=down_op._num_outputs, sub_progress_bar_names=down_op._sub_progress_bar_names, name=name)
        input_op = up_logical_op
        if isinstance(down_logical_op, RandomShuffle):
            logical_op = RandomShuffle(input_op, name=name, ray_remote_args=ray_remote_args)
        elif isinstance(down_logical_op, Repartition):
            logical_op = Repartition(input_op, num_outputs=down_logical_op._num_outputs, shuffle=down_logical_op._shuffle)
        self._op_map[op] = logical_op
        return op

def _are_remote_args_compatible(prev_args, next_args):
    if False:
        print('Hello World!')
    'Check if Ray remote arguments are compatible for merging.'
    prev_args = _canonicalize(prev_args)
    next_args = _canonicalize(next_args)
    remote_args = next_args.copy()
    for key in INHERITABLE_REMOTE_ARGS:
        if key in prev_args:
            remote_args[key] = prev_args[key]
    if prev_args != remote_args:
        return False
    return True

def _canonicalize(remote_args: dict) -> dict:
    if False:
        for i in range(10):
            print('nop')
    'Returns canonical form of given remote args.'
    remote_args = remote_args.copy()
    if 'num_cpus' not in remote_args or remote_args['num_cpus'] is None:
        remote_args['num_cpus'] = 1
    if 'num_gpus' not in remote_args or remote_args['num_gpus'] is None:
        remote_args['num_gpus'] = 0
    resources = remote_args.get('resources', {})
    for (k, v) in list(resources.items()):
        if v is None or v == 0.0:
            del resources[k]
    remote_args['resources'] = resources
    return remote_args