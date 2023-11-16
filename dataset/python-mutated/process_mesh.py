import copy
import numpy as np
import paddle
from paddle.framework import core
_g_previous_process_mesh = None
_g_current_process_mesh = None
_g_unique_process_mesh_map = {}

def get_current_process_mesh():
    if False:
        i = 10
        return i + 15
    global _g_current_process_mesh
    return _g_current_process_mesh

def set_current_process_mesh(process_mesh):
    if False:
        for i in range(10):
            print('nop')
    global _g_previous_process_mesh
    global _g_current_process_mesh
    _g_previous_process_mesh = _g_current_process_mesh
    _g_current_process_mesh = process_mesh

def reset_current_process_mesh():
    if False:
        for i in range(10):
            print('nop')
    global _g_previous_process_mesh
    global _g_current_process_mesh
    _g_current_process_mesh = _g_previous_process_mesh

def get_unique_id_for_process_mesh(shape, process_ids):
    if False:
        print('Hello World!')
    key = f'shape {shape}, process_ids {process_ids}'
    global _g_unique_process_mesh_map
    if key in _g_unique_process_mesh_map:
        unique_id = _g_unique_process_mesh_map[key]
    else:
        unique_id = len(_g_unique_process_mesh_map) + 1
        _g_unique_process_mesh_map[key] = unique_id
    return unique_id

def retrive_unique_id_for_process_mesh(shape, process_ids):
    if False:
        while True:
            i = 10
    key = f'shape {shape}, process_ids {process_ids}'
    global _g_unique_process_mesh_map
    assert key in _g_unique_process_mesh_map
    return _g_unique_process_mesh_map[key]

def get_unique_process_mesh_map():
    if False:
        i = 10
        return i + 15
    global _g_unique_process_mesh_map
    return _g_unique_process_mesh_map

class ProcessMesh(core.ProcessMesh):
    """
    The `ProcessMesh` object describes the Cartesian topology of the used processes.

    Args:
        mesh (list|numpy.array): an n-dimensional array describes the topology
            of the processes.
        dim_names (list, optional): the i-th element of this list gives the name of the
            i-th dimension of the mesh.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> mesh = dist.ProcessMesh([[2, 4, 5], [0, 1, 3]], dim_names=["x", "y"])
            >>> assert mesh.shape == [2, 3]
            >>> assert mesh.process_ids == [2, 4, 5, 0, 1, 3]

    """

    def __init__(self, mesh=None, dim_names=None, shape=None, process_ids=None):
        if False:
            for i in range(10):
                print('nop')
        if mesh is None:
            assert shape is not None
            assert process_ids is not None
            mesh = np.array(process_ids).reshape(shape)
        if not isinstance(mesh, list) and (not isinstance(mesh, np.ndarray)):
            raise ValueError('The mesh must be an instance of list or np.ndarray.')
        if isinstance(mesh, list):
            mesh = np.array(mesh)
        if dim_names is not None and (not isinstance(dim_names, list)):
            raise ValueError('The dim_names must be an instance of list.')
        self._mesh = mesh
        self._shape = list(self._mesh.shape)
        self._process_ids = self._mesh.flatten().tolist()
        assert all((isinstance(p, int) for p in self._process_ids)), 'All elements of the mesh must be integer'
        assert min(self._process_ids) >= 0, 'All elements of the mesh must be >= 0.'
        unique_process_ids = set(self._process_ids)
        assert len(unique_process_ids) == len(self._process_ids), 'All elements of the mesh must be unique.'
        if dim_names is not None:
            assert len(dim_names) == len(self._shape), 'The length of dims_names must be same as the shape of the mesh.'
            self._dim_names = copy.deepcopy(dim_names)
        else:
            self._dim_names = ['d' + str(i) for i in range(len(self._shape))]
        unique_dim_names = set(self._dim_names)
        assert len(unique_dim_names) == len(self._dim_names), f'All dim_names {dim_names} must be unique.'
        core.ProcessMesh.__init__(self, self._shape, self._process_ids, self._dim_names)
        from .static.dist_context import get_default_distributed_context
        default_dist_cxt = get_default_distributed_context()
        default_dist_cxt.add_process_mesh(self)
        from .static.process_group import get_process_group
        pg0 = get_process_group(0)
        pg0.add_ranks(self.process_ids)
        self._unique_id = get_unique_id_for_process_mesh(self._shape, self._process_ids)

    @property
    def mesh(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the underlying mesh of ProcessMesh.\n        '
        return self._mesh

    @property
    def dim_names(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the underlying dimension names of ProcessMesh.\n        '
        return self._dim_names

    @property
    def unique_id(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the unique id of ProcessMesh.\n        NOTE\n        Unique id only take process_ids and shape into account.\n        Different ProcessMesh with same process_ids and shape have same unique id.\n        '
        return self._unique_id

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(index, tuple):
            new_dim_names = []
            for (i, item) in enumerate(index):
                if isinstance(item, slice):
                    new_dim_names.append(self._dim_names[i])
            new_mesh = self._mesh[index]
            if new_mesh.shape:
                return ProcessMesh(new_mesh, new_dim_names)
            else:
                return ProcessMesh([new_mesh])
        elif isinstance(index, slice):
            new_mesh = self._mesh[index]
            new_dim_names = self._dim_names
            return ProcessMesh(new_mesh, new_dim_names)
        else:
            new_mesh = self._mesh[index]
            new_dim_names = self._dim_names[1:]
            if new_mesh.shape:
                return ProcessMesh(new_mesh, new_dim_names)
            else:
                return ProcessMesh([new_mesh])

    def get_dim_size(self, dim_name):
        if False:
            return 10
        assert dim_name in self._dim_names
        return self._shape[self._dim_names.index(dim_name)]

    def get_mesh_with_dim(self, dim_name):
        if False:
            return 10
        assert dim_name in self._dim_names, f'{dim_name} is not a valid dim name.'
        index_axis = self._dim_names.index(dim_name)
        new_order = [index_axis] + [i for i in range(len(self._dim_names)) if i != index_axis]
        new_dim_names = [dim_name] + [dim for dim in self._dim_names if dim != dim_name]
        new_mesh = self._mesh.transpose(new_order)
        return ProcessMesh(new_mesh, new_dim_names)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        set_current_process_mesh(self)
        default_prog = paddle.static.default_main_program()
        cur_block = default_prog.current_block()
        self._old_var_names = list(cur_block.vars.keys())
        self._old_op_size = len(cur_block.ops)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if False:
            for i in range(10):
                print('nop')
        from .static.dist_op import DistributedOperator
        from .static.dist_tensor import DistributedTensor
        default_prog = paddle.static.default_main_program()
        cur_block = default_prog.current_block()
        new_var_names = list(cur_block.vars.keys())
        new_op_size = len(cur_block.ops)
        from .static.dist_context import get_default_distributed_context
        default_dist_ctx = get_default_distributed_context()
        for name in new_var_names:
            if name not in self._old_var_names:
                tensor = cur_block.vars[name]
                dist_tensor = default_dist_ctx.get_dist_tensor_for_program(tensor)
                if dist_tensor is None:
                    dist_tensor = DistributedTensor(cur_block.vars[name])
                    dist_tensor.dist_attr.process_mesh = self
                    dist_tensor.dist_attr.mark_annotated('process_mesh')
                    default_dist_ctx.add_dist_tensor_for_program(dist_tensor)
                elif dist_tensor.dist_attr.process_mesh is None:
                    dist_tensor.dist_attr.process_mesh = self
                    dist_tensor.dist_attr.mark_annotated('process_mesh')
        for idx in range(self._old_op_size, new_op_size):
            op = cur_block.ops[idx]
            dist_op = default_dist_ctx.get_dist_op_for_program(op)
            if dist_op is None:
                dist_op = DistributedOperator(op)
                dist_op.dist_attr.process_mesh = self
                dist_op.dist_attr.mark_annotated('process_mesh')
                default_dist_ctx.add_dist_op_for_program(dist_op)
            elif dist_op.dist_attr.process_mesh is None:
                dist_op.dist_attr.process_mesh = self
                dist_op.dist_attr.mark_annotated('process_mesh')
        reset_current_process_mesh()

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        if id(self) in memo:
            return memo[id(self)]
        new_process_mesh = ProcessMesh(np.array(self.mesh), self.dim_names)
        memo[id(self)] = new_process_mesh
        return new_process_mesh

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, (ProcessMesh, core.ProcessMesh)):
            return False
        if self.shape != other.shape or self.process_ids != other.process_ids:
            return False
        return True

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def __str__(self):
        if False:
            return 10
        str = 'shape {}, process_ids {}, dim_nams {}'.format(self.shape, self.process_ids, self.dim_names)
        return str

def compute_compatible_process_mesh(process_mesh_list):
    if False:
        for i in range(10):
            print('nop')
    'Compute the compatible process mesh given a list of process meshes.'
    if not process_mesh_list:
        return None

    def _compute_compatible_process_mesh_of_two(pm1, pm2):
        if False:
            print('Hello World!')
        if pm1 is None:
            return (True, pm2)
        if pm2 is None:
            return (True, pm1)
        if pm1 == pm2:
            return (True, pm1)
        if pm1.process_ids == pm2.process_ids:
            if len(pm1.shape) >= len(pm2.shape):
                return (True, pm1)
            else:
                return (True, pm2)
        process_set1 = set(pm1.process_ids)
        process_set2 = set(pm2.process_ids)
        if process_set1.issubset(process_set2):
            return (True, pm2)
        if process_set2.issubset(process_set1):
            return (True, pm1)
        return (False, None)
    compatible_result = None
    for process_mesh in process_mesh_list:
        (compatible, compatible_result) = _compute_compatible_process_mesh_of_two(compatible_result, process_mesh)
        if not compatible:
            return None
    return copy.deepcopy(compatible_result)

def merge_process_meshes(process_meshes):
    if False:
        print('Hello World!')
    'Merge a list of process meshes.'
    merged_process_mesh = None
    merged_process_ids = set()
    for process_mesh in process_meshes:
        if process_mesh is not None:
            process_ids = set(process_mesh.process_ids)
            merged_process_ids = merged_process_ids.union(process_ids)
    if len(merged_process_ids) != 0:
        merged_process_mesh = ProcessMesh(list(merged_process_ids))
    return merged_process_mesh