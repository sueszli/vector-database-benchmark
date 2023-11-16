from typing import Dict, List
from vyper import ast as vy_ast
from vyper.exceptions import StorageLayoutException
from vyper.semantics.analysis.base import CodeOffset, StorageSlot
from vyper.typing import StorageLayout
from vyper.utils import ceil32

def set_data_positions(vyper_module: vy_ast.Module, storage_layout_overrides: StorageLayout=None) -> StorageLayout:
    if False:
        i = 10
        return i + 15
    '\n    Parse the annotated Vyper AST, determine data positions for all variables,\n    and annotate the AST nodes with the position data.\n\n    Arguments\n    ---------\n    vyper_module : vy_ast.Module\n        Top-level Vyper AST node that has already been annotated with type data.\n    '
    code_offsets = set_code_offsets(vyper_module)
    storage_slots = set_storage_slots_with_overrides(vyper_module, storage_layout_overrides) if storage_layout_overrides is not None else set_storage_slots(vyper_module)
    return {'storage_layout': storage_slots, 'code_layout': code_offsets}

class StorageAllocator:
    """
    Keep track of which storage slots have been used. If there is a collision of
    storage slots, this will raise an error and fail to compile
    """

    def __init__(self):
        if False:
            return 10
        self.occupied_slots: Dict[int, str] = {}

    def reserve_slot_range(self, first_slot: int, n_slots: int, var_name: str) -> None:
        if False:
            print('Hello World!')
        '\n        Reserves `n_slots` storage slots, starting at slot `first_slot`\n        This will raise an error if a storage slot has already been allocated.\n        It is responsibility of calling function to ensure first_slot is an int\n        '
        list_to_check = [x + first_slot for x in range(n_slots)]
        self._reserve_slots(list_to_check, var_name)

    def _reserve_slots(self, slots: List[int], var_name: str) -> None:
        if False:
            i = 10
            return i + 15
        for slot in slots:
            self._reserve_slot(slot, var_name)

    def _reserve_slot(self, slot: int, var_name: str) -> None:
        if False:
            return 10
        if slot < 0 or slot >= 2 ** 256:
            raise StorageLayoutException(f'Invalid storage slot for var {var_name}, out of bounds: {slot}')
        if slot in self.occupied_slots:
            collided_var = self.occupied_slots[slot]
            raise StorageLayoutException(f"Storage collision! Tried to assign '{var_name}' to slot {slot} but it has already been reserved by '{collided_var}'")
        self.occupied_slots[slot] = var_name

def set_storage_slots_with_overrides(vyper_module: vy_ast.Module, storage_layout_overrides: StorageLayout) -> StorageLayout:
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse module-level Vyper AST to calculate the layout of storage variables.\n    Returns the layout as a dict of variable name -> variable info\n    '
    ret: Dict[str, Dict] = {}
    reserved_slots = StorageAllocator()
    for node in vyper_module.get_children(vy_ast.FunctionDef):
        type_ = node._metadata['type']
        if type_.nonreentrant is None:
            continue
        variable_name = f'nonreentrant.{type_.nonreentrant}'
        if variable_name in ret:
            _slot = ret[variable_name]['slot']
            type_.set_reentrancy_key_position(StorageSlot(_slot))
            continue
        if variable_name in storage_layout_overrides:
            reentrant_slot = storage_layout_overrides[variable_name]['slot']
            reserved_slots.reserve_slot_range(reentrant_slot, 1, variable_name)
            type_.set_reentrancy_key_position(StorageSlot(reentrant_slot))
            ret[variable_name] = {'type': 'nonreentrant lock', 'slot': reentrant_slot}
        else:
            raise StorageLayoutException(f'Could not find storage_slot for {variable_name}. Have you used the correct storage layout file?', node)
    for node in vyper_module.get_children(vy_ast.VariableDecl):
        if node.get('annotation.func.id') == 'immutable':
            continue
        varinfo = node.target._metadata['varinfo']
        if node.target.id in storage_layout_overrides:
            var_slot = storage_layout_overrides[node.target.id]['slot']
            storage_length = varinfo.typ.storage_size_in_words
            reserved_slots.reserve_slot_range(var_slot, storage_length, node.target.id)
            varinfo.set_position(StorageSlot(var_slot))
            ret[node.target.id] = {'type': str(varinfo.typ), 'slot': var_slot}
        else:
            raise StorageLayoutException(f'Could not find storage_slot for {node.target.id}. Have you used the correct storage layout file?', node)
    return ret

class SimpleStorageAllocator:

    def __init__(self, starting_slot: int=0):
        if False:
            return 10
        self._slot = starting_slot

    def allocate_slot(self, n, var_name):
        if False:
            print('Hello World!')
        ret = self._slot
        if self._slot + n >= 2 ** 256:
            raise StorageLayoutException(f'Invalid storage slot for var {var_name}, tried to allocate slots {self._slot} through {self._slot + n}')
        self._slot += n
        return ret

def set_storage_slots(vyper_module: vy_ast.Module) -> StorageLayout:
    if False:
        i = 10
        return i + 15
    '\n    Parse module-level Vyper AST to calculate the layout of storage variables.\n    Returns the layout as a dict of variable name -> variable info\n    '
    allocator = SimpleStorageAllocator()
    ret: Dict[str, Dict] = {}
    for node in vyper_module.get_children(vy_ast.FunctionDef):
        type_ = node._metadata['type']
        if type_.nonreentrant is None:
            continue
        variable_name = f'nonreentrant.{type_.nonreentrant}'
        if variable_name in ret:
            _slot = ret[variable_name]['slot']
            type_.set_reentrancy_key_position(StorageSlot(_slot))
            continue
        slot = allocator.allocate_slot(1, variable_name)
        type_.set_reentrancy_key_position(StorageSlot(slot))
        ret[variable_name] = {'type': 'nonreentrant lock', 'slot': slot}
    for node in vyper_module.get_children(vy_ast.VariableDecl):
        if node.is_constant or node.is_immutable:
            continue
        varinfo = node.target._metadata['varinfo']
        type_ = varinfo.typ
        n_slots = type_.storage_size_in_words
        slot = allocator.allocate_slot(n_slots, node.target.id)
        varinfo.set_position(StorageSlot(slot))
        ret[node.target.id] = {'type': str(type_), 'slot': slot}
    return ret

def set_calldata_offsets(fn_node: vy_ast.FunctionDef) -> None:
    if False:
        while True:
            i = 10
    pass

def set_memory_offsets(fn_node: vy_ast.FunctionDef) -> None:
    if False:
        i = 10
        return i + 15
    pass

def set_code_offsets(vyper_module: vy_ast.Module) -> Dict:
    if False:
        for i in range(10):
            print('nop')
    ret = {}
    offset = 0
    for node in vyper_module.get_children(vy_ast.VariableDecl, filters={'is_immutable': True}):
        varinfo = node.target._metadata['varinfo']
        type_ = varinfo.typ
        varinfo.set_position(CodeOffset(offset))
        len_ = ceil32(type_.size_in_bytes)
        ret[node.target.id] = {'type': str(type_), 'offset': offset, 'length': len_}
        offset += len_
    return ret