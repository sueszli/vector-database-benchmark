import lldb
import os
import sys
import re
_MAX_CHILDREN = 250
_MAX_CTRL_INDEX = 1000
_MODULE_NAME = os.path.basename(__file__).split('.')[0]

def _get_function_name(instance=None):
    if False:
        i = 10
        return i + 15
    'Return the name of the calling function'
    class_name = f'{type(instance).__name__}.' if instance else ''
    return class_name + sys._getframe(1).f_code.co_name

class flat_map_slot_type:
    CLASS_PATTERN = '^phmap::priv::raw_hash_set<phmap::priv::FlatHashMapPolicy.*>::slot_type$'
    HAS_SUMMARY = True
    IS_SYNTHETIC_PROVIDER = False

    @staticmethod
    def summary(valobj, _):
        if False:
            for i in range(10):
                print('nop')
        try:
            valobj = valobj.GetChildMemberWithName('value')
            first = valobj.GetChildMemberWithName('first').GetSummary()
            if not first:
                first = '{...}'
            second = valobj.GetChildMemberWithName('second').GetSummary()
            if not second:
                second = '{...}'
            return f'{{{first}, {second}}}'
        except BaseException as ex:
            print(f'{_get_function_name()} -> {ex}')
        return ''

class node_map_slot_type:
    CLASS_PATTERN = 'phmap::priv::raw_hash_set<phmap::priv::NodeHashMapPolicy.*>::slot_type$'
    HAS_SUMMARY = True
    IS_SYNTHETIC_PROVIDER = False

    @staticmethod
    def summary(valobj, _):
        if False:
            while True:
                i = 10
        try:
            valobj = valobj.Dereference()
            first = valobj.GetChildMemberWithName('first').GetSummary()
            if not first:
                first = '{...}'
            second = valobj.GetChildMemberWithName('second').GetSummary()
            if not second:
                second = '{...}'
            return f'{{{first}, {second}}}'
        except BaseException as ex:
            print(f'{_get_function_name()} -> {ex}')
        return '{?}'

class node_set_slot_type:
    CLASS_PATTERN = 'phmap::priv::raw_hash_set<phmap::priv::NodeHashSetPolicy.*>::slot_type$'
    HAS_SUMMARY = True
    IS_SYNTHETIC_PROVIDER = False

    @staticmethod
    def summary(valobj, _):
        if False:
            i = 10
            return i + 15
        try:
            summary = valobj.Dereference().GetSummary()
            if not summary:
                summary = '{...}'
            return summary
        except BaseException as ex:
            print(f'{_get_function_name()} -> {ex}')
        return '{?}'

class flat_hash_map_or_set:
    CLASS_PATTERN = '^phmap::flat_hash_(map|set)<.*>$'
    HAS_SUMMARY = True
    IS_SYNTHETIC_PROVIDER = True

    @staticmethod
    def summary(valobj, _):
        if False:
            return 10
        try:
            valobj = valobj.GetNonSyntheticValue()
            size = valobj.GetChildMemberWithName('size_').GetValueAsUnsigned()
            capacity = valobj.GetChildMemberWithName('capacity_').GetValueAsUnsigned()
            return f'size = {size} (capacity = {capacity})'
        except BaseException as ex:
            print(f'{_get_function_name()} -> {ex}')
        return '{?}'

    def __init__(self, valobj, _):
        if False:
            print('Hello World!')
        self.valobj = valobj
        self.slots_ = self.slot_type = self.ctrl_ = None
        self.size_ = self.capacity_ = self.slot_size = 0

    def num_children(self):
        if False:
            while True:
                i = 10
        return min(self.size_, _MAX_CHILDREN)

    def has_children(self):
        if False:
            return 10
        return True

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.size_ = self.valobj.GetChildMemberWithName('size_').GetValueAsUnsigned()
            self.capacity_ = self.valobj.GetChildMemberWithName('capacity_').GetValueAsUnsigned()
            self.slots_ = self.valobj.GetChildMemberWithName('slots_')
            self.slot_type = self.slots_.GetType().GetPointeeType()
            self.slot_size = self.slot_type.GetByteSize()
            self.ctrl_ = self.valobj.GetChildMemberWithName('ctrl_')
        except BaseException as ex:
            print(f'{_get_function_name(self)} -> {ex}')

    def get_child_index(self, name):
        if False:
            i = 10
            return i + 15
        try:
            if name in ('size_', 'capacity_'):
                return -1
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1

    def get_child_at_index(self, index):
        if False:
            while True:
                i = 10
        try:
            if index < 0:
                return None
            if index >= self.size_ or index >= _MAX_CHILDREN:
                return None
            real_idx = -1
            for idx in range(min(self.capacity_ + 3, _MAX_CTRL_INDEX)):
                ctrl = self.ctrl_.GetChildAtIndex(idx).GetValueAsSigned()
                if ctrl >= -1:
                    real_idx += 1
                    if real_idx == index:
                        return self.slots_.CreateChildAtOffset(f'[{index}]', idx * self.slot_size, self.slot_type)
        except BaseException as ex:
            print(f'{_get_function_name(self)} -> {ex}')
        return None

class parallel_flat_or_node_map_or_set:
    CLASS_PATTERN = '^phmap::parallel_(flat|node)_hash_(map|set)<.*>$'
    HAS_SUMMARY = True
    IS_SYNTHETIC_PROVIDER = True
    REGEX_EXTRACT_ARRAY_SIZE = re.compile('std::array\\s*<.*,\\s*(\\d+)\\s*>')

    @staticmethod
    def _get_size_and_capacity(valobj):
        if False:
            return 10
        try:
            valobj = valobj.GetNonSyntheticValue()
            sets = valobj.GetChildMemberWithName('sets_')
            m = parallel_flat_or_node_map_or_set.REGEX_EXTRACT_ARRAY_SIZE.match(sets.GetType().GetName())
            n_buckets = int(m.group(1))
            buckets = sets.GetChildMemberWithName('_M_elems')
            size = capacity = 0
            for idx in range(n_buckets):
                bucket = buckets.GetChildAtIndex(idx).GetChildMemberWithName('set_')
                size += bucket.GetChildMemberWithName('size_').GetValueAsUnsigned()
                capacity += bucket.GetChildMemberWithName('capacity_').GetValueAsUnsigned()
            return (size, capacity, n_buckets)
        except:
            return ('?', '?', 0)

    @staticmethod
    def summary(valobj, _):
        if False:
            print('Hello World!')
        (size, capacity, _) = parallel_flat_or_node_map_or_set._get_size_and_capacity(valobj)
        return f'size = {size} (capacity = {capacity})'

    def __init__(self, valobj, _):
        if False:
            i = 10
            return i + 15
        self.valobj = valobj
        self.buckets = self.slot_type = None
        self.size_ = self.capacity_ = self.n_buckets_ = self.slot_type = self.ctrl_size = 0

    def num_children(self):
        if False:
            i = 10
            return i + 15
        return min(self.size_, _MAX_CHILDREN)

    def has_children(self):
        if False:
            while True:
                i = 10
        return True

    def update(self):
        if False:
            print('Hello World!')
        try:
            (self.size_, self.capacity_, self.n_buckets_) = self._get_size_and_capacity(self.valobj)
            self.buckets = self.valobj.GetChildMemberWithName('sets_').GetChildMemberWithName('_M_elems')
            bucket0 = self.buckets.GetChildAtIndex(0).GetChildMemberWithName('set_')
            self.slot_type = bucket0.GetChildMemberWithName('slots_').GetType().GetPointeeType()
            self.slot_size = self.slot_type.GetByteSize()
        except BaseException as ex:
            print(f'{_get_function_name(self)} -> {ex}')

    def get_child_index(self, name):
        if False:
            print('Hello World!')
        try:
            if name in 'sets_':
                return -1
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1

    def get_child_at_index(self, index):
        if False:
            i = 10
            return i + 15
        try:
            if index < 0:
                return None
            if index >= self.size_ or index >= _MAX_CHILDREN:
                return None
            real_idx = -1
            total_idx = 0
            for idx in range(self.n_buckets_):
                bucket = self.buckets.GetChildAtIndex(idx).GetChildMemberWithName('set_')
                size = bucket.GetChildMemberWithName('size_').GetValueAsUnsigned()
                if size:
                    slots_ = bucket.GetChildMemberWithName('slots_')
                    ctrl_ = bucket.GetChildMemberWithName('ctrl_')
                    for jdx in range(size):
                        ctrl = ctrl_.GetChildAtIndex(jdx).GetValueAsSigned()
                        if ctrl >= -1:
                            real_idx += 1
                            if real_idx == index:
                                return slots_.CreateChildAtOffset(f'[{index}]', jdx * self.slot_size, self.slot_type)
                    total_idx += size
                    if total_idx > _MAX_CHILDREN:
                        return None
        except BaseException as ex:
            print(f'{_get_function_name(self)} -> {ex}')
        return None

def __lldb_init_module(debugger, internal_dict):
    if False:
        while True:
            i = 10
    for sp in (flat_map_slot_type, node_map_slot_type, node_set_slot_type, flat_hash_map_or_set, parallel_flat_or_node_map_or_set):
        if sp.HAS_SUMMARY:
            debugger.HandleCommand(f'type summary add --regex "{sp.CLASS_PATTERN}" --python-function {_MODULE_NAME}.{sp.__name__}.summary --category phmap --expand')
        if sp.IS_SYNTHETIC_PROVIDER:
            debugger.HandleCommand(f'type synthetic add --regex "{sp.CLASS_PATTERN}" --python-class {_MODULE_NAME}.{sp.__name__} --category phmap')
    debugger.HandleCommand('type category enable phmap')