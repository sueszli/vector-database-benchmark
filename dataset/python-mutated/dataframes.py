from typing import Any, Dict, List
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe
from torch.utils.data.datapipes.dataframe.structures import DataChunkDF
__all__ = ['Capture', 'CaptureA', 'CaptureAdd', 'CaptureCall', 'CaptureControl', 'CaptureDataFrame', 'CaptureDataFrameWithDataPipeOps', 'CaptureF', 'CaptureGetAttr', 'CaptureGetItem', 'CaptureInitial', 'CaptureLikeMock', 'CaptureMul', 'CaptureSetItem', 'CaptureSub', 'CaptureVariable', 'CaptureVariableAssign', 'DataFrameTracer', 'DataFrameTracedOps', 'disable_capture', 'get_val']

def disable_capture():
    if False:
        print('Hello World!')
    CaptureControl.disabled = True

class CaptureControl:
    disabled = False

class DataFrameTracedOps(DFIterDataPipe):

    def __init__(self, source_datapipe, output_var):
        if False:
            i = 10
            return i + 15
        self.source_datapipe = source_datapipe
        self.output_var = output_var

    def __iter__(self):
        if False:
            print('Hello World!')
        for item in self.source_datapipe:
            yield self.output_var.apply_ops(item)
DATAPIPES_OPS = ['_dataframes_as_tuples', 'groupby', '_dataframes_filter', 'map', 'to_datapipe', 'shuffle', 'concat', 'batch', '_dataframes_per_row', '_dataframes_concat', '_dataframes_shuffle']
UNIMPLEMENTED_ATTR = ['__deepcopy__', '__setstate__', 'is_shardable', 'apply_sharding']

class Capture:

    def __init__(self, schema_df=None):
        if False:
            i = 10
            return i + 15
        self.ctx = {'operations': [], 'variables': [], 'schema_df': schema_df}

    def __str__(self):
        if False:
            while True:
                i = 10
        return self._ops_str()

    def _ops_str(self):
        if False:
            return 10
        res = ''
        for op in self.ctx['operations']:
            if len(res) > 0:
                res += '\n'
            res += str(op)
        return res

    def __getstate__(self):
        if False:
            while True:
                i = 10
        self.ctx['schema_df'] = None
        for var in self.ctx['variables']:
            var.calculated_value = None
        state = {}
        for item in self.__dict__:
            state[item] = getattr(self, item)
        return state

    def __setstate__(self, state):
        if False:
            return 10
        for (k, v) in state.items():
            setattr(self, k, v)

    def __getattr__(self, attrname):
        if False:
            return 10
        if attrname == 'kwarg' or attrname == 'kwargs':
            raise Exception('no kwargs!')
        if attrname in ['__deepcopy__']:
            raise AttributeError()
        result = CaptureGetAttr(self, attrname, ctx=self.ctx)
        return result

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return CaptureGetItem(self, key, ctx=self.ctx)

    def __setitem__(self, key, value):
        if False:
            return 10
        self.ctx['operations'].append(CaptureSetItem(self, key, value, ctx=self.ctx))

    def __add__(self, add_val):
        if False:
            while True:
                i = 10
        res = CaptureAdd(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx['operations'].append(CaptureVariableAssign(variable=var, value=res, ctx=self.ctx))
        return var

    def __sub__(self, add_val):
        if False:
            return 10
        res = CaptureSub(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        self.ctx['operations'].append(CaptureVariableAssign(variable=var, value=res, ctx=self.ctx))
        return var

    def __mul__(self, add_val):
        if False:
            i = 10
            return i + 15
        res = CaptureMul(self, add_val, ctx=self.ctx)
        var = CaptureVariable(res, ctx=self.ctx)
        t = CaptureVariableAssign(variable=var, value=res, ctx=self.ctx)
        self.ctx['operations'].append(t)
        return var

    def _is_context_empty(self):
        if False:
            return 10
        return len(self.ctx['operations']) == 0 and len(self.ctx['variables']) == 0

    def apply_ops_2(self, dataframe):
        if False:
            print('Hello World!')
        self.ctx['variables'][0].calculated_value = dataframe
        for op in self.ctx['operations']:
            op.execute()

    @property
    def columns(self):
        if False:
            print('Hello World!')
        self.apply_ops_2(self.ctx['schema_df'])
        value = self.execute()
        return value.columns

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self._is_context_empty():
            for arg in args:
                if isinstance(arg, Capture) and (not arg._is_context_empty()):
                    self.ctx = arg.ctx
                    break
            if self._is_context_empty():
                for (k, v) in kwargs.items():
                    if isinstance(k, Capture) and (not k._is_context_empty()):
                        self.ctx = k.ctx
                        break
                    if isinstance(v, Capture) and (not v._is_context_empty()):
                        self.ctx = v.ctx
                        break
        res = CaptureCall(self, ctx=self.ctx, args=args, kwargs=kwargs)
        var = CaptureVariable(None, ctx=self.ctx)
        t = CaptureVariableAssign(ctx=self.ctx, variable=var, value=res)
        self.ctx['operations'].append(t)
        return var

class CaptureF(Capture):

    def __init__(self, ctx=None, **kwargs):
        if False:
            while True:
                i = 10
        if ctx is None:
            self.ctx = {'operations': [], 'variables': []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs

class CaptureA(CaptureF):

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return f"{self.kwargs['name']}"

    def execute(self):
        if False:
            while True:
                i = 10
        value = self.kwargs['real_attribute']
        return value

class CaptureLikeMock:

    def __init__(self, name):
        if False:
            print('Hello World!')
        import unittest.mock as mock
        (get_target, attribute) = mock._get_target(name)
        self.get_target = get_target
        self.attribute = attribute
        self.name = name

    def __enter__(self):
        if False:
            return 10
        self.save = getattr(self.get_target(), self.attribute)
        capt = CaptureA(name=self.name, real_attribute=self.save)
        setattr(self.get_target(), self.attribute, capt)

    def __exit__(self, *exc_info):
        if False:
            i = 10
            return i + 15
        setattr(self.get_target(), self.attribute, self.save)

class CaptureCall(Capture):

    def __init__(self, callable, ctx=None, **kwargs):
        if False:
            print('Hello World!')
        if ctx is None:
            self.ctx = {'operations': [], 'variables': []}
        else:
            self.ctx = ctx
        self.kwargs = kwargs
        self.callable = callable

    def __str__(self):
        if False:
            return 10
        return '{callable}({args},{kwargs})'.format(callable=self.callable, **self.kwargs)

    def execute(self):
        if False:
            while True:
                i = 10
        executed_args = []
        for arg in self.kwargs['args']:
            if isinstance(arg, Capture):
                executed_args.append(arg.execute())
            else:
                executed_args.append(arg)
        left = get_val(self.callable)
        return left(*executed_args, **self.kwargs['kwargs'])

class CaptureVariableAssign(CaptureF):

    def __str__(self):
        if False:
            while True:
                i = 10
        variable = self.kwargs['variable']
        value = self.kwargs['value']
        return f'{variable} = {value}'

    def execute(self):
        if False:
            for i in range(10):
                print('nop')
        self.kwargs['variable'].calculated_value = self.kwargs['value'].execute()

class CaptureVariable(Capture):
    names_idx = 0

    def __init__(self, value, ctx):
        if False:
            i = 10
            return i + 15
        if CaptureControl.disabled:
            raise Exception('Attempting to create capture variable with capture off')
        self.ctx = ctx
        self.value = value
        self.name = f'var_{CaptureVariable.names_idx}'
        CaptureVariable.names_idx += 1
        self.ctx['variables'].append(self)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.name

    def execute(self):
        if False:
            print('Hello World!')
        return self.calculated_value

    def apply_ops(self, dataframe):
        if False:
            print('Hello World!')
        self.ctx['variables'][0].calculated_value = dataframe
        for op in self.ctx['operations']:
            op.execute()
        return self.calculated_value

class CaptureGetItem(Capture):

    def __init__(self, left, key, ctx):
        if False:
            while True:
                i = 10
        self.ctx = ctx
        self.left = left
        self.key = key

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'{self.left}[{get_val(self.key)}]'

    def execute(self):
        if False:
            return 10
        left = self.left.execute()
        return left[self.key]

class CaptureSetItem(Capture):

    def __init__(self, left, key, value, ctx):
        if False:
            while True:
                i = 10
        self.ctx = ctx
        self.left = left
        self.key = key
        self.value = value

    def __str__(self):
        if False:
            return 10
        return f'{self.left}[{get_val(self.key)}] = {self.value}'

    def execute(self):
        if False:
            print('Hello World!')
        left = self.left.execute()
        value = self.value.execute()
        left[self.key] = value

class CaptureAdd(Capture):

    def __init__(self, left, right, ctx):
        if False:
            while True:
                i = 10
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self):
        if False:
            return 10
        return f'{self.left} + {self.right}'

    def execute(self):
        if False:
            for i in range(10):
                print('nop')
        return get_val(self.left) + get_val(self.right)

class CaptureMul(Capture):

    def __init__(self, left, right, ctx):
        if False:
            return 10
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self):
        if False:
            return 10
        return f'{self.left} * {self.right}'

    def execute(self):
        if False:
            return 10
        return get_val(self.left) * get_val(self.right)

class CaptureSub(Capture):

    def __init__(self, left, right, ctx):
        if False:
            while True:
                i = 10
        self.ctx = ctx
        self.left = left
        self.right = right

    def __str__(self):
        if False:
            print('Hello World!')
        return f'{self.left} - {self.right}'

    def execute(self):
        if False:
            for i in range(10):
                print('nop')
        return get_val(self.left) - get_val(self.right)

class CaptureGetAttr(Capture):

    def __init__(self, src, name, ctx):
        if False:
            return 10
        self.ctx = ctx
        self.src = src
        self.name = name

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return f'{self.src}.{self.name}'

    def execute(self):
        if False:
            for i in range(10):
                print('nop')
        val = get_val(self.src)
        return getattr(val, self.name)

def get_val(capture):
    if False:
        return 10
    if isinstance(capture, Capture):
        return capture.execute()
    elif isinstance(capture, str):
        return f'"{capture}"'
    else:
        return capture

class CaptureInitial(CaptureVariable):

    def __init__(self, schema_df=None):
        if False:
            print('Hello World!')
        new_ctx: Dict[str, List[Any]] = {'operations': [], 'variables': [], 'schema_df': schema_df}
        super().__init__(None, new_ctx)
        self.name = f'input_{self.name}'

class CaptureDataFrame(CaptureInitial):
    pass

class CaptureDataFrameWithDataPipeOps(CaptureDataFrame):

    def as_datapipe(self):
        if False:
            i = 10
            return i + 15
        return DataFrameTracedOps(self.ctx['variables'][0].source_datapipe, self)

    def raw_iterator(self):
        if False:
            i = 10
            return i + 15
        return self.as_datapipe().__iter__()

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._dataframes_as_tuples())

    def batch(self, batch_size=10, drop_last: bool=False, wrapper_class=DataChunkDF):
        if False:
            for i in range(10):
                print('nop')
        dp = self._dataframes_per_row()._dataframes_concat(batch_size)
        dp = dp.as_datapipe().batch(1, drop_last=drop_last, wrapper_class=wrapper_class)
        dp._dp_contains_dataframe = True
        return dp

    def groupby(self, group_key_fn, *, buffer_size=10000, group_size=None, guaranteed_group_size=None, drop_remaining=False):
        if False:
            print('Hello World!')
        dp = self._dataframes_per_row()
        dp = dp.as_datapipe().groupby(group_key_fn, buffer_size=buffer_size, group_size=group_size, guaranteed_group_size=guaranteed_group_size, drop_remaining=drop_remaining)
        return dp

    def shuffle(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self._dataframes_shuffle(*args, **kwargs)

    def filter(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._dataframes_filter(*args, **kwargs)

    def collate(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise Exception("Can't collate unbatched DataFrames stream")

    def __getattr__(self, attrname):
        if False:
            return 10
        if attrname in UNIMPLEMENTED_ATTR:
            raise AttributeError('Attempting to get ', attrname)
        if attrname in DATAPIPES_OPS:
            return self.as_datapipe().__getattr__(attrname)
        return super().__getattr__(attrname)

@functional_datapipe('trace_as_dataframe')
class DataFrameTracer(CaptureDataFrameWithDataPipeOps, IterDataPipe):
    source_datapipe = None

    def set_shuffle_settings(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

    def is_shardable(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def __init__(self, source_datapipe, schema_df=None):
        if False:
            return 10
        self.source_datapipe = source_datapipe
        if schema_df is None:
            schema_df = next(iter(self.source_datapipe))
        super().__init__(schema_df=schema_df)