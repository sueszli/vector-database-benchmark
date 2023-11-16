import string
import numpy
from cupy._core import _codeblock
from cupy._core._fusion_variable import _TraceVariable
from cupy._core._fusion_variable import _TraceArray
from cupy._core._fusion_variable import _VariableSet
from cupy._core import _fusion_thread_local
from cupy._core import _kernel
from cupy._core import _reduction
from cupy._core._scalar import get_typename

class _UfuncRoutine:
    """A device function for single elementwise operations.
    """

    def __init__(self, name, ufunc, routine_code, in_params, out_params, compute_dtypes):
        if False:
            print('Hello World!')
        assert isinstance(name, str)
        assert isinstance(ufunc, _kernel.ufunc)
        assert isinstance(routine_code, str)
        assert isinstance(compute_dtypes, tuple)
        assert all((isinstance(t, numpy.dtype) for t in compute_dtypes))
        assert isinstance(in_params, list)
        assert all((isinstance(p, _TraceVariable) for p in in_params))
        assert isinstance(out_params, list)
        assert all((isinstance(p, _TraceArray) for p in out_params))
        self.name = name
        self.in_params = in_params
        self.out_params = out_params
        self.preamble = ufunc._preamble
        self.routine_code = routine_code
        self.compute_dtypes = compute_dtypes

    def emit_code(self):
        if False:
            return 10
        'Returns a CUDA device function code.\n\n        Returns a string like:\n        ```\n        __device__ void cupy_add_0(int &in0_, float &in1_, double &out0_) {\n            typedef double in0_type;\n            typedef double in1_type;\n            typedef double out0_type;\n            double in0 = (double) in0_;\n            double in1 = (double) in1_;\n            double out0 = (double) out0_;\n            out0 = in0 + in1;\n            out0_ = out0;\n        }\n        ```\n        '
        nin = len(self.in_params)
        dtypes = self.compute_dtypes
        assert len(self.in_params) == len(self.compute_dtypes[:nin])
        in_params = [(get_typename(p.dtype), get_typename(t), 'in{}'.format(i)) for (i, (p, t)) in enumerate(zip(self.in_params, dtypes[:nin]))]
        out_params = [(get_typename(p.dtype), get_typename(t), 'out{}'.format(i)) for (i, (p, t)) in enumerate(zip(self.out_params, dtypes[nin:]))]
        params = in_params + out_params
        params_code = ', '.join(['{} &{}_'.format(t, s) for (t, _, s) in params])
        typedef = ['typedef {} {}_type;'.format(t, s) for (_, t, s) in params]
        read = ['{} {} = ({}) {}_;'.format(t, s, t, s) for (_, t, s) in params]
        write = ['{}_ = {};'.format(s, s) for (_, _, s) in out_params]
        return _codeblock.CodeBlock('__device__ void {}({})'.format(self.name, params_code), typedef + read + [self.routine_code + ';'] + write)

    def emit_call_code(self):
        if False:
            i = 10
            return i + 15
        params = self.in_params + self.out_params
        return '{op_name}({params});'.format(op_name=self.name, params=', '.join([var.lvar_name for var in params]))

class _ElementwiseTraceOp:
    """Ufunc or elementwise kernel with types.
    """

    def __init__(self, ufunc_routines, in_params, out_params, ashape):
        if False:
            return 10
        _fusion_thread_local.check_not_runtime()
        assert isinstance(ufunc_routines, list)
        assert all((isinstance(r, _UfuncRoutine) for r in ufunc_routines))
        assert isinstance(ashape, tuple)
        self.ops = ufunc_routines
        self.in_params = _VariableSet(*in_params)
        self.out_params = _VariableSet(*out_params)
        self.ashape = ashape

    @property
    def params(self):
        if False:
            while True:
                i = 10
        'Returns the set of all variable the loop uses.\n        '
        res = _VariableSet()
        for op in self.ops:
            res += _VariableSet(*op.in_params)
            res += _VariableSet(*op.out_params)
        return res

    @staticmethod
    def _emit_declaration(params, in_params):
        if False:
            return 10
        'Returns a tuple of size 2.\n\n        1. CUDA code: declaring local variables.\n            2. The set of arrays which require indexer.\n        '
        _fusion_thread_local.check_not_runtime()
        indexed_arrays = _VariableSet()
        code = []
        for var in params:
            if var in in_params:
                if isinstance(var, _TraceArray):
                    indexed_arrays.add(var)
                    f = '${type} ${lvar} = ${var}[${indexer}.get()];'
                else:
                    f = '${type} ${lvar} = ${var};'
            else:
                f = '${type} ${lvar};'
            code.append(var.format(f))
        return (code, indexed_arrays)

    @staticmethod
    def _emit_after_operation(out_params):
        if False:
            print('Hello World!')
        'Returns a tuple of size 2.\n        1. CUDA code: writing the results of operations back to global memory.\n        2. The set of arrays which require indexer.\n        '
        _fusion_thread_local.check_not_runtime()
        indexed_arrays = _VariableSet()
        codes = []
        for var in out_params:
            if isinstance(var, _TraceArray):
                indexed_arrays.add(var)
                f = '${var}[${indexer}.get()] = ${lvar};'
            else:
                f = '${var} = ${lvar};'
            codes.append(var.format(f))
        return (codes, indexed_arrays)

    @staticmethod
    def _emit_set_index(indexed_params, tid):
        if False:
            return 10
        'Returns a CUDA code: setting a raw index to indexers.\n        '
        _fusion_thread_local.check_not_runtime()
        assert isinstance(indexed_params, _VariableSet)
        return [p.format('${indexer}.set(${tid});', tid=tid) for p in indexed_params]

    def emit_code(self):
        if False:
            for i in range(10):
                print('nop')
        _fusion_thread_local.check_not_runtime()
        (declaration, s1) = self._emit_declaration(self.params, self.in_params)
        operation = [op.emit_call_code() for op in self.ops]
        (after_operation, s2) = self._emit_after_operation(self.out_params)
        index_name = 'i'
        indexed_array = s1 + s2
        indexer_name = next(iter(indexed_array)).indexer_name
        indexer_setup = self._emit_set_index(indexed_array, index_name)
        return _codeblock.CodeBlock('CUPY_FOR({}, {}.size())'.format(index_name, indexer_name), indexer_setup + declaration + operation + after_operation)

    def emit_preamble_codes(self):
        if False:
            while True:
                i = 10
        return [subm.preamble for subm in self.ops if subm.preamble != '']

    def emit_submodule_codes(self):
        if False:
            return 10
        return [str(subm.emit_code()) for subm in self.ops]

class _ReductionTraceOp:

    def __init__(self, name, reduce_func, expr, in_param, out_param, axis):
        if False:
            while True:
                i = 10
        'Reduction operation.\n        '
        _fusion_thread_local.check_not_runtime()
        assert isinstance(name, str)
        assert isinstance(reduce_func, _reduction._SimpleReductionKernel)
        assert isinstance(in_param, _TraceArray)
        assert isinstance(out_param, _TraceArray)
        assert isinstance(axis, tuple)
        assert all((0 <= x < in_param.ndim for x in axis))
        self.name = name
        self.preamble = reduce_func.preamble
        self.in_params = _VariableSet(in_param)
        self.out_params = _VariableSet(out_param)
        self.block_stride_name = 'block_stride_' + name
        self.axis = axis
        if reduce_func.identity is None:
            self.identity = ''
        else:
            self.identity = str(reduce_func.identity)
        (_, self.expr, self.postmap_cast_code, self.reduce_ctype) = expr
        if self.reduce_ctype is None:
            (out_param,) = self.out_params
            self.reduce_ctype = get_typename(out_param.dtype)
        self.premap_op = None
        self.postmap_op = None

    @property
    def params(self):
        if False:
            return 10
        return self.in_params + self.out_params

    def emit_code(self):
        if False:
            print('Hello World!')
        _fusion_thread_local.check_not_runtime()
        assert len(self.in_params) == 1
        assert len(self.out_params) == 1
        in_param = list(self.in_params)[0]
        out_param = list(self.out_params)[0]
        params = ', '.join([in_param.var_name, out_param.var_name, in_param.indexer_name, out_param.indexer_name])
        return '{}({}, {});'.format(self.name, params, self.block_stride_name)

    def emit_preamble_codes(self):
        if False:
            print('Hello World!')
        preamble = self.preamble
        return [preamble] if preamble != '' else []

    def emit_submodule_codes(self):
        if False:
            i = 10
            return i + 15
        'Returns a CUDA device function code.\n\n        The emitted code assumes that ``block_stride`` and `blockDim.x` is a\n        power of 2.\n        '
        (in_param,) = self.in_params
        (out_param,) = self.out_params
        op_name = '{}_op'.format(self.name)
        postmap_name = '{}_postmap'.format(self.name)
        template = string.Template('\n#define ${op_name}(a, b) (${reduce_expr})\n#define ${postmap_name}(a, out0) (${postmap_cast})\n\ntemplate <typename InType, typename OutType, typename InIndexerType, typename OutIndexerType>\n__device__ void ${name}(\n        InType in_arr, OutType out_arr,\n        InIndexerType in_ind, OutIndexerType out_ind, int block_stride) {\n    typedef ${in_type} type_in0_raw;\n    typedef ${out_type} type_out0_raw;\n    typedef ${reduce_ctype} _type_reduce;\n    extern __shared__ char _sdata_raw[];\n    _type_reduce *sdata = reinterpret_cast<_type_reduce*>(_sdata_raw);\n    unsigned int tid = threadIdx.x;\n    int _J = tid >> __popc(block_stride - 1);\n    ptrdiff_t _j = (ptrdiff_t)_J * out_ind.size();\n    int J_stride = blockDim.x >> __popc(block_stride - 1);\n    ptrdiff_t j_stride = (ptrdiff_t)J_stride * out_ind.size();\n\n    for (ptrdiff_t _i = (ptrdiff_t)blockIdx.x * block_stride; _i < out_ind.size(); _i += (ptrdiff_t)gridDim.x * block_stride) {\n        _type_reduce s = _type_reduce(${identity});\n        ptrdiff_t i = _i + (tid & (block_stride - 1));\n        for (ptrdiff_t j = i + _j; j < in_ind.size(); j += j_stride) {\n            in_ind.set(j);\n            s = ${op_name}(s, static_cast<_type_reduce>(in_arr[in_ind.get()]));\n        }\n        sdata[tid] = s;\n        __syncthreads();\n        for (unsigned int block = blockDim.x / 2; block >= block_stride; block >>= 1) {\n            if (tid < block) {\n                sdata[tid] = ${op_name}(sdata[tid], sdata[tid + block]);\n            }\n            __syncthreads();\n        }\n        if (tid < block_stride) {\n            s = sdata[tid];\n        }\n        if (tid < block_stride && i < out_ind.size()) {\n            out_ind.set(i);\n            ${postmap_name}(s, out_arr[out_ind.get()]);\n        }\n        __syncthreads();\n    }\n}')
        code = template.substitute(name=self.name, op_name=op_name, postmap_name=postmap_name, in_type=get_typename(in_param.dtype), out_type=get_typename(out_param.dtype), reduce_ctype=self.reduce_ctype, reduce_expr=self.expr, identity=self.identity, postmap_cast=self.postmap_cast_code)
        return [code]