def ex_compiler_pass():
    if False:
        print('Hello World!')
    from numba import njit
    from numba.core import ir
    from numba.core.compiler import CompilerBase, DefaultPassBuilder
    from numba.core.compiler_machinery import FunctionPass, register_pass
    from numba.core.untyped_passes import IRProcessing
    from numbers import Number

    @register_pass(mutates_CFG=False, analysis_only=False)
    class ConstsAddOne(FunctionPass):
        _name = 'consts_add_one'

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            FunctionPass.__init__(self)

        def run_pass(self, state):
            if False:
                return 10
            func_ir = state.func_ir
            mutated = False
            for blk in func_ir.blocks.values():
                for assgn in blk.find_insts(ir.Assign):
                    if isinstance(assgn.value, ir.Const):
                        const_val = assgn.value
                        if isinstance(const_val.value, Number):
                            const_val.value += 1
                            mutated |= True
            return mutated

    class MyCompiler(CompilerBase):

        def define_pipelines(self):
            if False:
                while True:
                    i = 10
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pm.add_pass_after(ConstsAddOne, IRProcessing)
            pm.finalize()
            return [pm]

    @njit(pipeline_class=MyCompiler)
    def foo(x):
        if False:
            i = 10
            return i + 15
        a = 10
        b = 20.2
        c = x + a + b
        return c
    print(foo(100))
    compile_result = foo.overloads[foo.signatures[0]]
    nopython_times = compile_result.metadata['pipeline_times']['nopython']
    for k in nopython_times.keys():
        if ConstsAddOne._name in k:
            print(nopython_times[k])
    assert foo(100) == 132.2
ex_compiler_pass()