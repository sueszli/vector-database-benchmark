import warnings
from numba.core import errors, types, typing, funcdesc, config, pylowering, transforms
from numba.core.compiler_machinery import FunctionPass, LoweringPass, register_pass
from collections import defaultdict

@register_pass(mutates_CFG=True, analysis_only=False)
class ObjectModeFrontEnd(FunctionPass):
    _name = 'object_mode_front_end'

    def __init__(self):
        if False:
            print('Hello World!')
        FunctionPass.__init__(self)

    def _frontend_looplift(self, state):
        if False:
            return 10
        '\n        Loop lifting analysis and transformation\n        '
        loop_flags = state.flags.copy()
        outer_flags = state.flags.copy()
        outer_flags.enable_looplift = False
        loop_flags.enable_looplift = False
        if not state.flags.enable_pyobject_looplift:
            loop_flags.enable_pyobject = False
        loop_flags.enable_ssa = False
        (main, loops) = transforms.loop_lifting(state.func_ir, typingctx=state.typingctx, targetctx=state.targetctx, locals=state.locals, flags=loop_flags)
        if loops:
            if config.DEBUG_FRONTEND or config.DEBUG:
                for loop in loops:
                    print('Lifting loop', loop.get_source_location())
            from numba.core.compiler import compile_ir
            cres = compile_ir(state.typingctx, state.targetctx, main, state.args, state.return_type, outer_flags, state.locals, lifted=tuple(loops), lifted_from=None, is_lifted_loop=True)
            return cres

    def run_pass(self, state):
        if False:
            print('Hello World!')
        from numba.core.compiler import _EarlyPipelineCompletion
        if state.flags.enable_looplift:
            assert not state.lifted
            cres = self._frontend_looplift(state)
            if cres is not None:
                raise _EarlyPipelineCompletion(cres)
        state.typemap = defaultdict(lambda : types.pyobject)
        state.calltypes = defaultdict(lambda : types.pyobject)
        state.return_type = types.pyobject
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class ObjectModeBackEnd(LoweringPass):
    _name = 'object_mode_back_end'

    def __init__(self):
        if False:
            while True:
                i = 10
        LoweringPass.__init__(self)

    def _py_lowering_stage(self, targetctx, library, interp, flags):
        if False:
            i = 10
            return i + 15
        fndesc = funcdesc.PythonFunctionDescriptor.from_object_mode_function(interp)
        with targetctx.push_code_library(library):
            lower = pylowering.PyLower(targetctx, library, fndesc, interp)
            lower.lower()
            if not flags.no_cpython_wrapper:
                lower.create_cpython_wrapper()
            env = lower.env
            call_helper = lower.call_helper
            del lower
        from numba.core.compiler import _LowerResult
        if flags.no_compile:
            return _LowerResult(fndesc, call_helper, cfunc=None, env=env)
        else:
            cfunc = targetctx.get_executable(library, fndesc, env)
            return _LowerResult(fndesc, call_helper, cfunc=cfunc, env=env)

    def run_pass(self, state):
        if False:
            for i in range(10):
                print('nop')
        '\n        Lowering for object mode\n        '
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            state.library.enable_object_caching()

        def backend_object_mode():
            if False:
                while True:
                    i = 10
            '\n            Object mode compilation\n            '
            if len(state.args) != state.nargs:
                state.args = tuple(state.args) + (types.pyobject,) * (state.nargs - len(state.args))
            return self._py_lowering_stage(state.targetctx, state.library, state.func_ir, state.flags)
        lowered = backend_object_mode()
        signature = typing.signature(state.return_type, *state.args)
        from numba.core.compiler import compile_result
        state.cr = compile_result(typing_context=state.typingctx, target_context=state.targetctx, entry_point=lowered.cfunc, typing_error=state.status.fail_reason, type_annotation=state.type_annotation, library=state.library, call_helper=lowered.call_helper, signature=signature, objectmode=True, lifted=state.lifted, fndesc=lowered.fndesc, environment=lowered.env, metadata=state.metadata, reload_init=state.reload_init)
        if not state.flags.force_pyobject:
            if len(state.lifted) > 0:
                warn_msg = 'Function "%s" was compiled in object mode without forceobj=True, but has lifted loops.' % (state.func_id.func_name,)
            else:
                warn_msg = 'Function "%s" was compiled in object mode without forceobj=True.' % (state.func_id.func_name,)
            warnings.warn(errors.NumbaWarning(warn_msg, state.func_ir.loc))
            url = 'https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit'
            msg = '\nFall-back from the nopython compilation path to the object mode compilation path has been detected. This is deprecated behaviour that will be removed in Numba 0.59.0.\n\nFor more information visit %s' % url
            warnings.warn(errors.NumbaDeprecationWarning(msg, state.func_ir.loc))
            if state.flags.release_gil:
                warn_msg = "Code running in object mode won't allow parallel execution despite nogil=True."
                warnings.warn_explicit(warn_msg, errors.NumbaWarning, state.func_id.filename, state.func_id.firstlineno)
        return True