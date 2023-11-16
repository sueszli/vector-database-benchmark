import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from copy import copy
import warnings
from numba.core import errors, types, typing, ir, funcdesc, rewrites, typeinfer, config, lowering
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as _parfor_ParforPreLoweringPass
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import FunctionPass, LoweringPass, AnalysisPass, register_pass
from numba.core.annotations import type_annotations
from numba.core.ir_utils import raise_on_unsupported_feature, warn_deprecated, check_and_legalize_ir, guard, dead_code_elimination, simplify_CFG, get_definition, build_definitions, compute_cfg_from_blocks, is_operator_or_getitem
from numba.core import postproc
from llvmlite import binding as llvm
_TypingResults = namedtuple('_TypingResults', ['typemap', 'return_type', 'calltypes', 'typing_errors'])

@contextmanager
def fallback_context(state, msg):
    if False:
        return 10
    '\n    Wraps code that would signal a fallback to object mode\n    '
    try:
        yield
    except Exception as e:
        if not state.status.can_fallback:
            raise
        else:
            e = e.with_traceback(None)
            loop_lift = '' if state.flags.enable_looplift else 'OUT'
            msg_rewrite = '\nCompilation is falling back to object mode WITH%s looplifting enabled because %s' % (loop_lift, msg)
            warnings.warn_explicit('%s due to: %s' % (msg_rewrite, e), errors.NumbaWarning, state.func_id.filename, state.func_id.firstlineno)
            raise

def type_inference_stage(typingctx, targetctx, interp, args, return_type, locals={}, raise_errors=True):
    if False:
        return 10
    if len(args) != interp.arg_count:
        raise TypeError('Mismatch number of argument types')
    warnings = errors.WarningsFixer(errors.NumbaWarning)
    infer = typeinfer.TypeInferer(typingctx, interp, warnings)
    callstack_ctx = typingctx.callstack.register(targetctx.target, infer, interp.func_id, args)
    with callstack_ctx, warnings:
        for (index, (name, ty)) in enumerate(zip(interp.arg_names, args)):
            infer.seed_argument(name, index, ty)
        if return_type is not None:
            infer.seed_return(return_type)
        for (k, v) in locals.items():
            infer.seed_type(k, v)
        infer.build_constraint()
        errs = infer.propagate(raise_errors=raise_errors)
        (typemap, restype, calltypes) = infer.unify(raise_errors=raise_errors)
    return _TypingResults(typemap, restype, calltypes, errs)

class BaseTypeInference(FunctionPass):
    _raise_errors = True

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if False:
            return 10
        '\n        Type inference and legalization\n        '
        with fallback_context(state, 'Function "%s" failed type inference' % (state.func_id.func_name,)):
            (typemap, return_type, calltypes, errs) = type_inference_stage(state.typingctx, state.targetctx, state.func_ir, state.args, state.return_type, state.locals, raise_errors=self._raise_errors)
            state.typemap = typemap
            state.typing_errors = errs
            if self._raise_errors:
                state.return_type = return_type
            state.calltypes = calltypes

        def legalize_return_type(return_type, interp, targetctx):
            if False:
                return 10
            '\n            Only accept array return type iff it is passed into the function.\n            Reject function object return types if in nopython mode.\n            '
            if not targetctx.enable_nrt and isinstance(return_type, types.Array):
                retstmts = []
                caststmts = {}
                argvars = set()
                for (bid, blk) in interp.blocks.items():
                    for inst in blk.body:
                        if isinstance(inst, ir.Return):
                            retstmts.append(inst.value.name)
                        elif isinstance(inst, ir.Assign):
                            if isinstance(inst.value, ir.Expr) and inst.value.op == 'cast':
                                caststmts[inst.target.name] = inst.value
                            elif isinstance(inst.value, ir.Arg):
                                argvars.add(inst.target.name)
                assert retstmts, 'No return statements?'
                for var in retstmts:
                    cast = caststmts.get(var)
                    if cast is None or cast.value.name not in argvars:
                        if self._raise_errors:
                            msg = 'Only accept returning of array passed into the function as argument'
                            raise errors.NumbaTypeError(msg)
            elif isinstance(return_type, types.Function) or isinstance(return_type, types.Phantom):
                if self._raise_errors:
                    msg = "Can't return function object ({}) in nopython mode"
                    raise errors.NumbaTypeError(msg.format(return_type))
        with fallback_context(state, 'Function "%s" has invalid return type' % (state.func_id.func_name,)):
            legalize_return_type(state.return_type, state.func_ir, state.targetctx)
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class NopythonTypeInference(BaseTypeInference):
    _name = 'nopython_type_inference'

@register_pass(mutates_CFG=True, analysis_only=False)
class PartialTypeInference(BaseTypeInference):
    _name = 'partial_type_inference'
    _raise_errors = False

@register_pass(mutates_CFG=False, analysis_only=False)
class AnnotateTypes(AnalysisPass):
    _name = 'annotate_types'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        AnalysisPass.__init__(self)

    def get_analysis_usage(self, AU):
        if False:
            return 10
        AU.add_required(IRLegalization)

    def run_pass(self, state):
        if False:
            i = 10
            return i + 15
        '\n        Create type annotation after type inference\n        '
        func_ir = state.func_ir.copy()
        state.type_annotation = type_annotations.TypeAnnotation(func_ir=func_ir, typemap=state.typemap, calltypes=state.calltypes, lifted=state.lifted, lifted_from=state.lifted_from, args=state.args, return_type=state.return_type, html_output=config.HTML)
        if config.ANNOTATE:
            print('ANNOTATION'.center(80, '-'))
            print(state.type_annotation)
            print('=' * 80)
        if config.HTML:
            with open(config.HTML, 'w') as fout:
                state.type_annotation.html_annotate(fout)
        return False

@register_pass(mutates_CFG=True, analysis_only=False)
class NopythonRewrites(FunctionPass):
    _name = 'nopython_rewrites'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if False:
            while True:
                i = 10
        '\n        Perform any intermediate representation rewrites after type\n        inference.\n        '
        assert state.func_ir
        assert isinstance(getattr(state, 'typemap', None), dict)
        assert isinstance(getattr(state, 'calltypes', None), dict)
        msg = 'Internal error in post-inference rewriting pass encountered during compilation of function "%s"' % (state.func_id.func_name,)
        pp = postproc.PostProcessor(state.func_ir)
        pp.run(True)
        with fallback_context(state, msg):
            rewrites.rewrite_registry.apply('after-inference', state)
        pp.remove_dels()
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class PreParforPass(FunctionPass):
    _name = 'pre_parfor_pass'

    def __init__(self):
        if False:
            print('Hello World!')
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if False:
            i = 10
            return i + 15
        '\n        Preprocessing for data-parallel computations.\n        '
        assert state.func_ir
        preparfor_pass = _parfor_PreParforPass(state.func_ir, state.typemap, state.calltypes, state.typingctx, state.targetctx, state.flags.auto_parallel, state.parfor_diagnostics.replaced_fns)
        preparfor_pass.run()
        return True

def _reload_parfors():
    if False:
        print('Hello World!')
    'Reloader for cached parfors\n    '
    from numba.np.ufunc.parallel import _launch_threads
    _launch_threads()

@register_pass(mutates_CFG=True, analysis_only=False)
class ParforPass(FunctionPass):
    _name = 'parfor_pass'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if False:
            print('Hello World!')
        '\n        Convert data-parallel computations into Parfor nodes\n        '
        assert state.func_ir
        parfor_pass = _parfor_ParforPass(state.func_ir, state.typemap, state.calltypes, state.return_type, state.typingctx, state.targetctx, state.flags.auto_parallel, state.flags, state.metadata, state.parfor_diagnostics)
        parfor_pass.run()
        has_parfor = False
        for blk in state.func_ir.blocks.values():
            for stmnt in blk.body:
                if isinstance(stmnt, Parfor):
                    has_parfor = True
                    break
            else:
                continue
            break
        if not has_parfor:
            if not (config.DISABLE_PERFORMANCE_WARNINGS or state.func_ir.loc.filename == '<string>'):
                url = 'https://numba.readthedocs.io/en/stable/user/parallel.html#diagnostics'
                msg = "\nThe keyword argument 'parallel=True' was specified but no transformation for parallel execution was possible.\n\nTo find out why, try turning on parallel diagnostics, see %s for help." % url
                warnings.warn(errors.NumbaPerformanceWarning(msg, state.func_ir.loc))
        state.reload_init.append(_reload_parfors)
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class ParforFusionPass(FunctionPass):
    _name = 'parfor_fusion_pass'

    def __init__(self):
        if False:
            return 10
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if False:
            while True:
                i = 10
        '\n        Do fusion of parfor nodes.\n        '
        assert state.func_ir
        parfor_pass = _parfor_ParforFusionPass(state.func_ir, state.typemap, state.calltypes, state.return_type, state.typingctx, state.targetctx, state.flags.auto_parallel, state.flags, state.metadata, state.parfor_diagnostics)
        parfor_pass.run()
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class ParforPreLoweringPass(FunctionPass):
    _name = 'parfor_prelowering_pass'

    def __init__(self):
        if False:
            while True:
                i = 10
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if False:
            i = 10
            return i + 15
        '\n        Prepare parfors for lowering.\n        '
        assert state.func_ir
        parfor_pass = _parfor_ParforPreLoweringPass(state.func_ir, state.typemap, state.calltypes, state.return_type, state.typingctx, state.targetctx, state.flags.auto_parallel, state.flags, state.metadata, state.parfor_diagnostics)
        parfor_pass.run()
        return True

@register_pass(mutates_CFG=False, analysis_only=True)
class DumpParforDiagnostics(AnalysisPass):
    _name = 'dump_parfor_diagnostics'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        if False:
            while True:
                i = 10
        if state.flags.auto_parallel.enabled:
            if config.PARALLEL_DIAGNOSTICS:
                if state.parfor_diagnostics is not None:
                    state.parfor_diagnostics.dump(config.PARALLEL_DIAGNOSTICS)
                else:
                    raise RuntimeError('Diagnostics failed.')
        return True

class BaseNativeLowering(abc.ABC, LoweringPass):
    """The base class for a lowering pass. The lowering functionality must be
    specified in inheriting classes by providing an appropriate lowering class
    implementation in the overridden `lowering_class` property."""
    _name = None

    def __init__(self):
        if False:
            return 10
        LoweringPass.__init__(self)

    @property
    @abc.abstractmethod
    def lowering_class(self):
        if False:
            i = 10
            return i + 15
        'Returns the class that performs the lowering of the IR describing the\n        function that is the target of the current compilation.'
        pass

    def run_pass(self, state):
        if False:
            return 10
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            state.library.enable_object_caching()
        library = state.library
        targetctx = state.targetctx
        interp = state.func_ir
        typemap = state.typemap
        restype = state.return_type
        calltypes = state.calltypes
        flags = state.flags
        metadata = state.metadata
        pre_stats = llvm.passmanagers.dump_refprune_stats()
        msg = 'Function %s failed at nopython mode lowering' % (state.func_id.func_name,)
        with fallback_context(state, msg):
            fndesc = funcdesc.PythonFunctionDescriptor.from_specialized_function(interp, typemap, restype, calltypes, mangler=targetctx.mangler, inline=flags.forceinline, noalias=flags.noalias, abi_tags=[flags.get_mangle_string()])
            with targetctx.push_code_library(library):
                lower = self.lowering_class(targetctx, library, fndesc, interp, metadata=metadata)
                lower.lower()
                if not flags.no_cpython_wrapper:
                    lower.create_cpython_wrapper(flags.release_gil)
                if not flags.no_cfunc_wrapper:
                    for t in state.args:
                        if isinstance(t, (types.Omitted, types.Generator)):
                            break
                    else:
                        if isinstance(restype, (types.Optional, types.Generator)):
                            pass
                        else:
                            lower.create_cfunc_wrapper()
                env = lower.env
                call_helper = lower.call_helper
                del lower
            from numba.core.compiler import _LowerResult
            if flags.no_compile:
                state['cr'] = _LowerResult(fndesc, call_helper, cfunc=None, env=env)
            else:
                cfunc = targetctx.get_executable(library, fndesc, env)
                targetctx.insert_user_function(cfunc, fndesc, [library])
                state['cr'] = _LowerResult(fndesc, call_helper, cfunc=cfunc, env=env)
            post_stats = llvm.passmanagers.dump_refprune_stats()
            metadata['prune_stats'] = post_stats - pre_stats
            metadata['llvm_pass_timings'] = library.recorded_timings
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class NativeLowering(BaseNativeLowering):
    """Lowering pass for a native function IR described solely in terms of
     Numba's standard `numba.core.ir` nodes."""
    _name = 'native_lowering'

    @property
    def lowering_class(self):
        if False:
            i = 10
            return i + 15
        return lowering.Lower

@register_pass(mutates_CFG=True, analysis_only=False)
class NativeParforLowering(BaseNativeLowering):
    """Lowering pass for a native function IR described using Numba's standard
    `numba.core.ir` nodes and also parfor.Parfor nodes."""
    _name = 'native_parfor_lowering'

    @property
    def lowering_class(self):
        if False:
            i = 10
            return i + 15
        return ParforLower

@register_pass(mutates_CFG=False, analysis_only=True)
class NoPythonSupportedFeatureValidation(AnalysisPass):
    """NoPython Mode check: Validates the IR to ensure that features in use are
    in a form that is supported"""
    _name = 'nopython_supported_feature_validation'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        if False:
            i = 10
            return i + 15
        raise_on_unsupported_feature(state.func_ir, state.typemap)
        warn_deprecated(state.func_ir, state.typemap)
        return False

@register_pass(mutates_CFG=False, analysis_only=True)
class IRLegalization(AnalysisPass):
    _name = 'ir_legalization'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        if False:
            for i in range(10):
                print('nop')
        check_and_legalize_ir(state.func_ir, flags=state.flags)
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class NoPythonBackend(LoweringPass):
    _name = 'nopython_backend'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        LoweringPass.__init__(self)

    def run_pass(self, state):
        if False:
            i = 10
            return i + 15
        '\n        Back-end: Generate LLVM IR from Numba IR, compile to machine code\n        '
        lowered = state['cr']
        signature = typing.signature(state.return_type, *state.args)
        from numba.core.compiler import compile_result
        state.cr = compile_result(typing_context=state.typingctx, target_context=state.targetctx, entry_point=lowered.cfunc, typing_error=state.status.fail_reason, type_annotation=state.type_annotation, library=state.library, call_helper=lowered.call_helper, signature=signature, objectmode=False, lifted=state.lifted, fndesc=lowered.fndesc, environment=lowered.env, metadata=state.metadata, reload_init=state.reload_init)
        return True

@register_pass(mutates_CFG=True, analysis_only=False)
class InlineOverloads(FunctionPass):
    """
    This pass will inline a function wrapped by the numba.extending.overload
    decorator directly into the site of its call depending on the value set in
    the 'inline' kwarg to the decorator.

    This is a typed pass. CFG simplification and DCE are performed on
    completion.
    """
    _name = 'inline_overloads'

    def __init__(self):
        if False:
            print('Hello World!')
        FunctionPass.__init__(self)
    _DEBUG = False

    def run_pass(self, state):
        if False:
            return 10
        'Run inlining of overloads\n        '
        if self._DEBUG:
            print('before overload inline'.center(80, '-'))
            print(state.func_id.unique_name)
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        from numba.core.inline_closurecall import InlineWorker, callee_ir_validator
        inline_worker = InlineWorker(state.typingctx, state.targetctx, state.locals, state.pipeline, state.flags, callee_ir_validator, state.typemap, state.calltypes)
        modified = False
        work_list = list(state.func_ir.blocks.items())
        while work_list:
            (label, block) = work_list.pop()
            for (i, instr) in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr):
                        workfn = self._do_work_expr
                        if guard(workfn, state, work_list, block, i, expr, inline_worker):
                            modified = True
                            break
        if self._DEBUG:
            print('after overload inline'.center(80, '-'))
            print(state.func_id.unique_name)
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        if modified:
            cfg = compute_cfg_from_blocks(state.func_ir.blocks)
            for dead in cfg.dead_nodes():
                del state.func_ir.blocks[dead]
            dead_code_elimination(state.func_ir, typemap=state.typemap)
            state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)
        if self._DEBUG:
            print('after overload inline DCE'.center(80, '-'))
            print(state.func_id.unique_name)
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        return True

    def _get_attr_info(self, state, expr):
        if False:
            for i in range(10):
                print('nop')
        recv_type = state.typemap[expr.value.name]
        recv_type = types.unliteral(recv_type)
        matched = state.typingctx.find_matching_getattr_template(recv_type, expr.attr)
        if not matched:
            return None
        template = matched['template']
        if getattr(template, 'is_method', False):
            return None
        templates = [template]
        sig = typing.signature(matched['return_type'], recv_type)
        arg_typs = sig.args
        is_method = False
        return (templates, sig, arg_typs, is_method)

    def _get_callable_info(self, state, expr):
        if False:
            for i in range(10):
                print('nop')

        def get_func_type(state, expr):
            if False:
                while True:
                    i = 10
            func_ty = None
            if expr.op == 'call':
                try:
                    func_ty = state.typemap[expr.func.name]
                except KeyError:
                    return None
                if not hasattr(func_ty, 'get_call_type'):
                    return None
            elif is_operator_or_getitem(expr):
                func_ty = state.typingctx.resolve_value_type(expr.fn)
            else:
                return None
            return func_ty
        if expr.op == 'call':
            to_inline = None
            try:
                to_inline = state.func_ir.get_definition(expr.func)
            except Exception:
                return None
            if getattr(to_inline, 'op', False) == 'make_function':
                return None
        func_ty = get_func_type(state, expr)
        if func_ty is None:
            return None
        sig = state.calltypes[expr]
        if not sig:
            return None
        (templates, arg_typs, is_method) = (None, None, False)
        if getattr(func_ty, 'template', None) is not None:
            is_method = True
            templates = [func_ty.template]
            arg_typs = (func_ty.template.this,) + sig.args
        else:
            templates = getattr(func_ty, 'templates', None)
            arg_typs = sig.args
        return (templates, sig, arg_typs, is_method)

    def _do_work_expr(self, state, work_list, block, i, expr, inline_worker):
        if False:
            return 10

        def select_template(templates, args):
            if False:
                while True:
                    i = 10
            if templates is None:
                return None
            impl = None
            for template in templates:
                inline_type = getattr(template, '_inline', None)
                if inline_type is None:
                    continue
                if args not in template._inline_overloads:
                    continue
                if not inline_type.is_never_inline:
                    try:
                        impl = template._overload_func(*args)
                        if impl is None:
                            raise Exception
                        break
                    except Exception:
                        continue
            else:
                return None
            return (template, inline_type, impl)
        inlinee_info = None
        if expr.op == 'getattr':
            inlinee_info = self._get_attr_info(state, expr)
        else:
            inlinee_info = self._get_callable_info(state, expr)
        if not inlinee_info:
            return False
        (templates, sig, arg_typs, is_method) = inlinee_info
        inlinee = select_template(templates, arg_typs)
        if inlinee is None:
            return False
        (template, inlinee_type, impl) = inlinee
        return self._run_inliner(state, inlinee_type, sig, template, arg_typs, expr, i, impl, block, work_list, is_method, inline_worker)

    def _run_inliner(self, state, inline_type, sig, template, arg_typs, expr, i, impl, block, work_list, is_method, inline_worker):
        if False:
            for i in range(10):
                print('nop')
        do_inline = True
        if not inline_type.is_always_inline:
            from numba.core.typing.templates import _inline_info
            caller_inline_info = _inline_info(state.func_ir, state.typemap, state.calltypes, sig)
            iinfo = template._inline_overloads[arg_typs]['iinfo']
            if inline_type.has_cost_model:
                do_inline = inline_type.value(expr, caller_inline_info, iinfo)
            else:
                assert 'unreachable'
        if do_inline:
            if is_method:
                if not self._add_method_self_arg(state, expr):
                    return False
            arg_typs = template._inline_overloads[arg_typs]['folded_args']
            iinfo = template._inline_overloads[arg_typs]['iinfo']
            freevars = iinfo.func_ir.func_id.func.__code__.co_freevars
            (_, _, _, new_blocks) = inline_worker.inline_ir(state.func_ir, block, i, iinfo.func_ir, freevars, arg_typs=arg_typs)
            if work_list is not None:
                for blk in new_blocks:
                    work_list.append(blk)
            return True
        else:
            return False

    def _add_method_self_arg(self, state, expr):
        if False:
            print('Hello World!')
        func_def = guard(get_definition, state.func_ir, expr.func)
        if func_def is None:
            return False
        expr.args.insert(0, func_def.value)
        return True

@register_pass(mutates_CFG=False, analysis_only=False)
class DeadCodeElimination(FunctionPass):
    """
    Does dead code elimination
    """
    _name = 'dead_code_elimination'

    def __init__(self):
        if False:
            return 10
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if False:
            while True:
                i = 10
        dead_code_elimination(state.func_ir, state.typemap)
        return True

@register_pass(mutates_CFG=False, analysis_only=False)
class PreLowerStripPhis(FunctionPass):
    """Remove phi nodes (ir.Expr.phi) introduced by SSA.

    This is needed before Lowering because the phi nodes in Numba IR do not
    match the semantics of phi nodes in LLVM IR. In Numba IR, phi nodes may
    expand into multiple LLVM instructions.
    """
    _name = 'strip_phis'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if False:
            return 10
        state.func_ir = self._strip_phi_nodes(state.func_ir)
        state.func_ir._definitions = build_definitions(state.func_ir.blocks)
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run(emit_dels=False)
        if state.func_ir.generator_info is not None and state.typemap is not None:
            gentype = state.return_type
            state_vars = state.func_ir.generator_info.state_vars
            state_types = [state.typemap[k] for k in state_vars]
            state.return_type = types.Generator(gen_func=gentype.gen_func, yield_type=gentype.yield_type, arg_types=gentype.arg_types, state_types=state_types, has_finalizer=gentype.has_finalizer)
        return True

    def _strip_phi_nodes(self, func_ir):
        if False:
            return 10
        'Strip Phi nodes from ``func_ir``\n\n        For each phi node, put incoming value to their respective incoming\n        basic-block at possibly the latest position (i.e. after the latest\n        assignment to the corresponding variable).\n        '
        exporters = defaultdict(list)
        phis = set()
        for (label, block) in func_ir.blocks.items():
            for assign in block.find_insts(ir.Assign):
                if isinstance(assign.value, ir.Expr):
                    if assign.value.op == 'phi':
                        phis.add(assign)
                        phi = assign.value
                        for (ib, iv) in zip(phi.incoming_blocks, phi.incoming_values):
                            exporters[ib].append((assign.target, iv))
        newblocks = {}
        for (label, block) in func_ir.blocks.items():
            newblk = copy(block)
            newblocks[label] = newblk
            newblk.body = [stmt for stmt in block.body if stmt not in phis]
            for (target, rhs) in exporters[label]:
                if rhs is ir.UNDEFINED:
                    rhs = ir.Expr.null(loc=func_ir.loc)
                assign = ir.Assign(target=target, value=rhs, loc=rhs.loc)
                assignments = [stmt for stmt in newblk.find_insts(ir.Assign) if stmt.target == rhs]
                if assignments:
                    last_assignment = assignments[-1]
                    newblk.insert_after(assign, last_assignment)
                else:
                    newblk.prepend(assign)
        func_ir.blocks = newblocks
        return func_ir