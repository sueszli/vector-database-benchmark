import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented

class ComptimeVar:
    """
    A ComptimeVar represents a Python value, at some particular point
    in time, in the Python code we are symbolically evaluating with
    torchdynamo.  This must be distinguished from a runtime value, as
    at compile-time there are some properties of the variable we
    do not know (for example, if the ComptimeVar represents a Tensor,
    we only know metadata about the tensor; we do NOT know what the
    actual data in the Tensor is.)
    """

    def __init__(self, v):
        if False:
            while True:
                i = 10
        self.__variable = v

    def as_proxy(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns an fx.Proxy (or tuple/list of fx.Proxy) representing\n        this variable in the FX graph we are assembling to pass\n        to the user compiler.\n\n        This method only works for variables we actually track in\n        the FX graph, aka Tensors (and ints, if you are compiling\n        with dynamic shapes).  In particular, if you have a list\n        or tuple of tensors, you will get a list/tuple of proxies\n        (not a single proxy representing the entire list/tuple).\n        '
        return self.__variable.as_proxy()

    def is_proxy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if as_proxy() would succeed.\n        '
        return self.__variable.is_proxy()

    def as_fake(self):
        if False:
            while True:
                i = 10
        '\n        Returns a "fake" value (either a FakeTensor or a SymInt)\n        representing the variable in question.  This only works\n        for variables that denote Tensor or int.  You can use\n        this to query metadata; e.g., v.as_fake().size(0) will\n        tell you the compile-time known size of the tensor.\n\n        WARNING: Do NOT mutate the returned tensor.\n        '
        return self.__variable.as_proxy().node.meta['example_value']

    def size(self, dim: Optional[int]=None) -> Union[int, torch.SymInt]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the size of the tensor (if dim is None) or the size\n        at the dimension dim.  The returned size may be a SymInt.\n        '
        return self.as_fake().size(dim)

    def python_type(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns what type(v) would have returned for the variable\n        at compile time.\n        '
        return self.__variable.python_type()

    def as_python_constant(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the Python value this variable would have, but only if it is\n        completely known at compile-time (e.g., it is constant).\n\n        WARNING: Do NOT mutate the returned constant.  The returned constant\n        may or may not correspond to the actual value this variable may take\n        on at runtime; for example, if the variable in question is a constant\n        list, we may return a copy of that list.\n        '
        return self.__variable.as_python_constant()

    def is_python_constant(self):
        if False:
            return 10
        '\n        Returns True if as_python_constant would succeed.\n        '
        return self.__variable.is_python_constant()

    def _i_will_not_complain_if_bc_breaks_VariableTracker(self):
        if False:
            while True:
                i = 10
        '\n        Returns the internal data structure VariableTracker that Dynamo uses\n        to represent variables at compile time.  There are no BC guarantees on\n        this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if you rely on\n        it.\n        '
        return self.__variable

    def __repr__(self):
        if False:
            print('Hello World!')
        return repr(self.__variable)

class ComptimeContext:
    """
    This context class provides access to a public API for Dynamo's internals.
    If there is something here you would find useful that is missing, please
    file a feature request at https://github.com/pytorch/pytorch/
    """

    def __init__(self, tx):
        if False:
            for i in range(10):
                print('nop')
        self.__tx = tx

    def get_local(self, name: str, *, stacklevel=0) -> ComptimeVar:
        if False:
            return 10
        '\n        Retrieve the compile-time known information about a local.\n        '
        tx = self.__get_tx(stacklevel)
        return ComptimeVar(tx.symbolic_locals[name])

    def graph_break(self, msg='ComptimeContext.graph_break'):
        if False:
            return 10
        '\n        Manually trigger a graph break\n        '
        unimplemented(msg)

    def graph(self):
        if False:
            print('Hello World!')
        '\n        Retrieve the partially constructed FX graph that would be\n        passed to the user compiler after compilation.\n        '
        return self.__tx.output.graph

    def print_graph(self, *, verbose=True, file=None):
        if False:
            i = 10
            return i + 15
        '\n        Print the partially constructed FX graph that would be passed\n        to the user compiler after compilation.\n        '
        print(self.__tx.output.graph.python_code('self', verbose=verbose).src, file=file)

    def parent(self):
        if False:
            i = 10
            return i + 15
        return ComptimeContext(self.__tx.parent)

    def __get_tx(self, stacklevel):
        if False:
            i = 10
            return i + 15
        tx = self.__tx
        for _ in range(stacklevel):
            tx = tx.parent
        return tx

    def print_disas(self, *, file=None, stacklevel=0):
        if False:
            print('Hello World!')
        '\n        Print the current series of opcodes being executed (not including\n        parent frames), including where you are in the particular opcode\n        stream.\n        '
        tx = self.__get_tx(stacklevel)
        print(dis.Bytecode(tx.f_code, current_offset=tx.instructions[tx.instruction_pointer].offset).dis(), file=file)

    def print_value_stack(self, *, file=None, stacklevel=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print the current Python value stack.  Note that this is NOT the same\n        as the traceback; use print_bt() to print that.  Note that at\n        stacklevel=0, this will typically be empty, as comptime cannot\n        currently be used in an expression context where there would be\n        intermediates on the stack.  If you would find this useful, please\n        file a bug at https://github.com/pytorch/pytorch/\n\n        NB: Stack grows downwards in our print\n        '
        tx = self.__get_tx(stacklevel)
        for s in tx.stack:
            print(f'- {s}', file=file)

    def print_locals(self, *, file=None, stacklevel=0):
        if False:
            print('Hello World!')
        '\n        Print all of the locals available in the current context.\n        By default this view is very limited; you can get more information\n        about any individual local using get_local().\n        '
        tx = self.__get_tx(stacklevel)
        for (k, v) in tx.symbolic_locals.items():
            print(f'{k} = {v}', file=file)

    def print_bt(self, *, file=None, stacklevel=0):
        if False:
            i = 10
            return i + 15
        '\n        Print the user code backtrace, starting at the beginning of the\n        frame Dynamo started evaluating.  Note that this MAY NOT go all\n        the way to the torch.compile invocation, as we may have done\n        a graph break and are compiling an intermediate frame as the\n        starting point.  If you think the other behavior would be better,\n        file a bug at https://github.com/pytorch/pytorch/\n        '
        stack = []
        tx = self.__get_tx(stacklevel)
        while tx is not None:
            stack.append(tx.frame_summary())
            tx = getattr(tx, 'parent', None)
        print(''.join(traceback.StackSummary.from_list(reversed(stack)).format()), file=file)

    def print_guards(self, *, file=None):
        if False:
            while True:
                i = 10
        '\n        Print the currently installed guards for the Dynamo context.\n        This does NOT include guards associated with variables that\n        may or may not be installed in the future if those variables\n        are used.\n        '
        print('\n'.join((f'{repr(guard)}' for guard in sorted(self.__tx.output.guards))), file=file)

    def _i_will_not_complain_if_bc_breaks_InstructionTranslator(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the internal data structure InstructionTranslator that Dynamo\n        uses to track state of symbolic evaluation.  There are no BC\n        guarantees on this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if\n        you rely on it.\n        '
        return self.__tx

class _Comptime:

    @staticmethod
    def __call__(fn):
        if False:
            return 10
        'fn gets called at compile time in TorchDynamo, does nothing otherwise'
        return

    @staticmethod
    def graph_break():
        if False:
            return 10
        comptime(lambda ctx: ctx.graph_break())

    @staticmethod
    def print_graph():
        if False:
            for i in range(10):
                print('nop')
        comptime(lambda ctx: ctx.print_graph())

    @staticmethod
    def print_disas(*, stacklevel=0):
        if False:
            while True:
                i = 10
        comptime(lambda ctx: ctx.print_disas(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))

    @staticmethod
    def print_value_stack(*, stacklevel=0):
        if False:
            for i in range(10):
                print('nop')
        comptime(lambda ctx: ctx.print_value_stack(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))

    @staticmethod
    def print_value_stack_and_return(e, *, stacklevel=0):
        if False:
            i = 10
            return i + 15
        comptime(lambda ctx: ctx.print_value_stack(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))
        return e

    @staticmethod
    def print_locals(*, stacklevel=0):
        if False:
            return 10
        comptime(lambda ctx: ctx.print_locals(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))

    @staticmethod
    def print_bt(*, stacklevel=0):
        if False:
            for i in range(10):
                print('nop')
        comptime(lambda ctx: ctx.print_bt(stacklevel=ctx.get_local('stacklevel').as_python_constant() + 1))

    @staticmethod
    def print_guards():
        if False:
            return 10
        comptime(lambda ctx: ctx.print_guards())

    @staticmethod
    def breakpoint():
        if False:
            i = 10
            return i + 15
        '\n        Like pdb breakpoint(), but drop into pdb whenever this line\n        of code is compiled by dynamo.  Use it by putting\n        this in your model code::\n\n            from torch._dynamo.comptime import comptime\n            comptime.breakpoint()\n\n        And then, inside pdb, you can access \'ctx\' to query things\n        about the compilation context::\n\n            (Pdb) !ctx.print_bt()\n            (Pdb) !ctx.print_locals()\n            (Pdb) p ctx.get_local("attention").as_fake()\n        '

        def inner(inner_ctx):
            if False:
                for i in range(10):
                    print('nop')
            ctx = inner_ctx.parent()
            builtins.breakpoint()
        comptime(inner)
comptime = _Comptime()