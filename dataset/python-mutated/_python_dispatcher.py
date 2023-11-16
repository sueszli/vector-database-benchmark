import re
import torch._C as C
'\nPythonDispatcher class is a thin python-binding to C++ dispatcher and it\nis designed to show how dispatcher precompute works. In particular,\nit shows for a certain op `foo`, what the computed dispatch table looks\nlike after user register their kernels to certains dispatch keys.\n\nIn the real C++ dispatcher we support many dispatch keys for different\nfunctionalities. For simplicity PythonDispatcher only supports dispatch\nkeys for a single example of each use case. These use cases are listed below:\n\n- CPU/AutogradCPU: represents in-tree backends which we usually have dedicated inference &\n    autograd kernel in pytorch core library.\n    E.g. CPU, CUDA\n- FPGA/AutogradOther: represents in-tree backends which we usually have backend specific\n    inference kernels, but they share the same autograd kernel specified in AutogradOther.\n    E.g. FPGA, SparseCsrCPU\n- XLA/AutogradXLA: represents out-of-tree backends which we don\'t have either inference or autograd\n    kernel defined in pytorch core library. Backend owner is responsible for registering both\n    inference & autograd kernels in their extensions(e.g. torch-xla) for the operators they support.\n    E.g. XLA, XPU, MPS\n- CompositeExplicitAutograd: alias key mapped to inference kernels of all backends like CPU, CUDA, XLA etc.\n    Kernels registered to this key MUST work for inference for all backends.\n- Autograd: alias key mapped to autograd of all backends like AutogradCPU, AutogradXLA, AutogradOther.\n    Kernels registered to this key MUST work for autograd for all backends.\n- CompositeImplicitAutograd: alias key CompositeImplicitAutograd = CompositeExplicitAutograd + Autograd\n    Kernels registered to this key MUST work for both inference + autograd for all backends.\n\nNote we only allow registrations to alias keys inside pytorch core library. E.g\nyou shouldn\'t register a CompositeImplicitAutograd or CompositeExplicitAutograd\nkernel from torch-xla extension, instead you should upstream the kernel into\npytorch/pytorch repo so that it\'s available for all backends and continuously\ntested even without the extension.\n\nUsage:\n  dispatcher = PythonDispatcher()\n  dispatcher.register(["CPU", "XLA", "CompositeImplicitAutograd"])\n  print(dispatcher.dispatchTable()) # This tells you exactly which kernel is used for certain backend.\n  # For more debugging information\n  # print(dispatcher.keys())\n  # print(dispatcher.registrations())\n  # print(dispatcher.rawRegistrations())\n  # print(dispatcher.rawDispatchTable())\nPythonDispatcher calls C++ dispatcher under the hood for to precompute dispatch table.\nThis file only provides the simplified API for developers, relevant test code is located in\ntest/test_dispatch.py\n'

class PythonDispatcher:
    namespace = '__test__'
    name = 'foo'
    runtime_keys = ['CPU', 'AutogradCPU', 'FPGA', 'AutogradOther', 'XLA', 'AutogradXLA', 'Lazy', 'AutogradLazy']
    alias_keys = ['CompositeExplicitAutograd', 'Autograd', 'CompositeImplicitAutograd']
    supported_keys = runtime_keys + alias_keys

    def __init__(self):
        if False:
            return 10
        C._dispatch_check_invariants(self.name)
        self.ref = C._dispatch_library('FRAGMENT', self.namespace, '')
        self.ref.def_('foo(Tensor x) -> Tensor')
    '\n    Returns a list of dispatch keys supported by PythonDispatcher.\n    You can register kernels to these keys.\n    '

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        return self.supported_keys
    "\n    Register kernels to the target dispatchKeys.\n    dispatchKeys(list[str]): a list of dispatch keys that you want to register\n      your own kernel. Note that you don't need to write the kernel yourself in\n      this PythonDispatcher.E.g. for CPU key, a kernel(e.g fn_CPU for CPU) is\n      automatically generated and registered.\n    "

    def register(self, dispatchKeys):
        if False:
            i = 10
            return i + 15
        if len(set(dispatchKeys)) != len(dispatchKeys):
            raise RuntimeError(f'Overriden is not allowed but found duplicates in {dispatchKeys}.')
        if 'CompositeImplicitAutograd' in dispatchKeys and 'CompositeExplicitAutograd' in dispatchKeys:
            raise RuntimeError('Registration to both CompositeImplicitAutograd and CompositeExplicitAutograd is not allowed.')
        for key in dispatchKeys:
            if key not in self.supported_keys:
                raise RuntimeError(f'{key} is not supported, please select a dispatch key in {self.supported_keys}.')
            self.ref.impl_t_t('foo', dispatch=key, debug='fn_' + key)
    '\n    Helper function to format (key, kernel).\n    '

    def _format_line(self, key, kernel):
        if False:
            print('Hello World!')
        return f'{key:<15} {kernel}\n'
    '\n    Helper function to print a table header.\n    '

    def _format_header(self, header):
        if False:
            print('Hello World!')
        s = f'\n{header}\n'
        s += self._format_line('key', 'kernel')
        s += '---------------------------\n'
        return s
    '\n    Returns raw output of all registration info for debugging only.\n    Use registrations() for a simplified version.\n    '

    def rawRegistrations(self):
        if False:
            while True:
                i = 10
        return C._dispatch_dump(f'{self.namespace}::{self.name}')
    '\n    Returns raw output of computed dispatch table for debugging only.\n    Use dispatchTable() for a simplified version.\n    '

    def rawDispatchTable(self):
        if False:
            return 10
        return C._dispatch_dump_table(f'{self.namespace}::{self.name}')
    '\n    Returns a table(str) including all the registrations from users.\n    Note this includes registrations to both runtime keys and alias keys.\n    '

    def registrations(self):
        if False:
            return 10
        output = self._format_header('Registered Kernels')
        state = self.rawRegistrations()
        state_entries = state.split('\n')
        for line in state_entries:
            first = line.split(':')[0]
            if any((first.startswith(k) for k in self.supported_keys)):
                kernel = line.split('::')[0].split(' ')[1]
                output += self._format_line(first, kernel)
        return output
    '\n    Returns the computed dispatch table(str). Note this only include\n    runtime keys, registrations to alias keys have been decoded to their\n    mapped runtime keys.\n    '

    def dispatchTable(self):
        if False:
            for i in range(10):
                print('nop')
        output = self._format_header('Computed Dispatch Table')
        table = self.rawDispatchTable()
        table_entries = table.split('\n')
        regex = re.compile('registered at .*FallbackKernel\\.cpp.*(\\[)')
        for line in table_entries:
            k = line.split(':')[0]
            if k in self.runtime_keys:
                entry = regex.sub('[', line)
                output += self._format_line(k, entry.split(': ')[1])
        return output