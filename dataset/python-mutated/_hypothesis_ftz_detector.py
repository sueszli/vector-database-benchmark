"""
This is a toolkit for determining which module set the "flush to zero" flag.

For details, see the docstring and comments in `identify_ftz_culprit()`.  This module
is defined outside the main Hypothesis namespace so that we can avoid triggering
import of Hypothesis itself from each subprocess which must import the worker function.
"""
import importlib
import sys
KNOWN_EVER_CULPRITS = ('archive-pdf-tools', 'bgfx-python', 'bicleaner-ai-glove', 'BTrees', 'cadbiom', 'ctranslate2', 'dyNET', 'dyNET38', 'gevent', 'glove-python-binary', 'higra', 'hybridq', 'ikomia', 'ioh', 'jij-cimod', 'lavavu', 'lavavu-osmesa', 'MulticoreTSNE', 'neural-compressor', 'nwhy', 'openjij', 'openturns', 'perfmetrics', 'pHashPy', 'pyace-lite', 'pyapr', 'pycompadre', 'pycompadre-serial', 'PyKEP', 'pykep', 'pylimer-tools', 'pyqubo', 'pyscf', 'PyTAT', 'python-prtree', 'qiskit-aer', 'qiskit-aer-gpu', 'RelStorage', 'sail-ml', 'segmentation', 'sente', 'sinr', 'snapml', 'superman', 'symengine', 'systran-align', 'texture-tool', 'tsne-mp', 'xcsf')

def flush_to_zero():
    if False:
        return 10
    return 2.0 ** (-1073) == 0

def run_in_process(fn, *args):
    if False:
        print('Hello World!')
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    q = mp.Queue()
    p = mp.Process(target=target, args=(q, fn, *args))
    p.start()
    retval = q.get()
    p.join()
    return retval

def target(q, fn, *args):
    if False:
        while True:
            i = 10
    q.put(fn(*args))

def always_imported_modules():
    if False:
        while True:
            i = 10
    return (flush_to_zero(), set(sys.modules))

def modules_imported_by(mod):
    if False:
        print('Hello World!')
    'Return the set of modules imported transitively by mod.'
    before = set(sys.modules)
    try:
        importlib.import_module(mod)
    except Exception:
        return (None, set())
    imports = set(sys.modules) - before
    return (flush_to_zero(), imports)
KNOWN_FTZ = None
CHECKED_CACHE = set()

def identify_ftz_culprits():
    if False:
        for i in range(10):
            print('nop')
    'Find the modules in sys.modules which cause "mod" to be imported.'
    global KNOWN_FTZ
    if KNOWN_FTZ:
        return KNOWN_FTZ
    (always_enables_ftz, always_imports) = run_in_process(always_imported_modules)
    if always_enables_ftz:
        raise RuntimeError('Python is always in FTZ mode, even without imports!')
    CHECKED_CACHE.update(always_imports)

    def key(name):
        if False:
            while True:
                i = 10
        'Prefer known-FTZ modules, then top-level packages, then alphabetical.'
        return (name not in KNOWN_EVER_CULPRITS, name.count('.'), name)
    candidates = set(sys.modules) - CHECKED_CACHE
    triggering_modules = {}
    while candidates:
        mod = min(candidates, key=key)
        candidates.discard(mod)
        (enables_ftz, imports) = run_in_process(modules_imported_by, mod)
        imports -= CHECKED_CACHE
        if enables_ftz:
            triggering_modules[mod] = imports
            candidates &= imports
        else:
            candidates -= imports
            CHECKED_CACHE.update(imports)
    prefixes = tuple((n + '.' for n in triggering_modules))
    result = {k for k in triggering_modules if not k.startswith(prefixes)}
    for a in sorted(result):
        for b in sorted(result):
            if a in triggering_modules[b] and b not in triggering_modules[a]:
                result.discard(b)
    KNOWN_FTZ = min(result)
    return KNOWN_FTZ
if __name__ == '__main__':
    import grequests
    print(identify_ftz_culprits())