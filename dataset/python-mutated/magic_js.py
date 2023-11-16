import js as globalThis
from pyscript.util import NotSupported
RUNNING_IN_WORKER = not hasattr(globalThis, 'document')
if RUNNING_IN_WORKER:
    import polyscript
    PyWorker = NotSupported('pyscript.PyWorker', 'pyscript.PyWorker works only when running in the main thread')
    window = polyscript.xworker.window
    document = window.document
    sync = polyscript.xworker.sync

    def current_target():
        if False:
            while True:
                i = 10
        return polyscript.target
else:
    import _pyscript
    from _pyscript import PyWorker
    window = globalThis
    document = globalThis.document
    sync = NotSupported('pyscript.sync', 'pyscript.sync works only when running in a worker')

    def current_target():
        if False:
            i = 10
            return i + 15
        return _pyscript.target