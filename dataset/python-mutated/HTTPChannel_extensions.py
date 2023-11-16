from panda3d import core
from .extension_native_helpers import Dtool_funcToMethod

def spawnTask(self, name=None, callback=None, extraArgs=[]):
    if False:
        while True:
            i = 10
    'Spawns a task to service the download recently requested\n    via beginGetDocument(), etc., and/or downloadToFile() or\n    downloadToRam().  If a callback is specified, that function is\n    called when the download is complete, passing in the extraArgs\n    given.\n\n    Returns the newly-spawned task.\n    '
    if not name:
        name = str(self.getUrl())
    from direct.task import Task
    from direct.task.TaskManagerGlobal import taskMgr
    task = Task.Task(self.doTask)
    task.callback = callback
    task.callbackArgs = extraArgs
    return taskMgr.add(task, name)
if hasattr(core, 'HTTPChannel'):
    Dtool_funcToMethod(spawnTask, core.HTTPChannel)
del spawnTask

def doTask(self, task):
    if False:
        while True:
            i = 10
    from direct.task import Task
    if self.run():
        return Task.cont
    if task.callback:
        task.callback(*task.callbackArgs)
    return Task.done
if hasattr(core, 'HTTPChannel'):
    Dtool_funcToMethod(doTask, core.HTTPChannel)
del doTask