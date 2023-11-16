"""
Runnable
========
"""
from jnius import PythonJavaClass, java_method, autoclass
from android.config import ACTIVITY_CLASS_NAME
_PythonActivity = autoclass(ACTIVITY_CLASS_NAME)
__functionstable__ = {}

class Runnable(PythonJavaClass):
    """Wrapper around Java Runnable class. This class can be used to schedule a
    call of a Python function into the PythonActivity thread.
    """
    __javainterfaces__ = ['java/lang/Runnable']
    __runnables__ = []

    def __init__(self, func):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.func = func

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.args = args
        self.kwargs = kwargs
        Runnable.__runnables__.append(self)
        _PythonActivity.mActivity.runOnUiThread(self)

    @java_method('()V')
    def run(self):
        if False:
            i = 10
            return i + 15
        try:
            self.func(*self.args, **self.kwargs)
        except:
            import traceback
            traceback.print_exc()
        Runnable.__runnables__.remove(self)

def run_on_ui_thread(f):
    if False:
        return 10
    'Decorator to create automatically a :class:`Runnable` object with the\n    function. The function will be delayed and call into the Activity thread.\n    '
    if f not in __functionstable__:
        rfunction = Runnable(f)
        __functionstable__[f] = {'rfunction': rfunction}
    rfunction = __functionstable__[f]['rfunction']

    def f2(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        rfunction(*args, **kwargs)
    return f2