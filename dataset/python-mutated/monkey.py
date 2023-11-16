class MonkeyPatcher:
    """
    Cover up attributes with new objects. Neat for monkey-patching things for
    unit-testing purposes.
    """

    def __init__(self, *patches):
        if False:
            i = 10
            return i + 15
        self._patchesToApply = []
        self._originals = []
        for patch in patches:
            self.addPatch(*patch)

    def addPatch(self, obj, name, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a patch so that the attribute C{name} on C{obj} will be assigned to\n        C{value} when C{patch} is called or during C{runWithPatches}.\n\n        You can restore the original values with a call to restore().\n        '
        self._patchesToApply.append((obj, name, value))

    def _alreadyPatched(self, obj, name):
        if False:
            while True:
                i = 10
        '\n        Has the C{name} attribute of C{obj} already been patched by this\n        patcher?\n        '
        for (o, n, v) in self._originals:
            if (o, n) == (obj, name):
                return True
        return False

    def patch(self):
        if False:
            i = 10
            return i + 15
        '\n        Apply all of the patches that have been specified with L{addPatch}.\n        Reverse this operation using L{restore}.\n        '
        for (obj, name, value) in self._patchesToApply:
            if not self._alreadyPatched(obj, name):
                self._originals.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
    __enter__ = patch

    def restore(self):
        if False:
            return 10
        '\n        Restore all original values to any patched objects.\n        '
        while self._originals:
            (obj, name, value) = self._originals.pop()
            setattr(obj, name, value)

    def __exit__(self, excType=None, excValue=None, excTraceback=None):
        if False:
            print('Hello World!')
        self.restore()

    def runWithPatches(self, f, *args, **kw):
        if False:
            while True:
                i = 10
        '\n        Apply each patch already specified. Then run the function f with the\n        given args and kwargs. Restore everything when done.\n        '
        self.patch()
        try:
            return f(*args, **kw)
        finally:
            self.restore()