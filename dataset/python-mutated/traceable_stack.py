"""A simple stack that associates filename and line numbers with each object."""
import inspect

class TraceableObject(object):
    """Wrap an object together with its the code definition location."""
    (SUCCESS, HEURISTIC_USED, FAILURE) = (0, 1, 2)

    def __init__(self, obj, filename=None, lineno=None):
        if False:
            i = 10
            return i + 15
        self.obj = obj
        self.filename = filename
        self.lineno = lineno

    def set_filename_and_line_from_caller(self, offset=0):
        if False:
            return 10
        "Set filename and line using the caller's stack frame.\n\n    If the requested stack information is not available, a heuristic may\n    be applied and self.HEURISTIC USED will be returned.  If the heuristic\n    fails then no change will be made to the filename and lineno members\n    (None by default) and self.FAILURE will be returned.\n\n    Args:\n      offset: Integer.  If 0, the caller's stack frame is used.  If 1,\n          the caller's caller's stack frame is used.  Larger values are\n          permissible but if out-of-range (larger than the number of stack\n          frames available) the outermost stack frame will be used.\n\n    Returns:\n      TraceableObject.SUCCESS if appropriate stack information was found,\n      TraceableObject.HEURISTIC_USED if the offset was larger than the stack,\n      and TraceableObject.FAILURE if the stack was empty.\n    "
        retcode = self.SUCCESS
        frame = inspect.currentframe()
        for _ in range(offset + 1):
            parent = frame.f_back
            if parent is None:
                retcode = self.HEURISTIC_USED
                break
            frame = parent
        self.filename = frame.f_code.co_filename
        self.lineno = frame.f_lineno
        return retcode

    def copy_metadata(self):
        if False:
            while True:
                i = 10
        'Return a TraceableObject like this one, but without the object.'
        return self.__class__(None, filename=self.filename, lineno=self.lineno)

class TraceableStack(object):
    """A stack of TraceableObjects."""

    def __init__(self, existing_stack=None):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n    Args:\n      existing_stack: [TraceableObject, ...] If provided, this object will\n        set its new stack to a SHALLOW COPY of existing_stack.\n    '
        self._stack = existing_stack[:] if existing_stack else []

    def push_obj(self, obj, offset=0):
        if False:
            return 10
        "Add object to the stack and record its filename and line information.\n\n    Args:\n      obj: An object to store on the stack.\n      offset: Integer.  If 0, the caller's stack frame is used.  If 1,\n          the caller's caller's stack frame is used.\n\n    Returns:\n      TraceableObject.SUCCESS if appropriate stack information was found,\n      TraceableObject.HEURISTIC_USED if the stack was smaller than expected,\n      and TraceableObject.FAILURE if the stack was empty.\n    "
        traceable_obj = TraceableObject(obj)
        self._stack.append(traceable_obj)
        return traceable_obj.set_filename_and_line_from_caller(offset + 1)

    def pop_obj(self):
        if False:
            print('Hello World!')
        'Remove last-inserted object and return it, without filename/line info.'
        return self._stack.pop().obj

    def peek_top_obj(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the most recent stored object.'
        return self._stack[-1].obj

    def peek_objs(self):
        if False:
            print('Hello World!')
        'Return iterator over stored objects ordered newest to oldest.'
        return (t_obj.obj for t_obj in reversed(self._stack))

    def peek_traceable_objs(self):
        if False:
            i = 10
            return i + 15
        'Return iterator over stored TraceableObjects ordered newest to oldest.'
        return reversed(self._stack)

    def __len__(self):
        if False:
            while True:
                i = 10
        'Return number of items on the stack, and used for truth-value testing.'
        return len(self._stack)

    def copy(self):
        if False:
            i = 10
            return i + 15
        'Return a copy of self referencing the same objects but in a new list.\n\n    This method is implemented to support thread-local stacks.\n\n    Returns:\n      TraceableStack with a new list that holds existing objects.\n    '
        return TraceableStack(self._stack)