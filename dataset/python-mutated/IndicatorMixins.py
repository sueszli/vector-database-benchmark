""" Module for node class mixins that indicate runtime determined node facts.

These come into play after finalization only. All of the these attributes (and
we could use properties instead) are determined once or from a default and then
used like this.

"""

class MarkUnoptimizedFunctionIndicatorMixin(object):
    """Mixin for indication that a function contains an exec or star import.

    These do not access global variables directly, but check a locals dictionary
    first, because they do.
    """
    __slots__ = ()

    def __init__(self, flags):
        if False:
            i = 10
            return i + 15
        self.unoptimized_locals = flags is not None and 'has_exec' in flags
        self.unqualified_exec = flags is not None and 'has_unqualified_exec' in flags

    def isUnoptimized(self):
        if False:
            i = 10
            return i + 15
        return self.unoptimized_locals

    def isUnqualifiedExec(self):
        if False:
            for i in range(10):
                print('nop')
        return self.unoptimized_locals and self.unqualified_exec

class MarkNeedsAnnotationsMixin(object):
    __slots__ = ()

    def __init__(self):
        if False:
            return 10
        self.needs_annotations_dict = False

    def markAsNeedsAnnotationsDictionary(self):
        if False:
            print('Hello World!')
        'For use during building only. Indicate "__annotations__" need.'
        self.needs_annotations_dict = True

    def needsAnnotationsDictionary(self):
        if False:
            for i in range(10):
                print('nop')
        'For use during building only. Indicate "__annotations__" need.'
        return self.needs_annotations_dict

class EntryPointMixin(object):
    __slots__ = ()

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.trace_collection = None

    def setTraceCollection(self, trace_collection):
        if False:
            return 10
        previous = self.trace_collection
        self.trace_collection = trace_collection
        return previous

    def getTraceCollection(self):
        if False:
            return 10
        return self.trace_collection