import re
import sys
import gdb
import gdb.types
import gdb.xmethod

class SmallVectorImplWorker_at(gdb.xmethod.XMethodWorker):

    def __init__(self, t):
        if False:
            for i in range(10):
                print('nop')
        self.t = t

    def get_arg_types(self):
        if False:
            while True:
                i = 10
        return gdb.lookup_type('int')

    def get_result_type(self, *args):
        if False:
            return 10
        return self.t

    def __call__(self, obj, i):
        if False:
            return 10
        return (obj['m_begin_ptr'].cast(self.t.pointer()) + i).dereference()

class SmallVectorImplWorker_size(gdb.xmethod.XMethodWorker):

    def __init__(self, t):
        if False:
            for i in range(10):
                print('nop')
        self.t = t

    def get_arg_types(self):
        if False:
            i = 10
            return i + 15
        return None

    def get_result_type(self, *args):
        if False:
            print('Hello World!')
        return gdb.lookup_type('int')

    def __call__(self, obj):
        if False:
            return 10
        return obj['m_end_ptr'].cast(self.t.pointer()) - obj['m_begin_ptr'].cast(self.t.pointer())

class SmallVectorImplMatcher(gdb.xmethod.XMethodMatcher):

    def __init__(self):
        if False:
            return 10
        super().__init__('SmallVectorImplMatcher')

    def match(self, class_type, method_name):
        if False:
            i = 10
            return i + 15
        if re.match('^megdnn::SmallVector(Impl)?<.*>', class_type.tag):
            if method_name == 'at':
                return SmallVectorImplWorker_at(class_type.template_argument(0))
            if method_name == 'operator[]':
                return SmallVectorImplWorker_at(class_type.template_argument(0))
            if method_name == 'size':
                return SmallVectorImplWorker_size(class_type.template_argument(0))
if sys.version_info.major > 2:
    gdb.xmethod.register_xmethod_matcher(None, SmallVectorImplMatcher())