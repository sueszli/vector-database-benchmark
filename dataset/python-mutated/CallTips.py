import inspect
import string
import sys
import traceback

class CallTips:
    menudefs = []
    keydefs = {'<<paren-open>>': ['<Key-parenleft>'], '<<paren-close>>': ['<Key-parenright>'], '<<check-calltip-cancel>>': ['<KeyRelease>'], '<<calltip-cancel>>': ['<ButtonPress>', '<Key-Escape>']}
    windows_keydefs = {}
    unix_keydefs = {}

    def __init__(self, editwin):
        if False:
            i = 10
            return i + 15
        self.editwin = editwin
        self.text = editwin.text
        self.calltip = None
        if hasattr(self.text, 'make_calltip_window'):
            self._make_calltip_window = self.text.make_calltip_window
        else:
            self._make_calltip_window = self._make_tk_calltip_window

    def close(self):
        if False:
            while True:
                i = 10
        self._make_calltip_window = None

    def _make_tk_calltip_window(self):
        if False:
            while True:
                i = 10
        import CallTipWindow
        return CallTipWindow.CallTip(self.text)

    def _remove_calltip_window(self):
        if False:
            i = 10
            return i + 15
        if self.calltip:
            self.calltip.hidetip()
            self.calltip = None

    def paren_open_event(self, event):
        if False:
            while True:
                i = 10
        self._remove_calltip_window()
        arg_text = get_arg_text(self.get_object_at_cursor())
        if arg_text:
            self.calltip_start = self.text.index('insert')
            self.calltip = self._make_calltip_window()
            self.calltip.showtip(arg_text)
        return ''

    def paren_close_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._remove_calltip_window()
        return ''

    def check_calltip_cancel_event(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self.calltip:
            if self.text.compare('insert', '<=', self.calltip_start) or self.text.compare('insert', '>', self.calltip_start + ' lineend'):
                self._remove_calltip_window()
        return ''

    def calltip_cancel_event(self, event):
        if False:
            while True:
                i = 10
        self._remove_calltip_window()
        return ''

    def get_object_at_cursor(self, wordchars='._' + string.ascii_uppercase + string.ascii_lowercase + string.digits):
        if False:
            print('Hello World!')
        text = self.text
        chars = text.get('insert linestart', 'insert')
        i = len(chars)
        while i and chars[i - 1] in wordchars:
            i = i - 1
        word = chars[i:]
        if word:
            import __main__
            namespace = sys.modules.copy()
            namespace.update(__main__.__dict__)
            try:
                return eval(word, namespace)
            except:
                pass
        return None

def _find_constructor(class_ob):
    if False:
        return 10
    try:
        return class_ob.__init__
    except AttributeError:
        for base in class_ob.__bases__:
            rc = _find_constructor(base)
            if rc is not None:
                return rc
    return None

def get_arg_text(ob):
    if False:
        for i in range(10):
            print('nop')
    argText = ''
    if ob is not None:
        if inspect.isclass(ob):
            fob = _find_constructor(ob)
            if fob is None:
                fob = lambda : None
        else:
            fob = ob
        if inspect.isfunction(fob) or inspect.ismethod(fob):
            try:
                argText = str(inspect.signature(fob))
            except:
                print('Failed to format the args')
                traceback.print_exc()
        if hasattr(ob, '__doc__'):
            doc = ob.__doc__
            try:
                doc = doc.strip()
                pos = doc.find('\n')
            except AttributeError:
                pass
            else:
                if pos < 0 or pos > 70:
                    pos = 70
                if argText:
                    argText = argText + '\n'
                argText = argText + doc[:pos]
    return argText
if __name__ == '__main__':

    def t1():
        if False:
            for i in range(10):
                print('nop')
        '()'

    def t2(a, b=None):
        if False:
            print('Hello World!')
        '(a, b=None)'

    def t3(a, *args):
        if False:
            while True:
                i = 10
        '(a, *args)'

    def t4(*args):
        if False:
            while True:
                i = 10
        '(*args)'

    def t5(a, *args):
        if False:
            for i in range(10):
                print('nop')
        '(a, *args)'

    def t6(a, b=None, *args, **kw):
        if False:
            while True:
                i = 10
        '(a, b=None, *args, **kw)'

    class TC:
        """(self, a=None, *b)"""

        def __init__(self, a=None, *b):
            if False:
                i = 10
                return i + 15
            '(self, a=None, *b)'

        def t1(self):
            if False:
                return 10
            '(self)'

        def t2(self, a, b=None):
            if False:
                return 10
            '(self, a, b=None)'

        def t3(self, a, *args):
            if False:
                while True:
                    i = 10
            '(self, a, *args)'

        def t4(self, *args):
            if False:
                return 10
            '(self, *args)'

        def t5(self, a, *args):
            if False:
                i = 10
                return i + 15
            '(self, a, *args)'

        def t6(self, a, b=None, *args, **kw):
            if False:
                for i in range(10):
                    print('nop')
            '(self, a, b=None, *args, **kw)'

    def test(tests):
        if False:
            print('Hello World!')
        failed = []
        for t in tests:
            expected = t.__doc__ + '\n' + t.__doc__
            if get_arg_text(t) != expected:
                failed.append(t)
                print(f'{t} - expected {repr(expected)}, but got {repr(get_arg_text(t))}')
        print('%d of %d tests failed' % (len(failed), len(tests)))
    tc = TC()
    tests = (t1, t2, t3, t4, t5, t6, TC, tc.t1, tc.t2, tc.t3, tc.t4, tc.t5, tc.t6)
    test(tests)