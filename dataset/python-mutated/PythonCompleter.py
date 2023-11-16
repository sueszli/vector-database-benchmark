__all__ = ['PythonCompleter']

class PythonCompleter:

    def __init__(self, local_ns=None, global_ns=None):
        if False:
            for i in range(10):
                print('nop')
        if local_ns is not None:
            self.local_ns = local_ns
        else:
            self.local_ns = {}
        if global_ns is not None:
            self.global_ns = global_ns
        else:
            self.global_ns = {}

    def complete(self, text, state):
        if False:
            print('Hello World!')
        if state == 0:
            if '.' in text:
                self.matches = self.attr_matches(text)
            else:
                self.matches = self.var_matches(text)
        try:
            return self.matches[state]
        except IndexError:
            return None

    def _callable_postfix(self, val, word):
        if False:
            return 10
        if hasattr(val, '__call__'):
            word = word + '('
        return word

    def var_matches(self, text):
        if False:
            i = 10
            return i + 15
        import re
        m = re.match('(\\w*)', text)
        if not m:
            return []
        words = [x for x in self.local_ns.iterkeys() if x.startswith(m.group(1))]
        if '__builtins__' in words:
            words.remove('__builtins__')
        return words

    def attr_matches(self, text):
        if False:
            for i in range(10):
                print('nop')
        'Compute matches when text contains a dot.\n\n        Assuming the text is of the form NAME.NAME....[NAME], and is\n        evaluatable in self.namespace, it will be evaluated and its attributes\n        (as revealed by dir()) are used as possible completions.  (For class\n        instances, class members are also considered.)\n\n        WARNING: this can still invoke arbitrary C code, if an object\n        with a __getattr__ hook is evaluated.\n        '
        import re
        bsw = '[a-zA-Z0-9_\\(\\)\\[\\]"\']'
        m = re.match('(\\w+(\\.\\w+)*)\\.(\\w*)'.replace('\\w', bsw), text)
        if not m:
            return []
        (expr, attr) = m.group(1, 3)
        try:
            try:
                thisobject = eval(expr, self.global_ns, self.local_ns)
            except NameError:
                '\n                print str(e)\n                try:\n                    exec "import %s"%expr in global_ns, self.local_ns\n                    thisobject = eval(expr, global_ns, self.local_ns)\n                except ImportError:\n                    pass\n                '
        except:
            return []
        words = dir(thisobject)
        if '__builtins__' in words:
            words.remove('__builtins__')
        if hasattr(thisobject, '__class__'):
            words.append('__class__')
            words.extend(get_class_members(thisobject.__class__))
        words = [x for x in words if not x.startswith('__')]
        matches = []
        n = len(attr)
        for word in words:
            if word[:n] == attr and hasattr(thisobject, word):
                val = getattr(thisobject, word)
                word = self._callable_postfix(val, '%s.%s' % (expr, word))
                matches.append(word)
        return matches

def get_class_members(klass):
    if False:
        for i in range(10):
            print('nop')
    ret = dir(klass)
    if hasattr(klass, '__bases__'):
        for base in klass.__bases__:
            ret = ret + get_class_members(base)
    return ret
if __name__ == '__main__':
    import code
    import readline
    readline.set_completer(PythonCompleter().complete)
    readline.parse_and_bind('tab: complete')
    code.interact()