""" Tags and set of it.

Used by optimization to keep track of the current state of optimization, these
tags trigger the execution of optimization steps, which in turn may emit these
tags to execute other steps.

"""
allowed_tags = ('new_code', 'new_import', 'new_statements', 'new_expression', 'loop_analysis', 'var_usage', 'read_only_mvar', 'trusted_module_variables', 'new_builtin_ref', 'new_builtin', 'new_raise', 'new_constant')

class TagSet(set):

    def onSignal(self, signal):
        if False:
            i = 10
            return i + 15
        if type(signal) is str:
            signal = signal.split()
        for tag in signal:
            self.add(tag)

    def check(self, tags):
        if False:
            for i in range(10):
                print('nop')
        for tag in tags.split():
            assert tag in allowed_tags, tag
            if tag in self:
                return True
        return False

    def add(self, tag):
        if False:
            return 10
        assert tag in allowed_tags, tag
        set.add(self, tag)