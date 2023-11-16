class Flags(object):
    THROTTLE = 'throttle'
    DISABLE_BYPASS = 'disable_bypass'
    NEED_QT_GUI = 'need_qt_gui'
    DEPRECATED = 'deprecated'
    NOT_DSP = 'not_dsp'
    SHOW_ID = 'show_id'
    HAS_PYTHON = 'python'
    HAS_CPP = 'cpp'

    def __init__(self, flags=None):
        if False:
            for i in range(10):
                print('nop')
        if flags is None:
            flags = set()
        if isinstance(flags, str):
            flags = (f.strip() for f in flags.replace(',', '').split())
        self.data = set(flags)

    def __getattr__(self, item):
        if False:
            while True:
                i = 10
        return item in self

    def __contains__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return item in self.data

    def __str__(self):
        if False:
            while True:
                i = 10
        return ', '.join(self.data)

    def set(self, *flags):
        if False:
            return 10
        self.data.update(flags)