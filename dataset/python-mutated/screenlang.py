from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
import contextlib

class ScreenLangScreen(renpy.object.Object):
    """
    This represents a screen defined in the screen language.
    """
    __version__ = 1
    variant = 'None'
    predict = 'False'
    parameters = None
    location = None

    def __init__(self):
        if False:
            print('Hello World!')
        self.name = '<unknown>'
        self.modal = 'False'
        self.zorder = '0'
        self.tag = None
        self.code = None
        self.variant = 'None'
        self.predict = 'None'
        self.parameters = None
        raise Exception('Creating a new ScreenLangScreen is no longer supported.')

    def after_upgrade(self, version):
        if False:
            print('Hello World!')
        if version < 1:
            self.modal = 'False'
            self.zorder = '0'

    def define(self, location):
        if False:
            for i in range(10):
                print('nop')
        '\n        Defines a screen.\n        '
        renpy.display.screen.define_screen(self.name, self, modal=self.modal, zorder=self.zorder, tag=self.tag, variant=renpy.python.py_eval(self.variant), predict=renpy.python.py_eval(self.predict), parameters=self.parameters, location=self.location)

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        scope = kwargs['_scope']
        if self.parameters:
            args = scope.get('_args', ())
            kwargs = scope.get('_kwargs', {})
            values = renpy.ast.apply_arguments(self.parameters, args, kwargs)
            scope.update(values)
        renpy.python.py_exec_bytecode(self.code.bytecode, locals=scope)