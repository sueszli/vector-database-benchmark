""" Standard plug-in to take advantage of pylint or PyDev annotations.

Nuitka can detect some things that PyLint and PyDev will complain about too,
and sometimes it's a false alarm, so people add disable markers into their
source code. Nuitka does it itself.

This tries to parse the code for these markers and uses hooks to prevent Nuitka
from warning about things, disabled to PyLint or Eclipse. The idea is that we
won't have another mechanism for Nuitka, but use existing ones instead.

The code for this is very incomplete, barely good enough to cover Nuitka's own
usage of PyLint markers. PyDev is still largely to be started. You are welcome
to grow both.

"""
import re
from nuitka.__past__ import intern
from nuitka.plugins.PluginBase import NuitkaPluginBase

class NuitkaPluginPylintEclipseAnnotations(NuitkaPluginBase):
    plugin_name = 'pylint-warnings'
    plugin_desc = 'Support PyLint / PyDev linting source markers.'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.line_annotations = {}

    def checkModuleSourceCode(self, module_name, source_code):
        if False:
            for i in range(10):
                print('nop')
        annotations = {}
        for (count, line) in enumerate(source_code.split('\n'), 1):
            match = re.search('#.*pylint:\\s*disable=\\s*([\\w,-]+)', line)
            if match:
                comment_only = line[:line.find('#') - 1].strip() == ''
                if comment_only:
                    pass
                else:
                    annotations[count] = set((intern(match.strip()) for match in match.group(1).split(',')))
        if annotations:
            self.line_annotations[module_name] = annotations

    def suppressUnknownImportWarning(self, importing, module_name, source_ref):
        if False:
            while True:
                i = 10
        annotations = self.line_annotations.get(importing.getFullName(), {})
        line_annotations = annotations.get(source_ref.getLineNumber(), ())
        if 'F0401' in line_annotations or 'import-error' in line_annotations:
            return True
        return False
if False:

    class NuitkaPluginDetectorPylintEclipseAnnotations(NuitkaPluginBase):
        detector_for = NuitkaPluginPylintEclipseAnnotations

        def onModuleSourceCode(self, module_name, source_filename, source_code):
            if False:
                print('Hello World!')
            if re.search('#\\s*pylint:\\s*disable=\\s*(\\w+)', source_code):
                self.warnUnusedPlugin('Understand PyLint/PyDev annotations for warnings.')
            return source_code