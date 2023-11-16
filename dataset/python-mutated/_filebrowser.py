import os
from ... import event
from .._widget import PyWidget, Widget, create_element
sep = os.path.sep

class _FileBrowserJS(Widget):
    """ JS part of FileBrowserWidget.
    """
    CSS = '\n        .flx-_FileBrowserJS {\n            display: grid;\n            padding: 0.5em;\n            overflow-y: scroll;\n            grid-template-columns: auto 1fr auto;\n            grid-column-gap: 0.5em;\n            justify-items: start;\n            justify-content: stretch;\n            align-content: start;\n            -webkit-user-select: none;  /* Chrome all / Safari all */\n            -moz-user-select: none;     /* Firefox all */\n            -ms-user-select: none;      /* IE 10+ */\n            user-select: none;          /* Likely future */\n        }\n        .flx-_FileBrowserJS > b {\n            box-sizing: border-box;\n            background: #DDD;\n            border-radius: 4px;\n            width: 100%;\n            padding: 0.3em;\n        }\n        .flx-_FileBrowserJS > b > a {\n            cursor: pointer;\n            margin-right: 0.2em;\n        }\n        .flx-_FileBrowserJS > b > a:hover {\n            border-bottom: 2px solid rgba(0, 0, 0, 0.6);\n        }\n        .flx-_FileBrowserJS > u {\n            width: 100%;\n            cursor: pointer;\n            border-bottom: 1px solid rgba(0, 0, 0, 0.1);\n            text-decoration: none;\n        }\n        .flx-_FileBrowserJS > i {\n            border-bottom: 1px solid rgba(0, 0, 0, 0.1);\n            width: 100%;\n        }\n    '
    _items = event.ListProp(settable=True)
    _dirname = event.StringProp(settable=True)

    def init(self):
        if False:
            i = 10
            return i + 15
        self.node.onclick = self._nav

    def _render_dom(self):
        if False:
            return 10
        dirname = self._dirname.rstrip(sep)
        pparts = dirname.split(sep)
        path_els = []
        for i in range(0, len(pparts)):
            el = create_element('a', {'dirname': sep.join(pparts[:i + 1]) + sep}, pparts[i] + sep)
            path_els.append(el)
        elements = []
        elements.append(create_element('b', {}, create_element('a', {'dirname': sep.join(pparts[:-1]) + sep}, '..')))
        elements.append(create_element('b', {}, path_els))
        elements.append(create_element('s', {}, ''))
        for i in range(len(self._items)):
            (kind, fname, size) = self._items[i]
            elements.append(create_element('span', {}, ' ❑■'[kind] or ''))
            if kind == 1:
                elements.append(create_element('u', {'dirname': dirname + sep + fname, 'filename': None}, fname))
            else:
                elements.append(create_element('u', {'filename': dirname + sep + fname, 'dirname': None}, fname))
            if size >= 1048576:
                elements.append(create_element('i', {}, '{:0.1f} MiB'.format(size / 1048576)))
            elif size >= 1024:
                elements.append(create_element('i', {}, '{:0.1f} KiB'.format(size / 1024)))
            elif size >= 0:
                elements.append(create_element('i', {}, '{} B'.format(size)))
            else:
                elements.append(create_element('s', {}, ''))
        return elements

    @event.emitter
    def _nav(self, ev):
        if False:
            i = 10
            return i + 15
        dirname = ev.target.dirname or None
        filename = ev.target.filename or None
        if dirname or filename:
            return {'dirname': dirname, 'filename': filename}

class FileBrowserWidget(PyWidget):
    """ A PyWidget to browse the file system. Experimental. This could be the
    basis for a file open/save dialog.
    """
    _WidgetCls = _FileBrowserJS
    path = event.StringProp('~', doc='\n        The currectly shown directory (settable). Defaults to the user directory.\n        ')

    @event.action
    def set_path(self, dirname=None):
        if False:
            for i in range(10):
                print('nop')
        ' Set the current path. If an invalid directory is given,\n        the path is not changed. The given path can be absolute, or relative\n        to the current path.\n        '
        if dirname is None or not isinstance(dirname, str):
            dirname = '~'
        if dirname.startswith('~'):
            dirname = os.path.expanduser(dirname)
        if not os.path.isabs(dirname):
            dirname = os.path.abspath(os.path.join(self.path, dirname))
        if os.path.isdir(dirname):
            self._mutate('path', dirname)
        elif not self.path:
            self._mutate('path', os.path.expanduser('~'))

    @event.emitter
    def selected(self, filename):
        if False:
            i = 10
            return i + 15
        ' Emitter that fires when the user selects a file. The emitted event\n        has a "filename" attribute.\n        '
        return {'filename': filename}

    @event.reaction
    def _on_path(self):
        if False:
            print('Hello World!')
        path = self.path
        if not path:
            return
        items = []
        for fname in os.listdir(path):
            filename = os.path.join(path, fname)
            if os.path.isdir(filename):
                items.append((1, fname, -1))
            elif os.path.isfile(filename):
                items.append((2, fname, os.path.getsize(filename)))
        items.sort()
        self._jswidget._set_dirname(path)
        self._jswidget._set_items(items)

    @event.reaction('_jswidget._nav')
    def _on_nav(self, *events):
        if False:
            print('Hello World!')
        dirname = events[-1].dirname
        filename = events[-1].filename
        print(dirname, filename)
        if dirname:
            self.set_path(dirname)
        elif filename:
            self.selected(filename)