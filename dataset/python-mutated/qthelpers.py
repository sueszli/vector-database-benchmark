"""Qt utilities."""
import configparser
import functools
from math import pi
import logging
import os
import os.path as osp
import re
import sys
import types
from urllib.parse import unquote
from qtpy.compat import from_qvariant, to_qvariant
from qtpy.QtCore import QEvent, QLibraryInfo, QLocale, QObject, Qt, QTimer, QTranslator, QUrl, Signal, Slot
from qtpy.QtGui import QDesktopServices, QFontMetrics, QKeyEvent, QKeySequence, QPixmap
from qtpy.QtWidgets import QAction, QApplication, QDialog, QHBoxLayout, QLabel, QLineEdit, QMenu, QPlainTextEdit, QProxyStyle, QPushButton, QStyle, QToolButton, QVBoxLayout, QWidget
from spyder.api.config.fonts import SpyderFontsMixin, SpyderFontType
from spyder.api.config.mixins import SpyderConfigurationAccessor
from spyder.config.base import is_conda_based_app
from spyder.config.manager import CONF
from spyder.py3compat import is_text_string, to_text_string
from spyder.utils.icon_manager import ima
from spyder.utils import programs
from spyder.utils.image_path_manager import get_image_path
from spyder.utils.palette import QStylePalette
from spyder.utils.registries import ACTION_REGISTRY, TOOLBUTTON_REGISTRY
from spyder.widgets.waitingspinner import QWaitingSpinner
if sys.platform == 'darwin' and (not is_conda_based_app()):
    import applaunchservices as als
logger = logging.getLogger(__name__)
MENU_SEPARATOR = None

def start_file(filename):
    if False:
        while True:
            i = 10
    '\n    Generalized os.startfile for all platforms supported by Qt\n\n    This function is simply wrapping QDesktopServices.openUrl\n\n    Returns True if successful, otherwise returns False.\n    '
    url = QUrl()
    url.setUrl(filename)
    return QDesktopServices.openUrl(url)

def get_image_label(name, default='not_found'):
    if False:
        while True:
            i = 10
    'Return image inside a QLabel object'
    label = QLabel()
    label.setPixmap(QPixmap(get_image_path(name, default)))
    return label

def get_origin_filename():
    if False:
        i = 10
        return i + 15
    'Return the filename at the top of the stack'
    f = sys._getframe()
    while f.f_back is not None:
        f = f.f_back
    return f.f_code.co_filename

def qapplication(translate=True, test_time=3):
    if False:
        return 10
    "\n    Return QApplication instance\n    Creates it if it doesn't already exist\n\n    test_time: Time to maintain open the application when testing. It's given\n    in seconds\n    "
    app = QApplication.instance()
    if app is None:
        app = SpyderApplication(['Spyder', '--no-sandbox'])
        app.setApplicationName('Spyder')
    if sys.platform == 'darwin' and (not is_conda_based_app()) and CONF.get('main', 'mac_open_file', False):
        register_app_launchservices()
    if translate:
        install_translator(app)
    test_ci = os.environ.get('TEST_CI_WIDGETS', None)
    if test_ci is not None:
        timer_shutdown = QTimer(app)
        timer_shutdown.timeout.connect(app.quit)
        timer_shutdown.start(test_time * 1000)
    return app

def file_uri(fname):
    if False:
        print('Hello World!')
    'Select the right file uri scheme according to the operating system'
    if os.name == 'nt':
        if re.search('^[a-zA-Z]:', fname):
            return 'file:///' + fname
        else:
            return 'file://' + fname
    else:
        return 'file://' + fname
QT_TRANSLATOR = None

def install_translator(qapp):
    if False:
        while True:
            i = 10
    'Install Qt translator to the QApplication instance'
    global QT_TRANSLATOR
    if QT_TRANSLATOR is None:
        qt_translator = QTranslator()
        if qt_translator.load('qt_' + QLocale.system().name(), QLibraryInfo.location(QLibraryInfo.TranslationsPath)):
            QT_TRANSLATOR = qt_translator
    if QT_TRANSLATOR is not None:
        qapp.installTranslator(QT_TRANSLATOR)

def keybinding(attr):
    if False:
        return 10
    'Return keybinding'
    ks = getattr(QKeySequence, attr)
    return from_qvariant(QKeySequence.keyBindings(ks)[0], str)

def keyevent_to_keysequence_str(event):
    if False:
        for i in range(10):
            print('nop')
    'Get key sequence corresponding to a key event as a string.'
    try:
        return QKeySequence(event.modifiers() | event.key()).toString()
    except TypeError:
        key = event.key()
        alt = event.modifiers() & Qt.AltModifier
        shift = event.modifiers() & Qt.ShiftModifier
        ctrl = event.modifiers() & Qt.ControlModifier
        meta = event.modifiers() & Qt.MetaModifier
        key_sequence = key
        if ctrl:
            key_sequence += Qt.CTRL
        if shift:
            key_sequence += Qt.SHIFT
        if alt:
            key_sequence += Qt.ALT
        if meta:
            key_sequence += Qt.META
        return QKeySequence(key_sequence).toString()

def _process_mime_path(path, extlist):
    if False:
        i = 10
        return i + 15
    if path.startswith('file://'):
        if os.name == 'nt':
            if path.startswith('file:///'):
                path = path[8:]
            else:
                path = path[5:]
        else:
            path = path[7:]
    path = path.replace('\\', os.sep)
    if osp.exists(path):
        if extlist is None or osp.splitext(path)[1] in extlist:
            return path

def mimedata2url(source, extlist=None):
    if False:
        return 10
    "\n    Extract url list from MIME data\n    extlist: for example ('.py', '.pyw')\n    "
    pathlist = []
    if source.hasUrls():
        for url in source.urls():
            path = _process_mime_path(unquote(to_text_string(url.toString())), extlist)
            if path is not None:
                pathlist.append(path)
    elif source.hasText():
        for rawpath in to_text_string(source.text()).splitlines():
            path = _process_mime_path(rawpath, extlist)
            if path is not None:
                pathlist.append(path)
    if pathlist:
        return pathlist

def keyevent2tuple(event):
    if False:
        return 10
    'Convert QKeyEvent instance into a tuple'
    return (event.type(), event.key(), event.modifiers(), event.text(), event.isAutoRepeat(), event.count())

def tuple2keyevent(past_event):
    if False:
        return 10
    'Convert tuple into a QKeyEvent instance'
    return QKeyEvent(*past_event)

def restore_keyevent(event):
    if False:
        while True:
            i = 10
    if isinstance(event, tuple):
        (_, key, modifiers, text, _, _) = event
        event = tuple2keyevent(event)
    else:
        text = event.text()
        modifiers = event.modifiers()
        key = event.key()
    ctrl = modifiers & Qt.ControlModifier
    shift = modifiers & Qt.ShiftModifier
    return (event, text, key, ctrl, shift)

def create_toolbutton(parent, text=None, shortcut=None, icon=None, tip=None, toggled=None, triggered=None, autoraise=True, text_beside_icon=False, section=None, option=None, id_=None, plugin=None, context_name=None, register_toolbutton=False):
    if False:
        print('Hello World!')
    'Create a QToolButton'
    button = QToolButton(parent)
    if text is not None:
        button.setText(text)
    if icon is not None:
        if is_text_string(icon):
            icon = ima.get_icon(icon)
        button.setIcon(icon)
    if text is not None or tip is not None:
        button.setToolTip(text if tip is None else tip)
    if text_beside_icon:
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    button.setAutoRaise(autoraise)
    if triggered is not None:
        button.clicked.connect(triggered)
    if toggled is not None:
        setup_toggled_action(button, toggled, section, option)
    if shortcut is not None:
        button.setShortcut(shortcut)
    if id_ is not None:
        button.ID = id_
    if register_toolbutton:
        TOOLBUTTON_REGISTRY.register_reference(button, id_, plugin, context_name)
    return button

def create_waitspinner(size=32, n=11, parent=None):
    if False:
        i = 10
        return i + 15
    '\n    Create a wait spinner with the specified size built with n circling dots.\n    '
    dot_padding = 1
    dot_size = (pi * size - n * dot_padding) / (n + pi)
    inner_radius = (size - 2 * dot_size) / 2
    spinner = QWaitingSpinner(parent, centerOnParent=False)
    spinner.setTrailSizeDecreasing(True)
    spinner.setNumberOfLines(n)
    spinner.setLineLength(dot_size)
    spinner.setLineWidth(dot_size)
    spinner.setInnerRadius(inner_radius)
    spinner.setColor(QStylePalette.COLOR_TEXT_1)
    return spinner

def action2button(action, autoraise=True, text_beside_icon=False, parent=None, icon=None):
    if False:
        for i in range(10):
            print('nop')
    'Create a QToolButton directly from a QAction object'
    if parent is None:
        parent = action.parent()
    button = QToolButton(parent)
    button.setDefaultAction(action)
    button.setAutoRaise(autoraise)
    if text_beside_icon:
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    if icon:
        action.setIcon(icon)
    return button

def toggle_actions(actions, enable):
    if False:
        print('Hello World!')
    'Enable/disable actions'
    if actions is not None:
        for action in actions:
            if action is not None:
                action.setEnabled(enable)

def create_action(parent, text, shortcut=None, icon=None, tip=None, toggled=None, triggered=None, data=None, menurole=None, context=Qt.WindowShortcut, option=None, section=None, id_=None, plugin=None, context_name=None, register_action=False, overwrite=False):
    if False:
        print('Hello World!')
    'Create a QAction'
    action = SpyderAction(text, parent, action_id=id_)
    if triggered is not None:
        action.triggered.connect(triggered)
    if toggled is not None:
        setup_toggled_action(action, toggled, section, option)
    if icon is not None:
        if is_text_string(icon):
            icon = ima.get_icon(icon)
        action.setIcon(icon)
    if tip is not None:
        action.setToolTip(tip)
        action.setStatusTip(tip)
    if data is not None:
        action.setData(to_qvariant(data))
    if menurole is not None:
        action.setMenuRole(menurole)
    if sys.platform == 'darwin':
        action._shown_shortcut = None
        if context == Qt.WidgetShortcut:
            if shortcut is not None:
                action._shown_shortcut = shortcut
            else:
                action._shown_shortcut = 'missing'
        else:
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.setShortcutContext(context)
    else:
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.setShortcutContext(context)
    if register_action:
        ACTION_REGISTRY.register_reference(action, id_, plugin, context_name, overwrite)
    return action

def setup_toggled_action(action, toggled, section, option):
    if False:
        for i in range(10):
            print('nop')
    '\n    Setup a checkable action and wrap the toggle function to receive\n    configuration.\n    '
    toggled = wrap_toggled(toggled, section, option)
    action.toggled.connect(toggled)
    action.setCheckable(True)
    if section is not None and option is not None:
        CONF.observe_configuration(action, section, option)
        add_configuration_update(action)

def wrap_toggled(toggled, section, option):
    if False:
        i = 10
        return i + 15
    'Wrap a toggle function to set a value on a configuration option.'
    if section is not None and option is not None:

        @functools.wraps(toggled)
        def wrapped_toggled(value):
            if False:
                print('Hello World!')
            CONF.set(section, option, value, recursive_notification=True)
            toggled(value)
        return wrapped_toggled
    return toggled

def add_configuration_update(action):
    if False:
        while True:
            i = 10
    'Add on_configuration_change to a SpyderAction that depends on CONF.'

    def on_configuration_change(self, _option, _section, value):
        if False:
            i = 10
            return i + 15
        self.blockSignals(True)
        self.setChecked(value)
        self.blockSignals(False)
    method = types.MethodType(on_configuration_change, action)
    setattr(action, 'on_configuration_change', method)

def add_shortcut_to_tooltip(action, context, name):
    if False:
        print('Hello World!')
    'Add the shortcut associated with a given action to its tooltip'
    if not hasattr(action, '_tooltip_backup'):
        action._tooltip_backup = action.toolTip()
    try:
        shortcut = CONF.get_shortcut(context=context, name=name)
    except (configparser.NoSectionError, configparser.NoOptionError):
        shortcut = None
    if shortcut:
        keyseq = QKeySequence(shortcut)
        string = keyseq.toString(QKeySequence.NativeText)
        action.setToolTip(u'{0} ({1})'.format(action._tooltip_backup, string))

def add_actions(target, actions, insert_before=None):
    if False:
        return 10
    'Add actions to a QMenu or a QToolBar.'
    previous_action = None
    target_actions = list(target.actions())
    if target_actions:
        previous_action = target_actions[-1]
        if previous_action.isSeparator():
            previous_action = None
    for action in actions:
        if action is None and previous_action is not None:
            if insert_before is None:
                target.addSeparator()
            else:
                target.insertSeparator(insert_before)
        elif isinstance(action, QMenu):
            if insert_before is None:
                target.addMenu(action)
            else:
                target.insertMenu(insert_before, action)
        elif isinstance(action, QAction):
            if insert_before is None:
                try:
                    target.addAction(action)
                except RuntimeError:
                    continue
            else:
                target.insertAction(insert_before, action)
        previous_action = action

def get_item_user_text(item):
    if False:
        print('Hello World!')
    'Get QTreeWidgetItem user role string'
    return from_qvariant(item.data(0, Qt.UserRole), to_text_string)

def set_item_user_text(item, text):
    if False:
        while True:
            i = 10
    'Set QTreeWidgetItem user role string'
    item.setData(0, Qt.UserRole, to_qvariant(text))

def create_bookmark_action(parent, url, title, icon=None, shortcut=None):
    if False:
        while True:
            i = 10
    'Create bookmark action'

    @Slot()
    def open_url():
        if False:
            i = 10
            return i + 15
        return start_file(url)
    return create_action(parent, title, shortcut=shortcut, icon=icon, triggered=open_url)

def create_module_bookmark_actions(parent, bookmarks):
    if False:
        i = 10
        return i + 15
    '\n    Create bookmark actions depending on module installation:\n    bookmarks = ((module_name, url, title), ...)\n    '
    actions = []
    for (key, url, title) in bookmarks:
        create_act = True
        if key == 'winpython':
            if not programs.is_module_installed(key):
                create_act = False
        if create_act:
            act = create_bookmark_action(parent, url, title)
            actions.append(act)
    return actions

def create_program_action(parent, text, name, icon=None, nt_name=None):
    if False:
        return 10
    'Create action to run a program'
    if is_text_string(icon):
        icon = ima.get_icon(icon)
    if os.name == 'nt' and nt_name is not None:
        name = nt_name
    path = programs.find_program(name)
    if path is not None:
        return create_action(parent, text, icon=icon, triggered=lambda : programs.run_program(name))

def create_python_script_action(parent, text, icon, package, module, args=[]):
    if False:
        i = 10
        return i + 15
    'Create action to run a GUI based Python script'
    if is_text_string(icon):
        icon = ima.get_icon(icon)
    if programs.python_script_exists(package, module):
        return create_action(parent, text, icon=icon, triggered=lambda : programs.run_python_script(package, module, args))

class DialogManager(QObject):
    """
    Object that keep references to non-modal dialog boxes for another QObject,
    typically a QMainWindow or any kind of QWidget
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        QObject.__init__(self)
        self.dialogs = {}

    def show(self, dialog):
        if False:
            return 10
        'Generic method to show a non-modal dialog and keep reference\n        to the Qt C++ object'
        for dlg in list(self.dialogs.values()):
            if to_text_string(dlg.windowTitle()) == to_text_string(dialog.windowTitle()):
                dlg.show()
                dlg.raise_()
                break
        else:
            dialog.show()
            self.dialogs[id(dialog)] = dialog
            dialog.accepted.connect(lambda eid=id(dialog): self.dialog_finished(eid))
            dialog.rejected.connect(lambda eid=id(dialog): self.dialog_finished(eid))

    def dialog_finished(self, dialog_id):
        if False:
            i = 10
            return i + 15
        'Manage non-modal dialog boxes'
        return self.dialogs.pop(dialog_id)

    def close_all(self):
        if False:
            while True:
                i = 10
        'Close all opened dialog boxes'
        for dlg in list(self.dialogs.values()):
            dlg.reject()

def get_filetype_icon(fname):
    if False:
        print('Hello World!')
    'Return file type icon'
    ext = osp.splitext(fname)[1]
    if ext.startswith('.'):
        ext = ext[1:]
    return ima.get_icon('%s.png' % ext, ima.icon('FileIcon'))

class SpyderAction(QAction):
    """Spyder QAction class wrapper to handle cross platform patches."""

    def __init__(self, *args, action_id=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Spyder QAction class wrapper to handle cross platform patches.'
        super(SpyderAction, self).__init__(*args, **kwargs)
        self.action_id = action_id
        if sys.platform == 'darwin':
            self.setIconVisibleInMenu(False)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return "SpyderAction('{0}')".format(self.text())

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return "SpyderAction('{0}')".format(self.text())

class ShowStdIcons(QWidget):
    """
    Dialog showing standard icons
    """

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        QWidget.__init__(self, parent)
        layout = QHBoxLayout()
        row_nb = 14
        cindex = 0
        for child in dir(QStyle):
            if child.startswith('SP_'):
                if cindex == 0:
                    col_layout = QVBoxLayout()
                icon_layout = QHBoxLayout()
                icon = ima.get_std_icon(child)
                label = QLabel()
                label.setPixmap(icon.pixmap(32, 32))
                icon_layout.addWidget(label)
                icon_layout.addWidget(QLineEdit(child.replace('SP_', '')))
                col_layout.addLayout(icon_layout)
                cindex = (cindex + 1) % row_nb
                if cindex == 0:
                    layout.addLayout(col_layout)
        self.setLayout(layout)
        self.setWindowTitle('Standard Platform Icons')
        self.setWindowIcon(ima.get_std_icon('TitleBarMenuButton'))

def show_std_icons():
    if False:
        return 10
    '\n    Show all standard Icons\n    '
    app = qapplication()
    dialog = ShowStdIcons(None)
    dialog.show()
    sys.exit(app.exec_())

def calc_tools_spacing(tools_layout):
    if False:
        i = 10
        return i + 15
    "\n    Return a spacing (int) or None if we don't have the appropriate metrics\n    to calculate the spacing.\n\n    We're trying to adapt the spacing below the tools_layout spacing so that\n    the main_widget has the same vertical position as the editor widgets\n    (which have tabs above).\n\n    The required spacing is\n\n        spacing = tabbar_height - tools_height + offset\n\n    where the tabbar_heights were empirically determined for a combination of\n    operating systems and styles. Offsets were manually adjusted, so that the\n    heights of main_widgets and editor widgets match. This is probably\n    caused by a still not understood element of the layout and style metrics.\n    "
    metrics = {'nt.fusion': (32, 0), 'nt.windowsvista': (21, 3), 'nt.windowsxp': (24, 0), 'nt.windows': (21, 3), 'posix.breeze': (28, -1), 'posix.oxygen': (38, -2), 'posix.qtcurve': (27, 0), 'posix.windows': (26, 0), 'posix.fusion': (32, 0)}
    style_name = qapplication().style().property('name')
    key = '%s.%s' % (os.name, style_name)
    if key in metrics:
        (tabbar_height, offset) = metrics[key]
        tools_height = tools_layout.sizeHint().height()
        spacing = tabbar_height - tools_height + offset
        return max(spacing, 0)

def create_plugin_layout(tools_layout, main_widget=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a layout for a set of controls above a main widget. This is a\n    standard layout for many plugin panes (even though, it's currently\n    more often applied not to the pane itself but with in the one widget\n    contained in the pane.\n\n    tools_layout: a layout containing the top toolbar\n    main_widget: the main widget. Can be None, if you want to add this\n        manually later on.\n    "
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    spacing = calc_tools_spacing(tools_layout)
    if spacing is not None:
        layout.setSpacing(spacing)
    layout.addLayout(tools_layout)
    if main_widget is not None:
        layout.addWidget(main_widget)
    return layout

def set_menu_icons(menu, state):
    if False:
        i = 10
        return i + 15
    'Show/hide icons for menu actions.'
    menu_actions = menu.actions()
    for action in menu_actions:
        try:
            if action.menu() is not None:
                set_menu_icons(action.menu(), state)
            elif action.isSeparator():
                continue
            else:
                action.setIconVisibleInMenu(state)
        except RuntimeError:
            continue

class SpyderProxyStyle(QProxyStyle):
    """Style proxy to adjust qdarkstyle issues."""

    def styleHint(self, hint, option=0, widget=0, returnData=0):
        if False:
            for i in range(10):
                print('nop')
        'Override Qt method.'
        if hint == QStyle.SH_ComboBox_Popup:
            return 0
        return QProxyStyle.styleHint(self, hint, option, widget, returnData)

class QInputDialogMultiline(QDialog):
    """
    Build a replica interface of QInputDialog.getMultilineText.

    Based on: https://stackoverflow.com/a/58823967
    """

    def __init__(self, parent, title, label, text='', **kwargs):
        if False:
            print('Hello World!')
        super(QInputDialogMultiline, self).__init__(parent, **kwargs)
        if title is not None:
            self.setWindowTitle(title)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel(label))
        self.text_edit = QPlainTextEdit()
        self.layout().addWidget(self.text_edit)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton('OK')
        button_layout.addWidget(ok_button)
        cancel_button = QPushButton('Cancel')
        button_layout.addWidget(cancel_button)
        self.layout().addLayout(button_layout)
        self.text_edit.setPlainText(text)
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

class SpyderApplication(QApplication, SpyderConfigurationAccessor, SpyderFontsMixin):
    """Subclass with several adjustments for Spyder."""
    sig_open_external_file = Signal(str)

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        QApplication.__init__(self, *args)
        self._never_shown = True
        self._has_started = False
        self._pending_file_open = []
        self._original_handlers = {}

    def event(self, event):
        if False:
            print('Hello World!')
        if sys.platform == 'darwin' and event.type() == QEvent.FileOpen:
            fname = str(event.file())
            if sys.argv and sys.argv[0] == fname:
                pass
            elif self._has_started:
                self.sig_open_external_file.emit(fname)
            else:
                self._pending_file_open.append(fname)
        return QApplication.event(self, event)

    def set_font(self):
        if False:
            while True:
                i = 10
        'Set font for the entire application.'
        if self.get_conf('use_system_font', section='appearance'):
            family = self.font().family()
            size = self.font().pointSize()
            self.set_conf('app_font/family', family, section='appearance')
            self.set_conf('app_font/size', size, section='appearance')
        else:
            family = self.get_conf('app_font/family', section='appearance')
            size = self.get_conf('app_font/size', section='appearance')
        app_font = self.font()
        app_font.setFamily(family)
        app_font.setPointSize(size)
        self.set_monospace_interface_font(app_font)
        self.setFont(app_font)

    def set_monospace_interface_font(self, app_font):
        if False:
            while True:
                i = 10
        '\n        Set monospace interface font in our config system according to the app\n        one.\n        '
        x_height = QFontMetrics(app_font).xHeight()
        size = app_font.pointSize()
        plain_font = self.get_font(SpyderFontType.Monospace)
        plain_font.setPointSize(size)
        monospace_size = size
        while QFontMetrics(plain_font).xHeight() != x_height and size - 4 < monospace_size < size + 4:
            if QFontMetrics(plain_font).xHeight() > x_height:
                monospace_size -= 1
            else:
                monospace_size += 1
            plain_font.setPointSize(monospace_size)
        if not size - 4 < monospace_size < size + 4:
            monospace_size = size
        self.set_conf('monospace_app_font/family', plain_font.family(), section='appearance')
        self.set_conf('monospace_app_font/size', monospace_size, section='appearance')

def restore_launchservices():
    if False:
        return 10
    'Restore LaunchServices to the previous state'
    app = QApplication.instance()
    for (key, handler) in app._original_handlers.items():
        (UTI, role) = key
        als.set_UTI_handler(UTI, role, handler)

def register_app_launchservices(uniform_type_identifier='public.python-script', role='editor'):
    if False:
        while True:
            i = 10
    '\n    Register app to the Apple launch services so it can open Python files\n    '
    app = QApplication.instance()
    old_handler = als.get_UTI_handler(uniform_type_identifier, role)
    app._original_handlers[uniform_type_identifier, role] = old_handler
    app.aboutToQuit.connect(restore_launchservices)
    if not app._never_shown:
        bundle_identifier = als.get_bundle_identifier()
        als.set_UTI_handler(uniform_type_identifier, role, bundle_identifier)
        return

    def handle_applicationStateChanged(state):
        if False:
            i = 10
            return i + 15
        if state == Qt.ApplicationActive and app._never_shown:
            app._never_shown = False
            bundle_identifier = als.get_bundle_identifier()
            als.set_UTI_handler(uniform_type_identifier, role, bundle_identifier)
    app.applicationStateChanged.connect(handle_applicationStateChanged)
if __name__ == '__main__':
    show_std_icons()