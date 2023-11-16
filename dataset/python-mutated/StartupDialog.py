import os
import pathlib
from typing import Optional
from qtpy.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QRadioButton, QButtonGroup, QLabel, QFormLayout, QComboBox, QDialogButtonBox, QCheckBox, QListWidget, QListWidgetItem, QMessageBox, QStyleOptionFrame, QStyle, QLineEdit
from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QIcon, QPainter
from ryven.main.args_parser import unparse_sys_args
from ryven.main.config import Config
from ryven.main.utils import abs_path_from_package_dir, abs_path_from_ryven_dir, ryven_dir_path
from ryven.main.packages.nodes_package import process_nodes_packages
from ryven.gui.styling.window_theme import apply_stylesheet
LBL_CREATE_PROJECT = '<create new project>'
LBL_DEFAULT_FLOW_THEME = '<default>'

class ElideLabel(QLabel):
    """A QLabel with ellipsis, if the text is too long to be fully displayed.

    See:
        https://stackoverflow.com/questions/68092087/one-line-elideable-qlabel#answer-68092991

    Copyright (C) 2021  https://github.com/MaurizioB/
    """
    _elideMode = Qt.ElideMiddle

    def setText(self, label, *args, **kwargs):
        if False:
            return 10
        'Sets text and tooltip.'
        s = str(label)
        super().setText(s, *args, **kwargs)
        self.setToolTip(s)

    def elideMode(self):
        if False:
            return 10
        return self._elideMode

    def setElideMode(self, mode):
        if False:
            for i in range(10):
                print('nop')
        if self._elideMode != mode and mode != Qt.ElideNone:
            self._elideMode = mode
            self.updateGeometry()

    def minimumSizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sizeHint()

    def sizeHint(self):
        if False:
            return 10
        hint = self.fontMetrics().boundingRect(self.text()).size()
        (l, t, r, b) = self.getContentsMargins()
        margin = self.margin() * 2
        return QSize(min(100, hint.width()) + l + r + margin, min(self.fontMetrics().height(), hint.height()) + t + b + margin)

    def paintEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        qp = QPainter(self)
        opt = QStyleOptionFrame()
        self.initStyleOption(opt)
        self.style().drawControl(QStyle.CE_ShapedFrame, opt, qp, self)
        (l, t, r, b) = self.getContentsMargins()
        margin = self.margin()
        try:
            m = self.fontMetrics().horizontalAdvance('x') / 2 - margin
        except:
            m = self.fontMetrics().width('x') / 2 - margin
        r = self.contentsRect().adjusted(margin + m, margin, -(margin + m), -margin)
        qp.drawText(r, self.alignment(), self.fontMetrics().elidedText(self.text(), self.elideMode(), r.width()))

class ShowCommandDialog(QDialog):

    def __init__(self, command, config, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        layout = QVBoxLayout()
        command_lineedit = QLineEdit(command)
        layout.addWidget(command_lineedit)
        self.config_text_edit = QTextEdit()
        self.config_text_edit.setText(config)
        layout.addWidget(self.config_text_edit)
        buttons_layout = QHBoxLayout()
        save_config_button = QPushButton('save config')
        save_config_button.clicked.connect(self.on_save_config_clicked)
        buttons_layout.addWidget(save_config_button)
        close_button = QPushButton('close')
        close_button.clicked.connect(self.accept)
        buttons_layout.addWidget(close_button)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def on_save_config_clicked(self):
        if False:
            i = 10
            return i + 15
        config = self.config_text_edit.toPlainText()
        file = QFileDialog.getSaveFileName(self, 'select config file', ryven_dir_path(), 'ryven config files (*.cfg)')[0]
        if file != '':
            p = pathlib.Path(file)
            with open(p, 'w') as f:
                f.write(config)

class StartupDialog(QDialog):
    """The welcome dialog.

    The user can choose between creating a new project and loading a saved or
    example project. When a project is loaded, it scans for validity of all
    the required packages for the project, and in case some paths are invalid,
    they are shown in the dialog. The user than can autodiscover those missing
    packages or cherry-pick by choosing paths manually.

    The user can also set some common configuration options, and can generate
    an analogous command and config file and save the configuration in it.
    """

    def __init__(self, config: Config, parent=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the `StartupDialog` class.\n\n        Parameters\n        ----------\n        config : Config\n            The global configuration, parsed from command line or run() function\n            interface.\n            Notice that this class operates on args directly, so all values\n            are either of primitive type (including strings), except paths\n            which are pathlib.Path objects. Translation to NodePackage objects,\n            WindowTheme objects etc. will happen at a later stage, not here.\n        parent : QWidget, optional\n            The parent `QWidget`.\n            The default is `None`.\n\n        Returns\n        -------\n        None.\n\n        '
        super().__init__(parent)
        self.conf = config
        layout = QVBoxLayout()
        info_text_edit = QTextEdit()
        info_text_edit.setHtml(f'''\n            <div style="font-family: Corbel; font-size: large;">\n                <img style="float:right;" height=120 src="{abs_path_from_package_dir('resources/pics/Ryven_icon_blurred.png')}"\n                >Ryven is not a stable piece of software, it's experimental, and nothing is\n                guaranteed to work as expected. Make sure to save frequently, and to\n                different files. If you spot an issue, please report it on the \n                <a href="https://github.com/leon-thomm/ryven/issues">GitHub page</a>.\n                <br><br>\n                Ryven doesn't come with batteries (nodes) included. It provides some\n                small examples but nothing more. Development of large node packages\n                is not part of the Ryven editor project itself.\n                See the GitHub for a quickstart guide.\n                Cheers.\n            </div>\n        ''')
        info_text_edit.setReadOnly(True)
        layout.addWidget(info_text_edit)
        fbox = QFormLayout()
        project_label = QLabel('Project:')
        project_layout = QVBoxLayout()
        self.project_name = ElideLabel()
        if self.conf.project is not None:
            self.project_name.setText(str(config.project))
        else:
            self.project_name.setText(LBL_CREATE_PROJECT)
        project_layout.addWidget(self.project_name)
        project_buttons_widget = QDialogButtonBox()
        self.create_project_button = QPushButton('New')
        self.create_project_button.setToolTip('Create a new project')
        self.create_project_button.setDefault(True)
        self.create_project_button.clicked.connect(self.on_create_project_button_clicked)
        project_buttons_widget.addButton(self.create_project_button, QDialogButtonBox.ActionRole)
        load_project_button = QPushButton('Load')
        load_project_button.setToolTip('Load an existing project')
        load_project_button.clicked.connect(self.on_load_project_button_clicked)
        project_buttons_widget.addButton(load_project_button, QDialogButtonBox.ActionRole)
        load_example_project_button = QPushButton('Example')
        load_example_project_button.setToolTip('Load a Ryven example')
        load_example_project_button.clicked.connect(self.on_load_example_project_button_clicked)
        project_buttons_widget.addButton(load_example_project_button, QDialogButtonBox.ActionRole)
        project_layout.addWidget(project_buttons_widget)
        fbox.addRow(project_label, project_layout)
        packages_label = QLabel('Nodes packages:')
        packages_layout = QVBoxLayout()
        packages_sublayout = QHBoxLayout()
        packages_imported_layout = QVBoxLayout()
        label_imported = QLabel('Imported:')
        label_imported.setToolTip('Nodes packages which are required by the project and are found')
        label_imported.setAlignment(Qt.AlignCenter)
        packages_imported_layout.addWidget(label_imported)
        self.imported_list_widget = QListWidget()
        packages_imported_layout.addWidget(self.imported_list_widget)
        packages_sublayout.addLayout(packages_imported_layout)
        packages_missing_layout = QVBoxLayout()
        label_missing = QLabel('Missing:')
        label_missing.setToolTip('Nodes packages which are required by the project but could not be found')
        label_missing.setAlignment(Qt.AlignCenter)
        packages_missing_layout.addWidget(label_missing)
        self.missing_list_widget = QListWidget()
        packages_missing_layout.addWidget(self.missing_list_widget)
        packages_sublayout.addLayout(packages_missing_layout)
        packages_manually_layout = QVBoxLayout()
        label_manually = QLabel('Manually imported:')
        label_manually.setToolTip('Nodes packages which are manually imported\nThey will override the packages required by the project\nAdditional packages can be imported later …')
        label_manually.setAlignment(Qt.AlignCenter)
        packages_manually_layout.addWidget(label_manually)
        self.manually_list_widget = QListWidget()
        self.manually_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.manually_list_widget.itemSelectionChanged.connect(self.on_packages_manually_selection)
        packages_manually_layout.addWidget(self.manually_list_widget)
        packages_sublayout.addLayout(packages_manually_layout)
        packages_layout.addLayout(packages_sublayout)
        packages_buttons_widget = QDialogButtonBox()
        self.autodiscover_packages_button = QPushButton('Find')
        self.autodiscover_packages_button.setToolTip('Automatically find and import missing packages')
        self.autodiscover_packages_button.clicked.connect(self.on_autodiscover_package_clicked)
        packages_buttons_widget.addButton(self.autodiscover_packages_button, QDialogButtonBox.ActionRole)
        self.autodiscover_packages_button.setEnabled(False)
        import_package_button = QPushButton('Import')
        import_package_button.setToolTip('Manually load a nodes package')
        import_package_button.clicked.connect(self.on_import_package_clicked)
        packages_buttons_widget.addButton(import_package_button, QDialogButtonBox.ActionRole)
        self.remove_packages_button = QPushButton('Remove')
        self.remove_packages_button.setToolTip('Remove manually imported nodes packages')
        self.remove_packages_button.clicked.connect(self.on_remove_packages_clicked)
        self.remove_packages_button.setEnabled(False)
        packages_buttons_widget.addButton(self.remove_packages_button, QDialogButtonBox.ActionRole)
        self.clear_packages_button = QPushButton('Clear')
        self.clear_packages_button.setToolTip('Clear the list of manually imported nodes packages ')
        self.clear_packages_button.clicked.connect(self.on_clear_packages_clicked)
        self.clear_packages_button.setEnabled(False)
        packages_buttons_widget.addButton(self.clear_packages_button, QDialogButtonBox.ActionRole)
        packages_layout.addWidget(packages_buttons_widget)
        fbox.addRow(packages_label, packages_layout)
        windowtheme_label = QLabel('Window theme:')
        windowtheme_layout = QHBoxLayout()
        windowtheme_button_group = QButtonGroup(windowtheme_layout)
        self.window_theme_rbs = {theme: QRadioButton(theme) for theme in self.conf.get_available_window_themes()}
        for rb in self.window_theme_rbs.values():
            windowtheme_button_group.addButton(rb)
            windowtheme_layout.addWidget(rb)
        windowtheme_button_group.buttonToggled.connect(self.on_window_theme_toggled)
        fbox.addRow(windowtheme_label, windowtheme_layout)
        flowtheme_label = QLabel('Flow theme:')
        flowtheme_widget = QComboBox()
        flowtheme_widget.setToolTip('Select the theme of the flow display\nCan also be changed later …')
        flowtheme_widget.addItems([LBL_DEFAULT_FLOW_THEME] + list(self.conf.get_available_flow_themes()))
        flowtheme_widget.insertSeparator(1)
        flowtheme_widget.currentTextChanged.connect(self.on_flow_theme_selected)
        fbox.addRow(flowtheme_label, flowtheme_widget)
        performance_label = QLabel('Performance mode:')
        performance_layout = QHBoxLayout()
        performance_button_group = QButtonGroup(performance_layout)
        self.perf_mode_rbs = {mode: QRadioButton(mode) for mode in self.conf.get_available_performance_modes()}
        for rb in self.perf_mode_rbs.values():
            performance_button_group.addButton(rb)
            performance_layout.addWidget(rb)
        performance_button_group.buttonToggled.connect(self.on_performance_toggled)
        fbox.addRow(performance_label, performance_layout)
        animations_label = QLabel('Animations:')
        animations_cb = QCheckBox('Animations')
        animations_cb.toggled.connect(self.on_animations_toggled)
        fbox.addRow(animations_label, animations_cb)
        title_label = QLabel('Window title:')
        self.title_lineedit = QLineEdit()
        self.title_lineedit.textChanged.connect(self.on_title_changed)
        fbox.addRow(title_label, self.title_lineedit)
        verbose_output_label = QLabel('Verbose:')
        verbose_output_cb = QCheckBox('Enable verbose output')
        verbose_output_cb.setToolTip(f'Choose whether verbose output should be displayed. \n            Verbose output prevents stdout and stderr from being\n            displayed in the in-editor console, that usually means\n            all output goes to the terminal from which Ryven was\n            launched. Also, it causes lots of debug info to be \n            printed.')
        verbose_output_cb.toggled.connect(self.on_verbose_toggled)
        fbox.addRow(verbose_output_label, verbose_output_cb)
        layout.addLayout(fbox)
        buttons_layout = QHBoxLayout()
        gen_config_button = QPushButton('generate / save config')
        gen_config_button.clicked.connect(self.gen_config_clicked)
        buttons_layout.addWidget(gen_config_button)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons_layout.addWidget(buttons)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        self.load_project(self.conf.project)
        self.update_packages_lists()
        for (theme, rb) in self.window_theme_rbs.items():
            rb.setChecked(self.conf.window_theme == theme)
        for (mode, rb) in self.perf_mode_rbs.items():
            rb.setChecked(self.conf.performance_mode == mode)
        animations_cb.setChecked(self.conf.animations)
        self.title_lineedit.setText(self.conf.window_title)
        if self.conf.flow_theme is not None:
            idx = flowtheme_widget.findText(self.conf.flow_theme)
        else:
            idx = flowtheme_widget.findText(LBL_DEFAULT_FLOW_THEME)
        flowtheme_widget.setCurrentIndex(idx)
        verbose_output_cb.setChecked(self.conf.verbose)
        self.setWindowTitle('Ryven')
        self.setWindowIcon(QIcon(abs_path_from_package_dir('resources/pics/Ryven_icon.png')))

    def on_create_project_button_clicked(self):
        if False:
            while True:
                i = 10
        "Call-back method, whenever the 'New' button was clicked."
        self.load_project(None)

    def on_load_project_button_clicked(self):
        if False:
            for i in range(10):
                print('nop')
        "Call-back method, whenever the 'Load' button was clicked."
        project_path = self.get_project(abs_path_from_ryven_dir('saves'))
        if project_path is not None:
            self.load_project(project_path)

    def on_load_example_project_button_clicked(self):
        if False:
            while True:
                i = 10
        "Call-back method, whenever the 'Example' button was clicked."
        project_path = self.get_project(abs_path_from_package_dir('examples_projects'), title='Select Ryven example')
        if project_path is not None:
            self.load_project(project_path)

    def on_packages_manually_selection(self):
        if False:
            print('Hello World!')
        'Call-back method, whenever a package in the manual list was selected.'
        if self.manually_list_widget.selectedItems():
            self.remove_packages_button.setEnabled(True)
        else:
            self.remove_packages_button.setEnabled(False)

    def on_autodiscover_package_clicked(self):
        if False:
            return 10
        "Call-back method, whenever the 'Find' button was clicked."
        self.auto_discover(pathlib.Path(ryven_dir_path(), 'nodes'))
        self.update_packages_lists()
        self.auto_discover(pathlib.Path(abs_path_from_package_dir('example_nodes')))
        self.update_packages_lists()

    def on_import_package_clicked(self):
        if False:
            print('Hello World!')
        "Call-back method, whenever the 'Import' button was clicked."
        file_name = QFileDialog.getOpenFileName(self, 'Select', abs_path_from_ryven_dir('packages'), 'ryven nodes package (nodes.py)')[0]
        if file_name:
            file_path = pathlib.Path(file_name)
            if file_path.exists():
                self.conf.nodes.add(file_path.parent)
                self.update_packages_lists()

    def on_remove_packages_clicked(self):
        if False:
            i = 10
            return i + 15
        "Call-back method, whenever the 'Remove' button was clicked."
        for item in self.manually_list_widget.selectedItems():
            package_name = item.text()
            for pkg_path in self.conf.nodes:
                if pkg_path.stem == package_name:
                    self.conf.nodes.remove(pkg_path)
                    break
        self.update_packages_lists()

    def on_clear_packages_clicked(self):
        if False:
            return 10
        "Call-back method, for when the 'Clear' button was clicked."
        self.conf.nodes.clear()
        self.update_packages_lists()

    def on_window_theme_toggled(self):
        if False:
            for i in range(10):
                print('nop')
        'Call-back method, whenever a window theme radio button was toggled.'
        for (theme, rb) in self.window_theme_rbs.items():
            if rb.isChecked():
                self.conf.window_theme = theme
                break
        apply_stylesheet(self.conf.window_theme)

    def on_flow_theme_selected(self, theme):
        if False:
            for i in range(10):
                print('nop')
        'Call-back method, whenever a new flow theme was selected.'
        if theme == LBL_DEFAULT_FLOW_THEME:
            self.conf.flow_theme = Config.flow_theme
        else:
            self.conf.flow_theme = theme

    def on_performance_toggled(self):
        if False:
            i = 10
            return i + 15
        'Call-back method, whenever a new performance mode was selected'
        for (mode, rb) in self.perf_mode_rbs.items():
            if rb.isChecked():
                self.conf.performance_mode = mode
                break

    def on_animations_toggled(self, check):
        if False:
            print('Hello World!')
        'Call-back method, whenever animations are enabled/disabled'
        self.conf.animations = check

    def on_title_changed(self, t):
        if False:
            for i in range(10):
                print('nop')
        'Call-back method, whenever the title was changed'
        self.conf.window_title = t

    def on_verbose_toggled(self, check):
        if False:
            while True:
                i = 10
        'Call-back method, whenever the verbose checkbox was toggled.'
        self.conf.verbose = check

    def get_project(self, base_dir: str, title='Select project file') -> Optional[pathlib.Path]:
        if False:
            for i in range(10):
                print('nop')
        "Get a project file from the user.\n\n        Parameters\n        ----------\n        base_dir : str|pathlib.Path\n            The initial directory shown in the file dialog.\n        title : str, optional\n            The title of the file dialog.\n            The default is 'Select project file'.\n\n        Returns\n        -------\n        file_path : pathlib.Path|None\n            The path of the selected file.\n\n        "
        file_name = QFileDialog.getOpenFileName(self, title, str(base_dir), 'JSON (*.json)')[0]
        if file_name:
            file_path = pathlib.Path(file_name)
            if file_path.exists():
                return file_path
        return None

    def load_project(self, project_path: Optional[pathlib.Path]):
        if False:
            print('Hello World!')
        "Load a project file.\n\n        It opens the project file and scans for all required node packages.\n        These are displayed in the imported packages list. All packages which\n        could not be found are displayed in the missing packages list.\n\n        Parameters\n        ----------\n        project_path : pathlib.Path\n            The project's file name to be loaded.\n\n        Returns\n        -------\n        None.\n\n        "
        self.imported_list_widget.clear()
        self.missing_list_widget.clear()
        if project_path is None:
            self.conf.project = None
            self.project_name.setText(LBL_CREATE_PROJECT)
            self.create_project_button.setEnabled(False)
        else:
            self.conf.project = project_path
            self.project_name.setText(project_path)
            self.create_project_button.setEnabled(True)
            (required_nodes, missing_nodes, _) = process_nodes_packages(project_path)
            item_flags = ~Qt.ItemIsSelectable & ~Qt.ItemIsEditable
            for node in sorted(required_nodes, key=lambda n: n.name):
                node_item = QListWidgetItem(node.name)
                node_item.setToolTip(node.directory)
                node_item.setFlags(item_flags)
                self.imported_list_widget.addItem(node_item)
            for node_path in sorted(missing_nodes, key=lambda p: p.name):
                node_item = QListWidgetItem(node_path.name)
                node_item.setToolTip(str(node_path))
                node_item.setFlags(item_flags)
                self.missing_list_widget.addItem(node_item)
        self.update_packages_lists()

    def auto_discover(self, packages_dir):
        if False:
            i = 10
            return i + 15
        'Automatically find and import missing packages.\n\n        Parameters\n        ----------\n        packages_dir : str|pathlib.Path\n            The directory under which packages should be searched.\n\n        Returns\n        -------\n        None.\n\n        '
        missing_packages = [self.missing_list_widget.item(i).text() for i in range(self.missing_list_widget.count())]
        for (top, dirs, files) in os.walk(packages_dir):
            path = pathlib.Path(top)
            if path.name in missing_packages:
                node_path = path.joinpath('nodes.py')
                if node_path.exists():
                    self.conf.nodes.add(path)

    def update_packages_lists(self):
        if False:
            while True:
                i = 10
        "Update the packages lists and buttons.\n\n        1. Mark all imported packages, if they were manually imported.\n        2. Mark all missing packages, if they were manually imported.\n        3. Repopulate the list with manually imported packages.\n        4. En/Disable 'Ok', 'Find' and 'Clear' buttons.\n        "
        for i in range(self.imported_list_widget.count()):
            node_item = self.imported_list_widget.item(i)
            font = node_item.font()
            for pkg_path in self.conf.nodes:
                if node_item.text() == pkg_path.stem:
                    font.setStrikeOut(True)
                    break
            else:
                font.setStrikeOut(False)
            node_item.setFont(font)
        missing_packages = False
        for i in range(self.missing_list_widget.count()):
            node_item = self.missing_list_widget.item(i)
            font = node_item.font()
            for pkg_path in self.conf.nodes:
                if node_item.text() == pkg_path.stem:
                    font.setStrikeOut(True)
                    break
            else:
                font.setStrikeOut(False)
                missing_packages = True
            node_item.setFont(font)
        self.manually_list_widget.clear()
        for pkg_path in sorted(self.conf.nodes):
            node_item = QListWidgetItem(pkg_path.stem)
            node_item.setToolTip(str(pkg_path))
            node_item.setFlags(~Qt.ItemIsEditable)
            self.manually_list_widget.addItem(node_item)
        if missing_packages:
            self.ok_button.setEnabled(False)
            self.ok_button.setToolTip('Import all missing packages first')
            self.autodiscover_packages_button.setEnabled(True)
        else:
            self.ok_button.setEnabled(True)
            self.ok_button.setToolTip(None)
            self.autodiscover_packages_button.setEnabled(False)
        self.clear_packages_button.setEnabled(bool(self.conf.nodes))

    def gen_config_clicked(self):
        if False:
            i = 10
            return i + 15
        'Generates the command analogous to the specified settings\n        as well as the according config file content.\n        Opens a dialog with option to save to config file.\n        '
        (command, config) = unparse_sys_args(self.conf)
        d = ShowCommandDialog(command, config)
        d.exec_()