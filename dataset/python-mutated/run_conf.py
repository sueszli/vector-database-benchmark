"""Profiler run executor configurations."""
import os.path as osp
from qtpy.compat import getexistingdirectory
from qtpy.QtWidgets import QGroupBox, QVBoxLayout, QGridLayout, QCheckBox, QLineEdit
from spyder.api.translations import _
from spyder.plugins.run.api import RunExecutorConfigurationGroup, Context, RunConfigurationMetadata
from spyder.utils.misc import getcwd_or_home

class ProfilerPyConfigurationGroup(RunExecutorConfigurationGroup):
    """External console Python run configuration options."""

    def __init__(self, parent, context: Context, input_extension: str, input_metadata: RunConfigurationMetadata):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent, context, input_extension, input_metadata)
        self.dir = None
        common_group = QGroupBox(_('Script settings'))
        common_layout = QGridLayout(common_group)
        self.clo_cb = QCheckBox(_('Command line options:'))
        common_layout.addWidget(self.clo_cb, 0, 0)
        self.clo_edit = QLineEdit()
        self.clo_cb.toggled.connect(self.clo_edit.setEnabled)
        self.clo_edit.setEnabled(False)
        common_layout.addWidget(self.clo_edit, 0, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(common_group)
        layout.addStretch(100)

    def select_directory(self):
        if False:
            for i in range(10):
                print('nop')
        'Select directory'
        basedir = str(self.wd_edit.text())
        if not osp.isdir(basedir):
            basedir = getcwd_or_home()
        directory = getexistingdirectory(self, _('Select directory'), basedir)
        if directory:
            self.wd_edit.setText(directory)
            self.dir = directory

    @staticmethod
    def get_default_configuration() -> dict:
        if False:
            print('Hello World!')
        return {'args_enabled': False, 'args': ''}

    def set_configuration(self, config: dict):
        if False:
            return 10
        args_enabled = config['args_enabled']
        args = config['args']
        self.clo_cb.setChecked(args_enabled)
        self.clo_edit.setText(args)

    def get_configuration(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'args_enabled': self.clo_cb.isChecked(), 'args': self.clo_edit.text()}