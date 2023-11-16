"""
helper to inspect / initialize jupyterlab labconfig settings
that are required to open jupytext notebooks in jupyterlab by default
when these settings are not present, a double click on a jupytext
notebook will cause jupyterlab to open it in an editor, i.e. as a text file
"""
import copy
import json
import logging
from pathlib import Path

class LabConfig:
    SETTINGS = Path.home() / '.jupyter' / 'labconfig' / 'default_setting_overrides.json'
    DOCTYPES = ['python', 'markdown', 'myst', 'r-markdown', 'quarto', 'julia', 'r']

    def __init__(self, logger=None):
        if False:
            while True:
                i = 10
        self.logger = logger or logging.getLogger(__name__)
        self.config = {}
        self._prior_config = {}

    def read(self):
        if False:
            while True:
                i = 10
        '\n        read the labconfig settings file\n        '
        try:
            if self.SETTINGS.exists():
                with self.SETTINGS.open() as fid:
                    self.config = json.load(fid)
        except OSError as exc:
            self.logger.error(f'Could not read {self.SETTINGS}', exc)
            return False
        self._prior_config = copy.deepcopy(self.config)
        return self

    def list_default_viewer(self):
        if False:
            while True:
                i = 10
        '\n        list the current labconfig settings\n        '
        self.logger.debug(f'Current @jupyterlab/docmanager-extension:plugin in {self.SETTINGS}')
        docmanager = self.config.get('@jupyterlab/docmanager-extension:plugin', {})
        viewers = docmanager.get('defaultViewers', {})
        for (key, value) in viewers.items():
            print(f'{key}: {value}')

    def set_default_viewers(self, doctypes=None):
        if False:
            print('Hello World!')
        if not doctypes:
            doctypes = self.DOCTYPES
        for doctype in doctypes:
            self.set_default_viewer(doctype)
        return self

    def set_default_viewer(self, doctype):
        if False:
            i = 10
            return i + 15
        if '@jupyterlab/docmanager-extension:plugin' not in self.config:
            self.config['@jupyterlab/docmanager-extension:plugin'] = {}
        if 'defaultViewers' not in self.config['@jupyterlab/docmanager-extension:plugin']:
            self.config['@jupyterlab/docmanager-extension:plugin']['defaultViewers'] = {}
        viewers = self.config['@jupyterlab/docmanager-extension:plugin']['defaultViewers']
        if doctype not in viewers:
            viewers[doctype] = 'Jupytext Notebook'

    def unset_default_viewers(self, doctypes=None):
        if False:
            for i in range(10):
                print('nop')
        if not doctypes:
            doctypes = self.DOCTYPES
        for doctype in doctypes:
            self.unset_default_viewer(doctype)
        return self

    def unset_default_viewer(self, doctype):
        if False:
            while True:
                i = 10
        viewers = self.config.get('@jupyterlab/docmanager-extension:plugin', {}).get('defaultViewers', {})
        if doctype not in viewers:
            return
        del viewers[doctype]

    def write(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        write the labconfig settings file\n        '
        if self.config == self._prior_config:
            self.logger.info(f'Nothing to do for {self.SETTINGS}')
            return True
        try:
            self.SETTINGS.parent.mkdir(parents=True, exist_ok=True)
            with self.SETTINGS.open('w') as fid:
                json.dump(self.config, fid, indent=2)
            self._prior_config = copy.deepcopy(self.config)
            return True
        except OSError as exc:
            self.logger.error(f'Could not write {self.SETTINGS}', exc)
            return False