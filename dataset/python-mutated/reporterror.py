"""Report Error Dialog."""
import sys
from urllib.parse import quote
from qtpy.QtCore import Qt, QUrl, QUrlQuery, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QApplication, QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout
from spyder import __project_url__, __trouble_url__, dependencies, get_versions_text
from spyder.api.config.fonts import SpyderFontsMixin, SpyderFontType
from spyder.api.config.mixins import SpyderConfigurationAccessor
from spyder.config.base import _, is_conda_based_app
from spyder.plugins.console.widgets.console import ConsoleBaseWidget
from spyder.utils.conda import is_conda_env, get_conda_env_path, find_conda
from spyder.utils.icon_manager import ima
from spyder.utils.programs import run_program
from spyder.utils.qthelpers import restore_keyevent
from spyder.widgets.github.backend import GithubBackend
from spyder.widgets.mixins import BaseEditMixin, TracebackLinksMixin
from spyder.widgets.simplecodeeditor import SimpleCodeEditor
TITLE_MIN_CHARS = 15
DESC_MIN_CHARS = 50

class DescriptionWidget(SimpleCodeEditor, SpyderFontsMixin):
    """Widget to enter error description."""

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.setup_editor(language='md', font=self.get_font(SpyderFontType.MonospaceInterface, font_size_delta=1), wrap=True, linenumbers=False, highlight_current_line=False)
        self.header = '### What steps will reproduce the problem?\n\n<!--- You can use Markdown here --->\n\n'
        self.set_text(self.header)
        self.move_cursor(len(self.header))
        self.header_end_pos = self.get_position('eof')

    def remove_text(self):
        if False:
            while True:
                i = 10
        'Remove text.'
        self.truncate_selection(self.header_end_pos)
        self.remove_selected_text()

    def cut(self):
        if False:
            print('Hello World!')
        'Cut text'
        self.truncate_selection(self.header_end_pos)
        if self.has_selected_text():
            super().cut()

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Reimplemented Qt Method to avoid removing the header.'
        (event, text, key, ctrl, shift) = restore_keyevent(event)
        cursor_position = self.get_position('cursor')
        if cursor_position < self.header_end_pos:
            self.restrict_cursor_position(self.header_end_pos, 'eof')
        elif key == Qt.Key_Backspace:
            if self.has_selected_text():
                self.remove_text()
            elif self.header_end_pos == cursor_position:
                return
            else:
                self.stdkey_backspace()
        elif key == Qt.Key_X and ctrl:
            self.cut()
        else:
            super().keyPressEvent(event)

    def delete(self):
        if False:
            i = 10
            return i + 15
        'Reimplemented to avoid removing the header.'
        cursor_position = self.get_position('cursor')
        if cursor_position < self.header_end_pos:
            self.restrict_cursor_position(self.header_end_pos, 'eof')
        elif self.has_selected_text():
            self.remove_text()
        else:
            self.stdkey_clear()

    def contextMenuEvent(self, event):
        if False:
            while True:
                i = 10
        'Reimplemented Qt Method to not show the context menu.'
        pass

class ShowErrorWidget(TracebackLinksMixin, ConsoleBaseWidget, BaseEditMixin, SpyderFontsMixin):
    """Widget to show errors as they appear in the Internal console."""
    QT_CLASS = QPlainTextEdit
    sig_go_to_error_requested = Signal(str)

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        ConsoleBaseWidget.__init__(self, parent)
        BaseEditMixin.__init__(self)
        TracebackLinksMixin.__init__(self)
        self.setReadOnly(True)
        self.set_pythonshell_font(self.get_font(SpyderFontType.MonospaceInterface, font_size_delta=1))

class SpyderErrorDialog(QDialog, SpyderConfigurationAccessor):
    """Custom error dialog for error reporting."""

    def __init__(self, parent=None, is_report=False):
        if False:
            for i in range(10):
                print('nop')
        QDialog.__init__(self, parent)
        self.is_report = is_report
        self._testing = False
        self.setWindowTitle(_('Issue reporter'))
        self._github_org = 'spyder-ide'
        self._github_repo = 'spyder'
        self.error_traceback = ''
        if self.is_report:
            title = _('Please fill the following information')
        else:
            title = _('Spyder has encountered an internal problem!')
        self.main_label = QLabel(_('<h4>{title}</h4>Before reporting this problem, <i>please</i> consult our comprehensive <b><a href="{trouble_url}">Troubleshooting Guide</a></b> which should help solve most issues, and search for <b><a href="{project_url}">known bugs</a></b> matching your error message or problem description for a quicker solution.').format(title=title, trouble_url=__trouble_url__, project_url=__project_url__))
        self.main_label.setOpenExternalLinks(True)
        self.main_label.setWordWrap(True)
        self.title = QLineEdit()
        self.title.textChanged.connect(self._contents_changed)
        self.title_chars_label = QLabel(_('{} more characters to go...').format(TITLE_MIN_CHARS))
        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        red_asterisk = '<font color="Red">*</font>'
        title_label = QLabel(_('<b>Title</b>: {}').format(red_asterisk))
        form_layout.setWidget(0, QFormLayout.LabelRole, title_label)
        form_layout.setWidget(0, QFormLayout.FieldRole, self.title)
        steps_header = QLabel(_('<b>Steps to reproduce:</b> {}').format(red_asterisk))
        self.steps_text = QLabel(_('Please enter a detailed step-by-step description (in English) of what led up to the problem below. Issue reports without a clear way to reproduce them will be closed.'))
        self.steps_text.setWordWrap(True)
        self.input_description = DescriptionWidget(self)
        input_description_layout = QHBoxLayout()
        input_description_layout.addWidget(self.input_description)
        input_description_layout.setContentsMargins(4, 0, 0, 0)
        self.input_description.textChanged.connect(self._contents_changed)
        self.details = ShowErrorWidget(self)
        self.details.setStyleSheet('margin-left: 4px')
        self.details.hide()
        self.description_minimum_length = DESC_MIN_CHARS
        self.require_minimum_length = True
        self.initial_chars = len(self.input_description.toPlainText())
        self.desc_chars_label = QLabel(_('{} more characters to go...').format(self.description_minimum_length))
        self.dismiss_box = QCheckBox(_('Hide all future errors during this session'))
        self.dismiss_box.setStyleSheet('margin-left: 2px')
        self.include_env = QCheckBox(_('Include IPython console environment'))
        self.include_env.setStyleSheet('margin-left: 2px')
        self.include_env.hide()
        gh_icon = ima.icon('github')
        self.submit_btn = QPushButton(gh_icon, _('Submit to Github'))
        self.submit_btn.setEnabled(False)
        self.submit_btn.clicked.connect(self._submit_to_github)
        self.details_btn = QPushButton(_('Show details'))
        self.details_btn.clicked.connect(self._show_details)
        if self.is_report:
            self.details_btn.hide()
        self.close_btn = QPushButton(_('Close'))
        self.close_btn.clicked.connect(self.reject)
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.submit_btn)
        buttons_layout.addWidget(self.details_btn)
        buttons_layout.addWidget(self.close_btn)
        buttons_layout.setContentsMargins(4, 0, 0, 0)
        layout = QVBoxLayout()
        layout.addWidget(self.main_label)
        layout.addSpacing(15)
        layout.addLayout(form_layout)
        layout.addWidget(self.title_chars_label)
        layout.addSpacing(15)
        layout.addWidget(steps_header)
        layout.addSpacing(-1)
        layout.addWidget(self.steps_text)
        layout.addSpacing(1)
        layout.addLayout(input_description_layout)
        layout.addWidget(self.details)
        layout.addWidget(self.desc_chars_label)
        if not self.is_report:
            layout.addSpacing(15)
            layout.addWidget(self.dismiss_box)
        if not is_conda_based_app() or not self.get_conf('default', section='main_interpreter'):
            self.include_env.show()
            if self.is_report:
                layout.addSpacing(15)
            layout.addWidget(self.include_env)
            layout.addSpacing(5)
        else:
            layout.addSpacing(5)
        layout.addLayout(buttons_layout)
        layout.setContentsMargins(25, 20, 29, 10)
        self.setLayout(layout)
        self.resize(600, 650)
        self.setMinimumWidth(600)
        self.title.setFocus()
        self.setTabOrder(self.title, self.input_description)

    @classmethod
    def render_issue(cls, description='', traceback='', include_env=False):
        if False:
            return 10
        '\n        Render issue content.\n\n        Parameters\n        ----------\n        description: str\n            Description to include in issue message.\n        traceback: str\n            Traceback text.\n        include_env: bool (False)\n            Whether to include the IPython console environment.\n        '
        if not dependencies.DEPENDENCIES:
            try:
                dependencies.declare_dependencies()
            except ValueError:
                pass
        if not description:
            description = '### What steps reproduce the problem?'
        if traceback:
            error_section = '### Traceback\n```python-traceback\n{}\n```'.format(traceback)
        else:
            error_section = ''
        versions_text = get_versions_text()
        issue_template = f'## Description\n\n{description}\n\n{error_section}\n\n## Versions\n\n{versions_text}\n### Dependencies\n\n```\n{dependencies.status()}\n```\n'
        if include_env:
            pyexe = cls.get_conf(cls, 'executable', section='main_interpreter')
            if is_conda_env(pyexec=pyexe):
                path = get_conda_env_path(pyexe)
                exe = find_conda()
                args = ['list', '--prefix', path]
            else:
                exe = pyexe
                args = ['-m', 'pip', 'list']
            proc = run_program(exe, args=args)
            (ext_env, stderr) = proc.communicate()
            issue_template += f'\n### Environment\n\n<details><summary>Environment</summary>\n\n```\n{ext_env.decode()}\n```\n</details>\n'
        return issue_template

    @staticmethod
    def open_web_report(body, title=None):
        if False:
            print('Hello World!')
        '\n        Open a new issue on Github with prefilled information.\n\n        Parameters\n        ----------\n        body: str\n            The body content of the report.\n        title: str or None, optional\n            The title of the report. Default is None.\n        '
        url = QUrl(__project_url__ + '/issues/new')
        query = QUrlQuery()
        query.addQueryItem('body', quote(body))
        if title:
            query.addQueryItem('title', quote(title))
        url.setQuery(query)
        QDesktopServices.openUrl(url)

    def set_require_minimum_length(self, state):
        if False:
            while True:
                i = 10
        'Remove the requirement for minimum length.'
        self.require_minimum_length = state
        if state:
            self._contents_changed()
        else:
            self.desc_chars_label.setText('')

    def set_github_repo_org(self, repo_fullname):
        if False:
            return 10
        'Set the report Github organization and repository.'
        (org, repo) = repo_fullname.split('/')
        self._github_org = org
        self._github_repo = repo

    def _submit_to_github(self):
        if False:
            return 10
        'Action to take when pressing the submit button.'
        title = self.title.text()
        description = self.input_description.toPlainText()
        traceback = self.error_traceback[:-1]
        issue_text = self.render_issue(description=description, traceback=traceback, include_env=self.include_env.isChecked())
        try:
            org = self._github_org if not self._testing else 'ccordoba12'
            repo = self._github_repo
            github_backend = GithubBackend(org, repo, parent_widget=self)
            github_report = github_backend.send_report(title, issue_text)
            if github_report:
                self.close()
        except Exception:
            ret = QMessageBox.question(self, _('Error'), _('An error occurred while trying to send the issue to Github automatically. Would you like to open it manually?<br><br>If so, please make sure to paste your clipboard into the issue report box that will appear in a new browser tab before clicking <i>Submit</i> on that page.'))
            if ret in [QMessageBox.Yes, QMessageBox.Ok]:
                QApplication.clipboard().setText(issue_text)
                issue_body = ' \n<!---   *** BEFORE SUBMITTING: PASTE CLIPBOARD HERE TO COMPLETE YOUR REPORT ***   ---!>\n'
                self.open_web_report(body=issue_body, title=title)

    def append_traceback(self, text):
        if False:
            print('Hello World!')
        'Append text to the traceback, to be displayed in details.'
        self.error_traceback += text

    def _show_details(self):
        if False:
            for i in range(10):
                print('nop')
        'Show traceback on its own dialog'
        if self.details.isVisible():
            self.details.hide()
            self.details_btn.setText(_('Show details'))
        else:
            self.resize(570, 700)
            self.details.document().setPlainText('')
            self.details.append_text_to_shell(self.error_traceback, error=True, prompt=False)
            self.details.show()
            self.details_btn.setText(_('Hide details'))

    def _contents_changed(self):
        if False:
            while True:
                i = 10
        'Activate submit_btn.'
        if not self.require_minimum_length:
            return
        desc_chars = len(self.input_description.toPlainText()) - self.initial_chars
        if desc_chars < self.description_minimum_length:
            self.desc_chars_label.setText(u'{} {}'.format(self.description_minimum_length - desc_chars, _('more characters to go...')))
        else:
            self.desc_chars_label.setText(_('Description complete; thanks!'))
        title_chars = len(self.title.text())
        if title_chars < TITLE_MIN_CHARS:
            self.title_chars_label.setText(u'{} {}'.format(TITLE_MIN_CHARS - title_chars, _('more characters to go...')))
        else:
            self.title_chars_label.setText(_('Title complete; thanks!'))
        submission_enabled = desc_chars >= self.description_minimum_length and title_chars >= TITLE_MIN_CHARS
        self.submit_btn.setEnabled(submission_enabled)

    def set_title(self, title):
        if False:
            print('Hello World!')
        'Set the title for the report.'
        self.title.setText(title)

    def set_description(self, description):
        if False:
            while True:
                i = 10
        'Set the description for the report.'
        self.input_description.setPlainText(description)

    def set_color_scheme(self, color_scheme):
        if False:
            i = 10
            return i + 15
        'Set the color scheme for the description input.'
        self.input_description.set_color_scheme(color_scheme)

def test():
    if False:
        i = 10
        return i + 15
    from spyder.utils.qthelpers import qapplication
    app = qapplication()
    dlg = SpyderErrorDialog()
    dlg._testing = True
    dlg.show()
    sys.exit(dlg.exec_())
if __name__ == '__main__':
    test()