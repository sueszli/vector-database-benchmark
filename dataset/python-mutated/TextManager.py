import collections
from typing import Optional, Dict, List, cast
from PyQt6.QtCore import QObject, pyqtSlot
from UM.i18n import i18nCatalog
from UM.Resources import Resources
from UM.Version import Version
catalog = i18nCatalog('cura')

class TextManager(QObject):

    def __init__(self, parent: Optional['QObject']=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self._change_log_text = ''

    @pyqtSlot(result=str)
    def getChangeLogText(self) -> str:
        if False:
            print('Hello World!')
        if not self._change_log_text:
            self._change_log_text = self._loadChangeLogText()
        return self._change_log_text

    def _loadChangeLogText(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        try:
            file_path = Resources.getPath(Resources.Texts, 'change_log.txt')
        except FileNotFoundError as e:
            return catalog.i18nc('@text:window', 'The release notes could not be opened.') + '<br>' + str(e)
        change_logs_dict = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                open_version = None
                open_header = ''
                for line in f:
                    line = line.replace('\n', '')
                    if '[' in line and ']' in line:
                        line = line.replace('[', '')
                        line = line.replace(']', '')
                        open_version = Version(line)
                        if open_version < Version([0, 0, 1]):
                            open_version = Version([99, 99, 99])
                        if Version([14, 99, 99]) < open_version < Version([16, 0, 0]):
                            open_version = Version([0, open_version.getMinor(), open_version.getRevision(), open_version.getPostfixVersion()])
                        open_header = ''
                        change_logs_dict[open_version] = collections.OrderedDict()
                    elif line.startswith('*'):
                        open_header = line.replace('*', '')
                        change_logs_dict[cast(Version, open_version)][open_header] = []
                    elif line != '':
                        if open_header not in change_logs_dict[cast(Version, open_version)]:
                            change_logs_dict[cast(Version, open_version)][open_header] = []
                        change_logs_dict[cast(Version, open_version)][open_header].append(line)
        except EnvironmentError as e:
            return catalog.i18nc('@text:window', 'The release notes could not be opened.') + '<br>' + str(e)
        content = ''
        for version in sorted(change_logs_dict.keys(), reverse=True):
            text_version = version
            if version < Version([1, 0, 0]):
                text_version = Version([15, version.getMinor(), version.getRevision(), version.getPostfixVersion()])
            if version > Version([99, 0, 0]):
                text_version = ''
            content += '<h1>' + str(text_version) + '</h1><br>' if text_version else ''
            content += ''
            for change in change_logs_dict[version]:
                if str(change) != '':
                    content += '<b>' + str(change) + '</b><br>'
                for line in change_logs_dict[version][change]:
                    content += str(line) + '<br>'
                content += '<br>'
        return content