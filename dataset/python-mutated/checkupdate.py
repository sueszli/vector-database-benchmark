from functools import partial
from PyQt6 import QtCore
from PyQt6.QtWidgets import QMessageBox
from picard import PICARD_FANCY_VERSION_STR, PICARD_VERSION, log
from picard.const import PLUGINS_API, PROGRAM_UPDATE_LEVELS
from picard.util import webbrowser2
from picard.version import Version, VersionError

class UpdateCheckManager(QtCore.QObject):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent=parent)
        self._parent = parent
        self._available_versions = {}
        self._show_always = False
        self._update_level = 0

    def check_update(self, show_always=False, update_level=0, callback=None):
        if False:
            print('Hello World!')
        'Checks if an update is available.\n\n        Compares the version number of the currently running instance of Picard\n        and displays a dialog box informing the user  if an update is available,\n        with an option of opening the download site in their browser.  If there\n        is no update available, no dialog will be shown unless the "show_always"\n        parameter has been set to True.  This allows for silent checking during\n        startup if so configured.\n\n        Args:\n            show_always: Boolean value indicating whether the results dialog\n                should be shown even when there is no update available.\n            update_level: Determines what type of updates to check.  Options are:\n                0 = only stable release versions are checked.\n                1 = stable and beta releases are checked.\n                2 = stable, beta and dev releases are checked.\n\n        Returns:\n            none.\n\n        Raises:\n            none.\n        '
        self._show_always = show_always
        self._update_level = update_level
        if self._available_versions:
            self._display_results()
        else:
            self._query_available_updates(callback=callback)

    def _query_available_updates(self, callback=None):
        if False:
            print('Hello World!')
        'Gets list of releases from specified website api.'
        log.debug('Getting Picard release information from %s', PLUGINS_API['urls']['releases'])
        self.tagger.webservice.get_url(url=PLUGINS_API['urls']['releases'], handler=partial(self._releases_json_loaded, callback=callback), priority=True, important=True)

    def _releases_json_loaded(self, response, reply, error, callback=None):
        if False:
            print('Hello World!')
        'Processes response from specified website api query.'
        if error:
            log.error(_('Error loading Picard releases list: {error_message}').format(error_message=reply.errorString()))
            if self._show_always:
                QMessageBox.information(self._parent, _('Picard Update'), _('Unable to retrieve the latest version information from the website.\n({url})').format(url=PLUGINS_API['urls']['releases']), QMessageBox.StandardButton.Ok, QMessageBox.StandardButton.Ok)
        else:
            if response and 'versions' in response:
                self._available_versions = response['versions']
            else:
                self._available_versions = {}
            for key in self._available_versions:
                log.debug("Version key '%s' -> %s", key, self._available_versions[key])
            self._display_results()
        if callback:
            callback(not error)

    def _display_results(self):
        if False:
            for i in range(10):
                print('nop')
        key = ''
        high_version = PICARD_VERSION
        for test_key in PROGRAM_UPDATE_LEVELS:
            update_level = PROGRAM_UPDATE_LEVELS[test_key]['name']
            version_tuple = self._available_versions.get(update_level, {}).get('version', (0, 0, 0, ''))
            try:
                test_version = Version(*version_tuple)
            except (TypeError, VersionError):
                log.error('Invalid version %r for update level %s.', version_tuple, update_level)
                continue
            if self._update_level >= test_key and test_version > high_version:
                key = PROGRAM_UPDATE_LEVELS[test_key]['name']
                high_version = test_version
        if key:
            if QMessageBox.information(self._parent, _('Picard Update'), _('A new version of Picard is available.\n\nThis version: {picard_old_version}\nNew version: {picard_new_version}\n\nWould you like to download the new version?').format(picard_old_version=PICARD_FANCY_VERSION_STR, picard_new_version=self._available_versions[key]['tag']), QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Ok:
                webbrowser2.open(self._available_versions[key]['urls']['download'])
        elif self._show_always:
            if self._update_level in PROGRAM_UPDATE_LEVELS:
                update_level = PROGRAM_UPDATE_LEVELS[self._update_level]['title']
            else:
                update_level = N_('unknown')
            QMessageBox.information(self._parent, _('Picard Update'), _('There is no update currently available for your subscribed update level: {update_level}\n\nYour version: {picard_old_version}\n').format(update_level=gettext_constants(update_level), picard_old_version=PICARD_FANCY_VERSION_STR), QMessageBox.StandardButton.Ok, QMessageBox.StandardButton.Ok)