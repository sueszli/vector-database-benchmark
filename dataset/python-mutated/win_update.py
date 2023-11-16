"""
Classes for working with Windows Update Agent
"""
import logging
import subprocess
import salt.utils.args
import salt.utils.data
import salt.utils.winapi
from salt.exceptions import CommandExecutionError
try:
    import pywintypes
    import win32com.client
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False
log = logging.getLogger(__name__)
REBOOT_BEHAVIOR = {0: 'Never Requires Reboot', 1: 'Always Requires Reboot', 2: 'Can Require Reboot'}
__virtualname__ = 'win_update'

def __virtual__():
    if False:
        print('Hello World!')
    if not salt.utils.platform.is_windows():
        return (False, 'win_update: Only available on Windows')
    if not HAS_PYWIN32:
        return (False, 'win_update: Missing pywin32')
    return __virtualname__

class Updates:
    """
    Wrapper around the 'Microsoft.Update.UpdateColl' instance
    Adds the list and summary functions. For use by the WindowUpdateAgent class.

    Code Example:

    .. code-block:: python

        # Create an instance
        updates = Updates()

        # Bind to the collection object
        found = updates.updates

        # This exposes Collections properties and methods
        # https://msdn.microsoft.com/en-us/library/windows/desktop/aa386107(v=vs.85).aspx
        found.Count
        found.Add

        # To use custom functions, use the original instance
        # Return the number of updates inside the collection
        updates.count()

        # Return a list of updates in the collection and details in a dictionary
        updates.list()

        # Return a summary of the contents of the updates collection
        updates.summary()
    """
    update_types = {1: 'Software', 2: 'Driver'}

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Initialize the updates collection. Can be accessed via\n        ``Updates.updates``\n        '
        with salt.utils.winapi.Com():
            self.updates = win32com.client.Dispatch('Microsoft.Update.UpdateColl')

    def count(self):
        if False:
            print('Hello World!')
        '\n        Return how many records are in the Microsoft Update Collection\n\n        Returns:\n            int: The number of updates in the collection\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            updates = salt.utils.win_update.Updates()\n            updates.count()\n        '
        return self.updates.Count

    def list(self):
        if False:
            print('Hello World!')
        "\n        Create a dictionary with the details for the updates in the collection.\n\n        Returns:\n            dict: Details about each update\n\n        .. code-block:: cfg\n\n            Dict of Updates:\n            {'<GUID>': {\n                'Title': <title>,\n                'KB': <KB>,\n                'GUID': <the globally unique identifier for the update>,\n                'Description': <description>,\n                'Downloaded': <has the update been downloaded>,\n                'Installed': <has the update been installed>,\n                'Mandatory': <is the update mandatory>,\n                'UserInput': <is user input required>,\n                'EULAAccepted': <has the EULA been accepted>,\n                'Severity': <update severity>,\n                'NeedsReboot': <is the update installed and awaiting reboot>,\n                'RebootBehavior': <will the update require a reboot>,\n                'Categories': [\n                    '<category 1>',\n                    '<category 2>',\n                    ... ]\n            }}\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            updates = salt.utils.win_update.Updates()\n            updates.list()\n        "
        if self.count() == 0:
            return 'Nothing to return'
        log.debug('Building a detailed report of the results.')
        results = {}
        for update in self.updates:
            try:
                user_input = bool(update.InstallationBehavior.CanRequestUserInput)
            except AttributeError:
                log.debug('Windows Update: Error reading InstallationBehavior COM Object')
                user_input = False
            try:
                requires_reboot = update.InstallationBehavior.RebootBehavior
            except AttributeError:
                log.debug('Windows Update: Error reading InstallationBehavior COM Object')
                requires_reboot = 2
            results[update.Identity.UpdateID] = {'guid': update.Identity.UpdateID, 'Title': str(update.Title), 'Type': self.update_types[update.Type], 'Description': update.Description, 'Downloaded': bool(update.IsDownloaded), 'Installed': bool(update.IsInstalled), 'Mandatory': bool(update.IsMandatory), 'EULAAccepted': bool(update.EulaAccepted), 'NeedsReboot': bool(update.RebootRequired), 'Severity': str(update.MsrcSeverity), 'UserInput': user_input, 'RebootBehavior': REBOOT_BEHAVIOR[requires_reboot], 'KBs': ['KB' + item for item in update.KBArticleIDs], 'Categories': [item.Name for item in update.Categories], 'SupportUrl': update.SupportUrl}
        return results

    def summary(self):
        if False:
            i = 10
            return i + 15
        "\n        Create a dictionary with a summary of the updates in the collection.\n\n        Returns:\n            dict: Summary of the contents of the collection\n\n        .. code-block:: cfg\n\n            Summary of Updates:\n            {'Total': <total number of updates returned>,\n             'Available': <updates that are not downloaded or installed>,\n             'Downloaded': <updates that are downloaded but not installed>,\n             'Installed': <updates installed (usually 0 unless installed=True)>,\n             'Categories': {\n                <category 1>: <total for that category>,\n                <category 2>: <total for category 2>,\n                ... }\n            }\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            updates = salt.utils.win_update.Updates()\n            updates.summary()\n        "
        if self.count() == 0:
            return 'Nothing to return'
        results = {'Total': 0, 'Available': 0, 'Downloaded': 0, 'Installed': 0, 'Categories': {}, 'Severity': {}}
        for update in self.updates:
            results['Total'] += 1
            if not salt.utils.data.is_true(update.IsDownloaded) and (not salt.utils.data.is_true(update.IsInstalled)):
                results['Available'] += 1
            if salt.utils.data.is_true(update.IsDownloaded) and (not salt.utils.data.is_true(update.IsInstalled)):
                results['Downloaded'] += 1
            if salt.utils.data.is_true(update.IsInstalled):
                results['Installed'] += 1
            for category in update.Categories:
                if category.Name in results['Categories']:
                    results['Categories'][category.Name] += 1
                else:
                    results['Categories'][category.Name] = 1
            if update.MsrcSeverity:
                if update.MsrcSeverity in results['Severity']:
                    results['Severity'][update.MsrcSeverity] += 1
                else:
                    results['Severity'][update.MsrcSeverity] = 1
        return results

class WindowsUpdateAgent:
    """
    Class for working with the Windows update agent
    """
    fail_codes = {-2145107924: 'WinHTTP Send/Receive failed: 0x8024402C', -2145124300: 'Download failed: 0x80240034', -2145124302: 'Invalid search criteria: 0x80240032', -2145124305: 'Cancelled by policy: 0x8024002F', -2145124307: 'Missing source: 0x8024002D', -2145124308: 'Missing source: 0x8024002C', -2145124312: 'Uninstall not allowed: 0x80240028', -2145124315: 'Prevented by policy: 0x80240025', -2145124316: 'No Updates: 0x80240024', -2145124322: 'Service being shutdown: 0x8024001E', -2145124325: 'Self Update in Progress: 0x8024001B', -2145124327: 'Exclusive Install Conflict: 0x80240019', -2145124330: 'Install not allowed: 0x80240016', -2145124333: 'Duplicate item: 0x80240013', -2145124341: 'Operation cancelled: 0x8024000B', -2145124343: 'Operation in progress: 0x80240009', -2145124284: 'Access Denied: 0x8024044', -2145124283: 'Unsupported search scope: 0x80240045', -2147024891: 'Access is denied: 0x80070005', -2149843018: 'Setup in progress: 0x8024004A', -4292599787: 'Install still pending: 0x00242015', -4292607992: 'Already downloaded: 0x00240008', -4292607993: 'Already uninstalled: 0x00240007', -4292607994: 'Already installed: 0x00240006', -4292607995: 'Reboot required: 0x00240005'}

    def __init__(self, online=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the session and load all updates into the ``_updates``\n        collection. This collection is used by the other class functions instead\n        of querying Windows update (expensive).\n\n        Args:\n\n            online (bool):\n                Tells the Windows Update Agent go online to update its local\n                update database. ``True`` will go online. ``False`` will use the\n                local update database as is. Default is ``True``\n\n                .. versionadded:: 3001\n\n        Need to look at the possibility of loading this into ``__context__``\n        '
        with salt.utils.winapi.Com():
            self._session = win32com.client.Dispatch('Microsoft.Update.Session')
            self._updates = win32com.client.Dispatch('Microsoft.Update.UpdateColl')
        self.refresh(online=online)

    def updates(self):
        if False:
            while True:
                i = 10
        '\n        Get the contents of ``_updates`` (all updates) and puts them in an\n        Updates class to expose the list and summary functions.\n\n        Returns:\n\n            Updates:\n                An instance of the Updates class with all updates for the\n                system.\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            wua = salt.utils.win_update.WindowsUpdateAgent()\n            updates = wua.updates()\n\n            # To get a list\n            updates.list()\n\n            # To get a summary\n            updates.summary()\n        '
        updates = Updates()
        found = updates.updates
        for update in self._updates:
            found.Add(update)
        return updates

    def refresh(self, online=True):
        if False:
            i = 10
            return i + 15
        '\n        Refresh the contents of the ``_updates`` collection. This gets all\n        updates in the Windows Update system and loads them into the collection.\n        This is the part that is slow.\n\n        Args:\n\n            online (bool):\n                Tells the Windows Update Agent go online to update its local\n                update database. ``True`` will go online. ``False`` will use the\n                local update database as is. Default is ``True``\n\n                .. versionadded:: 3001\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            wua = salt.utils.win_update.WindowsUpdateAgent()\n            wua.refresh()\n        '
        search_string = "Type='Software' or Type='Driver'"
        searcher = self._session.CreateUpdateSearcher()
        searcher.Online = online
        self._session.ClientApplicationID = 'Salt: Load Updates'
        try:
            results = searcher.Search(search_string)
            if results.Updates.Count == 0:
                log.debug('No Updates found for:\n\t\t%s', search_string)
                return 'No Updates found: {}'.format(search_string)
        except pywintypes.com_error as error:
            (hr, msg, exc, arg) = error.args
            try:
                failure_code = self.fail_codes[exc[5]]
            except KeyError:
                failure_code = 'Unknown Failure: {}'.format(error)
            log.error('Search Failed: %s\n\t\t%s', failure_code, search_string)
            raise CommandExecutionError(failure_code)
        self._updates = results.Updates

    def installed(self):
        if False:
            print('Hello World!')
        '\n        Gets a list of all updates available on the system that have the\n        ``IsInstalled`` attribute set to ``True``.\n\n        Returns:\n\n            Updates: An instance of Updates with the results.\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            wua = salt.utils.win_update.WindowsUpdateAgent(online=False)\n            installed_updates = wua.installed()\n        '
        updates = Updates()
        for update in self._updates:
            if salt.utils.data.is_true(update.IsInstalled):
                updates.updates.Add(update)
        return updates

    def available(self, skip_hidden=True, skip_installed=True, skip_mandatory=False, skip_reboot=False, software=True, drivers=True, categories=None, severities=None):
        if False:
            while True:
                i = 10
        "\n        Gets a list of all updates available on the system that match the passed\n        criteria.\n\n        Args:\n\n            skip_hidden (bool):\n                Skip hidden updates. Default is ``True``\n\n            skip_installed (bool):\n                Skip installed updates. Default is ``True``\n\n            skip_mandatory (bool):\n                Skip mandatory updates. Default is ``False``\n\n            skip_reboot (bool):\n                Skip updates that can or do require reboot. Default is ``False``\n\n            software (bool):\n                Include software updates. Default is ``True``\n\n            drivers (bool):\n                Include driver updates. Default is ``True``\n\n            categories (list):\n                Include updates that have these categories. Default is none\n                (all categories). Categories include the following:\n\n                * Critical Updates\n                * Definition Updates\n                * Drivers (make sure you set drivers=True)\n                * Feature Packs\n                * Security Updates\n                * Update Rollups\n                * Updates\n                * Update Rollups\n                * Windows 7\n                * Windows 8.1\n                * Windows 8.1 drivers\n                * Windows 8.1 and later drivers\n                * Windows Defender\n\n            severities (list):\n                Include updates that have these severities. Default is none\n                (all severities). Severities include the following:\n\n                * Critical\n                * Important\n\n        .. note::\n\n            All updates are either software or driver updates. If both\n            ``software`` and ``drivers`` is ``False``, nothing will be returned.\n\n        Returns:\n\n            Updates: An instance of Updates with the results of the search.\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            wua = salt.utils.win_update.WindowsUpdateAgent()\n\n            # Gets all updates and shows a summary\n            updates = wua.available()\n            updates.summary()\n\n            # Get a list of Critical updates\n            updates = wua.available(categories=['Critical Updates'])\n            updates.list()\n        "
        updates = Updates()
        found = updates.updates
        for update in self._updates:
            if salt.utils.data.is_true(update.IsHidden) and skip_hidden:
                continue
            if salt.utils.data.is_true(update.IsInstalled) and skip_installed:
                continue
            if salt.utils.data.is_true(update.IsMandatory) and skip_mandatory:
                continue
            try:
                requires_reboot = salt.utils.data.is_true(update.InstallationBehavior.RebootBehavior)
            except AttributeError:
                log.debug('Windows Update: Error reading InstallationBehavior COM Object')
                requires_reboot = True
            if requires_reboot and skip_reboot:
                continue
            if not software and update.Type == 1:
                continue
            if not drivers and update.Type == 2:
                continue
            if categories is not None:
                match = False
                for category in update.Categories:
                    if category.Name in categories:
                        match = True
                if not match:
                    continue
            if severities is not None:
                if update.MsrcSeverity not in severities:
                    continue
            found.Add(update)
        return updates

    def search(self, search_string):
        if False:
            for i in range(10):
                print('nop')
        "\n        Search for either a single update or a specific list of updates. GUIDs\n        are searched first, then KB numbers, and finally Titles.\n\n        Args:\n\n            search_string (str, list):\n                The search string to use to find the update. This can be the\n                GUID or KB of the update (preferred). It can also be the full\n                Title of the update or any part of the Title. A partial Title\n                search is less specific and can return multiple results.\n\n        Returns:\n            Updates: An instance of Updates with the results of the search\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            wua = salt.utils.win_update.WindowsUpdateAgent()\n\n            # search for a single update and show its details\n            updates = wua.search('KB3194343')\n            updates.list()\n\n            # search for a list of updates and show their details\n            updates = wua.search(['KB3195432', '12345678-abcd-1234-abcd-1234567890ab'])\n            updates.list()\n        "
        updates = Updates()
        found = updates.updates
        if isinstance(search_string, str):
            search_string = [search_string]
        if isinstance(search_string, int):
            search_string = [str(search_string)]
        for update in self._updates:
            for find in search_string:
                if find == update.Identity.UpdateID:
                    found.Add(update)
                    continue
                if find in ['KB' + item for item in update.KBArticleIDs]:
                    found.Add(update)
                    continue
                if find in [item for item in update.KBArticleIDs]:
                    found.Add(update)
                    continue
                if find in update.Title:
                    found.Add(update)
                    continue
        return updates

    def download(self, updates):
        if False:
            i = 10
            return i + 15
        "\n        Download the updates passed in the updates collection. Load the updates\n        collection using ``search`` or ``available``\n\n        Args:\n\n            updates (Updates):\n                An instance of the Updates class containing a the updates to be\n                downloaded.\n\n        Returns:\n            dict: A dictionary containing the results of the download\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            wua = salt.utils.win_update.WindowsUpdateAgent()\n\n            # Download KB3195454\n            updates = wua.search('KB3195454')\n            results = wua.download(updates)\n        "
        if updates.count() == 0:
            ret = {'Success': False, 'Updates': 'Nothing to download'}
            return ret
        downloader = self._session.CreateUpdateDownloader()
        self._session.ClientApplicationID = 'Salt: Download Update'
        with salt.utils.winapi.Com():
            download_list = win32com.client.Dispatch('Microsoft.Update.UpdateColl')
            ret = {'Updates': {}}
            for update in updates.updates:
                uid = update.Identity.UpdateID
                ret['Updates'][uid] = {}
                ret['Updates'][uid]['Title'] = update.Title
                ret['Updates'][uid]['AlreadyDownloaded'] = bool(update.IsDownloaded)
                if not salt.utils.data.is_true(update.EulaAccepted):
                    log.debug('Accepting EULA: %s', update.Title)
                    update.AcceptEula()
                if not salt.utils.data.is_true(update.IsDownloaded):
                    log.debug('To Be Downloaded: %s', uid)
                    log.debug('\tTitle: %s', update.Title)
                    download_list.Add(update)
            if download_list.Count == 0:
                ret = {'Success': True, 'Updates': 'Nothing to download'}
                return ret
            downloader.Updates = download_list
            try:
                log.debug('Downloading Updates')
                result = downloader.Download()
            except pywintypes.com_error as error:
                (hr, msg, exc, arg) = error.args
                try:
                    failure_code = self.fail_codes[exc[5]]
                except KeyError:
                    failure_code = 'Unknown Failure: {}'.format(error)
                log.error('Download Failed: %s', failure_code)
                raise CommandExecutionError(failure_code)
            result_code = {0: 'Download Not Started', 1: 'Download In Progress', 2: 'Download Succeeded', 3: 'Download Succeeded With Errors', 4: 'Download Failed', 5: 'Download Aborted'}
            log.debug('Download Complete')
            log.debug(result_code[result.ResultCode])
            ret['Message'] = result_code[result.ResultCode]
            if result.ResultCode in [2, 3]:
                log.debug('Downloaded Successfully')
                ret['Success'] = True
            else:
                log.debug('Download Failed')
                ret['Success'] = False
            for i in range(download_list.Count):
                uid = download_list.Item(i).Identity.UpdateID
                ret['Updates'][uid]['Result'] = result_code[result.GetUpdateResult(i).ResultCode]
        return ret

    def install(self, updates):
        if False:
            return 10
        "\n        Install the updates passed in the updates collection. Load the updates\n        collection using the ``search`` or ``available`` functions. If the\n        updates need to be downloaded, use the ``download`` function.\n\n        Args:\n\n            updates (Updates):\n                An instance of the Updates class containing a the updates to be\n                installed.\n\n        Returns:\n            dict: A dictionary containing the results of the installation\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            wua = salt.utils.win_update.WindowsUpdateAgent()\n\n            # install KB3195454\n            updates = wua.search('KB3195454')\n            results = wua.download(updates)\n            results = wua.install(updates)\n        "
        if updates.count() == 0:
            ret = {'Success': False, 'Updates': 'Nothing to install'}
            return ret
        installer = self._session.CreateUpdateInstaller()
        self._session.ClientApplicationID = 'Salt: Install Update'
        with salt.utils.winapi.Com():
            install_list = win32com.client.Dispatch('Microsoft.Update.UpdateColl')
            ret = {'Updates': {}}
            for update in updates.updates:
                uid = update.Identity.UpdateID
                ret['Updates'][uid] = {}
                ret['Updates'][uid]['Title'] = update.Title
                ret['Updates'][uid]['AlreadyInstalled'] = bool(update.IsInstalled)
                if not salt.utils.data.is_true(update.IsInstalled):
                    log.debug('To Be Installed: %s', uid)
                    log.debug('\tTitle: %s', update.Title)
                    install_list.Add(update)
            if install_list.Count == 0:
                ret = {'Success': True, 'Updates': 'Nothing to install'}
                return ret
            installer.Updates = install_list
            try:
                log.debug('Installing Updates')
                result = installer.Install()
            except pywintypes.com_error as error:
                (hr, msg, exc, arg) = error.args
                try:
                    failure_code = self.fail_codes[exc[5]]
                except KeyError:
                    failure_code = 'Unknown Failure: {}'.format(error)
                log.error('Install Failed: %s', failure_code)
                raise CommandExecutionError(failure_code)
            result_code = {0: 'Installation Not Started', 1: 'Installation In Progress', 2: 'Installation Succeeded', 3: 'Installation Succeeded With Errors', 4: 'Installation Failed', 5: 'Installation Aborted'}
            log.debug('Install Complete')
            log.debug(result_code[result.ResultCode])
            ret['Message'] = result_code[result.ResultCode]
            if result.ResultCode in [2, 3]:
                ret['Success'] = True
                ret['NeedsReboot'] = result.RebootRequired
                log.debug('NeedsReboot: %s', result.RebootRequired)
            else:
                log.debug('Install Failed')
                ret['Success'] = False
            for i in range(install_list.Count):
                uid = install_list.Item(i).Identity.UpdateID
                ret['Updates'][uid]['Result'] = result_code[result.GetUpdateResult(i).ResultCode]
                try:
                    reboot_behavior = install_list.Item(i).InstallationBehavior.RebootBehavior
                except AttributeError:
                    log.debug('Windows Update: Error reading InstallationBehavior COM Object')
                    reboot_behavior = 2
                ret['Updates'][uid]['RebootBehavior'] = REBOOT_BEHAVIOR[reboot_behavior]
        return ret

    def uninstall(self, updates):
        if False:
            return 10
        "\n        Uninstall the updates passed in the updates collection. Load the updates\n        collection using the ``search`` or ``available`` functions.\n\n        .. note::\n\n            Starting with Windows 10 the Windows Update Agent is unable to\n            uninstall updates. An ``Uninstall Not Allowed`` error is returned.\n            If this error is encountered this function will instead attempt to\n            use ``dism.exe`` to perform the un-installation. ``dism.exe`` may\n            fail to to find the KB number for the package. In that case, removal\n            will fail.\n\n        Args:\n\n            updates (Updates):\n                An instance of the Updates class containing a the updates to be\n                uninstalled.\n\n        Returns:\n            dict: A dictionary containing the results of the un-installation\n\n        Code Example:\n\n        .. code-block:: python\n\n            import salt.utils.win_update\n            wua = salt.utils.win_update.WindowsUpdateAgent()\n\n            # uninstall KB3195454\n            updates = wua.search('KB3195454')\n            results = wua.uninstall(updates)\n        "
        if updates.count() == 0:
            ret = {'Success': False, 'Updates': 'Nothing to uninstall'}
            return ret
        installer = self._session.CreateUpdateInstaller()
        self._session.ClientApplicationID = 'Salt: Uninstall Update'
        with salt.utils.winapi.Com():
            uninstall_list = win32com.client.Dispatch('Microsoft.Update.UpdateColl')
            ret = {'Updates': {}}
            for update in updates.updates:
                uid = update.Identity.UpdateID
                ret['Updates'][uid] = {}
                ret['Updates'][uid]['Title'] = update.Title
                ret['Updates'][uid]['AlreadyUninstalled'] = not bool(update.IsInstalled)
                if salt.utils.data.is_true(update.IsInstalled):
                    log.debug('To Be Uninstalled: %s', uid)
                    log.debug('\tTitle: %s', update.Title)
                    uninstall_list.Add(update)
            if uninstall_list.Count == 0:
                ret = {'Success': False, 'Updates': 'Nothing to uninstall'}
                return ret
            installer.Updates = uninstall_list
            try:
                log.debug('Uninstalling Updates')
                result = installer.Uninstall()
            except pywintypes.com_error as error:
                (hr, msg, exc, arg) = error.args
                try:
                    failure_code = self.fail_codes[exc[5]]
                except KeyError:
                    failure_code = 'Unknown Failure: {}'.format(error)
                if exc[5] == -2145124312:
                    log.debug('Uninstall Failed with WUA, attempting with DISM')
                    try:
                        for item in uninstall_list:
                            for kb in item.KBArticleIDs:
                                cmd = ['dism', '/Online', '/Get-Packages']
                                pkg_list = self._run(cmd)[0].splitlines()
                                for item in pkg_list:
                                    if 'kb' + kb in item.lower():
                                        pkg = item.split(' : ')[1]
                                        ret['DismPackage'] = pkg
                                        cmd = ['dism', '/Online', '/Remove-Package', '/PackageName:{}'.format(pkg), '/Quiet', '/NoRestart']
                                        self._run(cmd)
                    except CommandExecutionError as exc:
                        log.debug('Uninstall using DISM failed')
                        log.debug('Command: %s', ' '.join(cmd))
                        log.debug('Error: %s', exc)
                        raise CommandExecutionError('Uninstall using DISM failed: {}'.format(exc))
                    log.debug('Uninstall Completed using DISM')
                    ret['Success'] = True
                    ret['Message'] = 'Uninstalled using DISM'
                    ret['NeedsReboot'] = needs_reboot()
                    log.debug('NeedsReboot: %s', ret['NeedsReboot'])
                    self.refresh(online=False)
                    for update in self._updates:
                        uid = update.Identity.UpdateID
                        for item in uninstall_list:
                            if item.Identity.UpdateID == uid:
                                if not update.IsInstalled:
                                    ret['Updates'][uid]['Result'] = 'Uninstallation Succeeded'
                                else:
                                    ret['Updates'][uid]['Result'] = 'Uninstallation Failed'
                                try:
                                    requires_reboot = update.InstallationBehavior.RebootBehavior
                                except AttributeError:
                                    log.debug('Windows Update: Error reading InstallationBehavior COM Object')
                                    requires_reboot = 2
                                ret['Updates'][uid]['RebootBehavior'] = REBOOT_BEHAVIOR[requires_reboot]
                    return ret
                log.error('Uninstall Failed: %s', failure_code)
                raise CommandExecutionError(failure_code)
            result_code = {0: 'Uninstallation Not Started', 1: 'Uninstallation In Progress', 2: 'Uninstallation Succeeded', 3: 'Uninstallation Succeeded With Errors', 4: 'Uninstallation Failed', 5: 'Uninstallation Aborted'}
            log.debug('Uninstall Complete')
            log.debug(result_code[result.ResultCode])
            ret['Message'] = result_code[result.ResultCode]
            if result.ResultCode in [2, 3]:
                ret['Success'] = True
                ret['NeedsReboot'] = result.RebootRequired
                log.debug('NeedsReboot: %s', result.RebootRequired)
            else:
                log.debug('Uninstall Failed')
                ret['Success'] = False
            for i in range(uninstall_list.Count):
                uid = uninstall_list.Item(i).Identity.UpdateID
                ret['Updates'][uid]['Result'] = result_code[result.GetUpdateResult(i).ResultCode]
                try:
                    reboot_behavior = uninstall_list.Item(i).InstallationBehavior.RebootBehavior
                except AttributeError:
                    log.debug('Windows Update: Error reading InstallationBehavior COM Object')
                    reboot_behavior = 2
                ret['Updates'][uid]['RebootBehavior'] = REBOOT_BEHAVIOR[reboot_behavior]
        return ret

    def _run(self, cmd):
        if False:
            while True:
                i = 10
        '\n        Internal function for running commands. Used by the uninstall function.\n\n        Args:\n            cmd (str, list):\n                The command to run\n\n        Returns:\n            str: The stdout of the command\n        '
        if isinstance(cmd, str):
            cmd = salt.utils.args.shlex_split(cmd)
        try:
            log.debug(cmd)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return p.communicate()
        except OSError as exc:
            log.debug('Command Failed: %s', ' '.join(cmd))
            log.debug('Error: %s', exc)
            raise CommandExecutionError(exc)

def needs_reboot():
    if False:
        i = 10
        return i + 15
    '\n    Determines if the system needs to be rebooted.\n\n    Returns:\n\n        bool: ``True`` if the system requires a reboot, ``False`` if not\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        import salt.utils.win_update\n\n        salt.utils.win_update.needs_reboot()\n\n    '
    with salt.utils.winapi.Com():
        try:
            obj_sys = win32com.client.Dispatch('Microsoft.Update.SystemInfo')
        except pywintypes.com_error as exc:
            (_, msg, _, _) = exc.args
            log.debug('Failed to create SystemInfo object: %s', msg)
            return False
        return salt.utils.data.is_true(obj_sys.RebootRequired)