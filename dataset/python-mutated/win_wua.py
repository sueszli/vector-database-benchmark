"""
Module for managing Windows Updates using the Windows Update Agent.

List updates on the system using the following functions:

- :py:func:`win_wua.available <salt.modules.win_wua.available>`
- :py:func:`win_wua.list <salt.modules.win_wua.list_>`

This is an easy way to find additional information about updates available to
to the system, such as the GUID, KB number, or description.

Once you have the GUID or a KB number for the update you can get information
about the update, download, install, or uninstall it using these functions:

- :py:func:`win_wua.get <salt.modules.win_wua.get>`
- :py:func:`win_wua.download <salt.modules.win_wua.download>`
- :py:func:`win_wua.install <salt.modules.win_wua.install>`
- :py:func:`win_wua.uninstall <salt.modules.win_wua.uninstall>`

The get function expects a name in the form of a GUID, KB, or Title and should
return information about a single update. The other functions accept either a
single item or a list of items for downloading/installing/uninstalling a
specific list of items.

The :py:func:`win_wua.list <salt.modules.win_wua.list_>` and
:py:func:`win_wua.get <salt.modules.win_wua.get>` functions are utility
functions. In addition to returning information about updates they can also
download and install updates by setting ``download=True`` or ``install=True``.
So, with py:func:`win_wua.list <salt.modules.win_wua.list_>` for example, you
could run the function with the filters you want to see what is available. Then
just add ``install=True`` to install everything on that list.

If you want to download, install, or uninstall specific updates, use
:py:func:`win_wua.download <salt.modules.win_wua.download>`,
:py:func:`win_wua.install <salt.modules.win_wua.install>`, or
:py:func:`win_wua.uninstall <salt.modules.win_wua.uninstall>`. To update your
system with the latest updates use :py:func:`win_wua.list
<salt.modules.win_wua.list_>` and set ``install=True``

You can also adjust the Windows Update settings using the
:py:func:`win_wua.set_wu_settings <salt.modules.win_wua.set_wu_settings>`
function. This function is only supported on the following operating systems:

- Windows Vista / Server 2008
- Windows 7 / Server 2008R2
- Windows 8 / Server 2012
- Windows 8.1 / Server 2012R2

As of Windows 10 and Windows Server 2016, the ability to modify the Windows
Update settings has been restricted. The settings can be modified in the Local
Group Policy using the ``lgpo`` module.

.. versionadded:: 2015.8.0

:depends: salt.utils.win_update
"""
import logging
import salt.utils.platform
import salt.utils.win_service
import salt.utils.win_update
import salt.utils.winapi
from salt.exceptions import CommandExecutionError
try:
    import win32com.client
    HAS_PYWIN32 = True
except ImportError:
    HAS_PYWIN32 = False
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}

def __virtual__():
    if False:
        return 10
    '\n    Only works on Windows systems with PyWin32\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'WUA: Only available on Windows systems')
    if not HAS_PYWIN32:
        return (False, 'WUA: Requires PyWin32 libraries')
    if not salt.utils.win_update.HAS_PYWIN32:
        return (False, 'WUA: Missing Libraries required by salt.utils.win_update')
    if salt.utils.win_service.info('wuauserv')['StartType'] == 'Disabled':
        return (False, 'WUA: The Windows Update service (wuauserv) must not be disabled')
    if salt.utils.win_service.info('msiserver')['StartType'] == 'Disabled':
        return (False, 'WUA: The Windows Installer service (msiserver) must not be disabled')
    if salt.utils.win_service.info('BITS')['StartType'] == 'Disabled':
        return (False, 'WUA: The Background Intelligent Transfer service (bits) must not be disabled')
    if salt.utils.win_service.info('CryptSvc')['StartType'] == 'Disabled':
        return (False, 'WUA: The Cryptographic Services service (CryptSvc) must not be disabled')
    if salt.utils.win_service.info('TrustedInstaller')['StartType'] == 'Disabled':
        return (False, 'WUA: The Windows Module Installer service (TrustedInstaller) must not be disabled')
    return True

def available(software=True, drivers=True, summary=False, skip_installed=True, skip_hidden=True, skip_mandatory=False, skip_reboot=False, categories=None, severities=None, online=True):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2017.7.0\n\n    List updates that match the passed criteria. This allows for more filter\n    options than :func:`list`. Good for finding a specific GUID or KB.\n\n    Args:\n\n        software (bool):\n            Include software updates in the results. Default is ``True``\n\n        drivers (bool):\n            Include driver updates in the results. Default is ``True``\n\n        summary (bool):\n            - ``True``: Return a summary of updates available for each category.\n            - ``False`` (default): Return a detailed list of available updates.\n\n        skip_installed (bool):\n            Skip updates that are already installed. Default is ``True``\n\n        skip_hidden (bool):\n            Skip updates that have been hidden. Default is ``True``\n\n        skip_mandatory (bool):\n            Skip mandatory updates. Default is ``False``\n\n        skip_reboot (bool):\n            Skip updates that require a reboot. Default is ``False``\n\n        categories (list):\n            Specify the categories to list. Must be passed as a list. All\n            categories returned by default.\n\n            Categories include the following:\n\n            * Critical Updates\n            * Definition Updates\n            * Drivers (make sure you set ``drivers=True``)\n            * Feature Packs\n            * Security Updates\n            * Update Rollups\n            * Updates\n            * Update Rollups\n            * Windows 7\n            * Windows 8.1\n            * Windows 8.1 drivers\n            * Windows 8.1 and later drivers\n            * Windows Defender\n\n        severities (list):\n            Specify the severities to include. Must be passed as a list. All\n            severities returned by default.\n\n            Severities include the following:\n\n            * Critical\n            * Important\n\n        online (bool):\n            Tells the Windows Update Agent go online to update its local update\n            database. ``True`` will go online. ``False`` will use the local\n            update database as is. Default is ``True``\n\n            .. versionadded:: 3001\n\n    Returns:\n\n        dict: Returns a dict containing either a summary or a list of updates:\n\n        .. code-block:: cfg\n\n            Dict of Updates:\n            {\'<GUID>\': {\n                \'Title\': <title>,\n                \'KB\': <KB>,\n                \'GUID\': <the globally unique identifier for the update>,\n                \'Description\': <description>,\n                \'Downloaded\': <has the update been downloaded>,\n                \'Installed\': <has the update been installed>,\n                \'Mandatory\': <is the update mandatory>,\n                \'UserInput\': <is user input required>,\n                \'EULAAccepted\': <has the EULA been accepted>,\n                \'Severity\': <update severity>,\n                \'NeedsReboot\': <is the update installed and awaiting reboot>,\n                \'RebootBehavior\': <will the update require a reboot>,\n                \'Categories\': [\n                    \'<category 1>\',\n                    \'<category 2>\',\n                    ... ]\n            }}\n\n            Summary of Updates:\n            {\'Total\': <total number of updates returned>,\n             \'Available\': <updates that are not downloaded or installed>,\n             \'Downloaded\': <updates that are downloaded but not installed>,\n             \'Installed\': <updates installed (usually 0 unless installed=True)>,\n             \'Categories\': {\n                <category 1>: <total for that category>,\n                <category 2>: <total for category 2>,\n                ... }\n            }\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        # Normal Usage (list all software updates)\n        salt \'*\' win_wua.available\n\n        # List all updates with categories of Critical Updates and Drivers\n        salt \'*\' win_wua.available categories=["Critical Updates","Drivers"]\n\n        # List all Critical Security Updates\n        salt \'*\' win_wua.available categories=["Security Updates"] severities=["Critical"]\n\n        # List all updates with a severity of Critical\n        salt \'*\' win_wua.available severities=["Critical"]\n\n        # A summary of all available updates\n        salt \'*\' win_wua.available summary=True\n\n        # A summary of all Feature Packs and Windows 8.1 Updates\n        salt \'*\' win_wua.available categories=["Feature Packs","Windows 8.1"] summary=True\n    '
    wua = salt.utils.win_update.WindowsUpdateAgent(online=online)
    updates = wua.available(skip_hidden=skip_hidden, skip_installed=skip_installed, skip_mandatory=skip_mandatory, skip_reboot=skip_reboot, software=software, drivers=drivers, categories=categories, severities=severities)
    return updates.summary() if summary else updates.list()

def get(name, download=False, install=False, online=True):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2017.7.0\n\n    Returns details for the named update\n\n    Args:\n\n        name (str):\n            The name of the update you're searching for. This can be the GUID, a\n            KB number, or any part of the name of the update. GUIDs and KBs are\n            preferred. Run ``list`` to get the GUID for the update you're\n            looking for.\n\n        download (bool):\n            Download the update returned by this function. Run this function\n            first to see if the update exists, then set ``download=True`` to\n            download the update.\n\n        install (bool):\n            Install the update returned by this function. Run this function\n            first to see if the update exists, then set ``install=True`` to\n            install the update.\n\n        online (bool):\n            Tells the Windows Update Agent go online to update its local update\n            database. ``True`` will go online. ``False`` will use the local\n            update database as is. Default is ``True``\n\n            .. versionadded:: 3001\n\n    Returns:\n\n        dict:\n            Returns a dict containing a list of updates that match the name if\n            download and install are both set to False. Should usually be a\n            single update, but can return multiple if a partial name is given.\n\n        If download or install is set to true it will return the results of the\n        operation.\n\n        .. code-block:: cfg\n\n            Dict of Updates:\n            {'<GUID>': {\n                'Title': <title>,\n                'KB': <KB>,\n                'GUID': <the globally unique identifier for the update>,\n                'Description': <description>,\n                'Downloaded': <has the update been downloaded>,\n                'Installed': <has the update been installed>,\n                'Mandatory': <is the update mandatory>,\n                'UserInput': <is user input required>,\n                'EULAAccepted': <has the EULA been accepted>,\n                'Severity': <update severity>,\n                'NeedsReboot': <is the update installed and awaiting reboot>,\n                'RebootBehavior': <will the update require a reboot>,\n                'Categories': [\n                    '<category 1>',\n                    '<category 2>',\n                    ... ]\n            }}\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        # Recommended Usage using GUID without braces\n        # Use this to find the status of a specific update\n        salt '*' win_wua.get 12345678-abcd-1234-abcd-1234567890ab\n\n        # Use the following if you don't know the GUID:\n\n        # Using a KB number\n        # Not all updates have an associated KB\n        salt '*' win_wua.get KB3030298\n\n        # Using part or all of the name of the update\n        # Could possibly return multiple results\n        # Not all updates have an associated KB\n        salt '*' win_wua.get 'Microsoft Camera Codec Pack'\n    "
    wua = salt.utils.win_update.WindowsUpdateAgent(online=online)
    updates = wua.search(name)
    ret = {}
    if download or install:
        ret['Download'] = wua.download(updates)
    if install:
        ret['Install'] = wua.install(updates)
    return ret if ret else updates.list()

def list(software=True, drivers=False, summary=False, skip_installed=True, categories=None, severities=None, download=False, install=False, online=True):
    if False:
        return 10
    "\n    .. versionadded:: 2017.7.0\n\n    Returns a detailed list of available updates or a summary. If ``download``\n    or ``install`` is ``True`` the same list will be downloaded and/or\n    installed.\n\n    Args:\n\n        software (bool):\n            Include software updates in the results. Default is ``True``\n\n        drivers (bool):\n            Include driver updates in the results. Default is ``False``\n\n        summary (bool):\n            - ``True``: Return a summary of updates available for each category.\n            - ``False`` (default): Return a detailed list of available updates.\n\n        skip_installed (bool):\n            Skip installed updates in the results. Default is ``True``\n\n        download (bool):\n            (Overrides reporting functionality) Download the list of updates\n            returned by this function. Run this function first with\n            ``download=False`` to see what will be downloaded, then set\n            ``download=True`` to download the updates. Default is ``False``\n\n        install (bool):\n            (Overrides reporting functionality) Install the list of updates\n            returned by this function. Run this function first with\n            ``install=False`` to see what will be installed, then set\n            ``install=True`` to install the updates. Default is ``False``\n\n        categories (list):\n            Specify the categories to list. Must be passed as a list. All\n            categories returned by default.\n\n            Categories include the following:\n\n            * Critical Updates\n            * Definition Updates\n            * Drivers (make sure you set ``drivers=True``)\n            * Feature Packs\n            * Security Updates\n            * Update Rollups\n            * Updates\n            * Update Rollups\n            * Windows 7\n            * Windows 8.1\n            * Windows 8.1 drivers\n            * Windows 8.1 and later drivers\n            * Windows Defender\n\n        severities (list):\n            Specify the severities to include. Must be passed as a list. All\n            severities returned by default.\n\n            Severities include the following:\n\n            * Critical\n            * Important\n\n        online (bool):\n            Tells the Windows Update Agent go online to update its local update\n            database. ``True`` will go online. ``False`` will use the local\n            update database as is. Default is ``True``\n\n            .. versionadded:: 3001\n\n    Returns:\n\n        dict: Returns a dict containing either a summary or a list of updates:\n\n        .. code-block:: cfg\n\n            Dict of Updates:\n            {'<GUID>': {\n                'Title': <title>,\n                'KB': <KB>,\n                'GUID': <the globally unique identifier for the update>,\n                'Description': <description>,\n                'Downloaded': <has the update been downloaded>,\n                'Installed': <has the update been installed>,\n                'Mandatory': <is the update mandatory>,\n                'UserInput': <is user input required>,\n                'EULAAccepted': <has the EULA been accepted>,\n                'Severity': <update severity>,\n                'NeedsReboot': <is the update installed and awaiting reboot>,\n                'RebootBehavior': <will the update require a reboot>,\n                'Categories': [\n                    '<category 1>',\n                    '<category 2>',\n                    ... ]\n            }}\n\n            Summary of Updates:\n            {'Total': <total number of updates returned>,\n             'Available': <updates that are not downloaded or installed>,\n             'Downloaded': <updates that are downloaded but not installed>,\n             'Installed': <updates installed (usually 0 unless installed=True)>,\n             'Categories': {\n                <category 1>: <total for that category>,\n                <category 2>: <total for category 2>,\n                ... }\n            }\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        # Normal Usage (list all software updates)\n        salt '*' win_wua.list\n\n        # List all updates with categories of Critical Updates and Drivers\n        salt '*' win_wua.list categories=['Critical Updates','Drivers']\n\n        # List all Critical Security Updates\n        salt '*' win_wua.list categories=['Security Updates'] severities=['Critical']\n\n        # List all updates with a severity of Critical\n        salt '*' win_wua.list severities=['Critical']\n\n        # A summary of all available updates\n        salt '*' win_wua.list summary=True\n\n        # A summary of all Feature Packs and Windows 8.1 Updates\n        salt '*' win_wua.list categories=['Feature Packs','Windows 8.1'] summary=True\n    "
    wua = salt.utils.win_update.WindowsUpdateAgent(online=online)
    updates = wua.available(skip_installed=skip_installed, software=software, drivers=drivers, categories=categories, severities=severities)
    ret = {}
    if download or install:
        ret['Download'] = wua.download(updates)
    if install:
        ret['Install'] = wua.install(updates)
    if not ret:
        return updates.summary() if summary else updates.list()
    return ret

def installed(summary=False, kbs_only=False):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 3001\n\n    Get a list of all updates that are currently installed on the system.\n\n    .. note::\n\n        This list may not necessarily match the Update History on the machine.\n        This will only show the updates that apply to the current build of\n        Windows. So, for example, the system may have shipped with Windows 10\n        Build 1607. That machine received updates to the 1607 build. Later the\n        machine was upgraded to a newer feature release, 1803 for example. Then\n        more updates were applied. This will only return the updates applied to\n        the 1803 build and not those applied when the system was at the 1607\n        build.\n\n    Args:\n\n        summary (bool):\n            Return a summary instead of a detailed list of updates. ``True``\n            will return a Summary, ``False`` will return a detailed list of\n            installed updates. Default is ``False``\n\n        kbs_only (bool):\n            Only return a list of KBs installed on the system. If this parameter\n            is passed, the ``summary`` parameter will be ignored. Default is\n            ``False``\n\n    Returns:\n        dict:\n            Returns a dictionary of either a Summary or a detailed list of\n            updates installed on the system when ``kbs_only=False``\n\n        list:\n            Returns a list of KBs installed on the system when ``kbs_only=True``\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        # Get a detailed list of all applicable updates installed on the system\n        salt '*' win_wua.installed\n\n        # Get a summary of all applicable updates installed on the system\n        salt '*' win_wua.installed summary=True\n\n        # Get a simple list of KBs installed on the system\n        salt '*' win_wua.installed kbs_only=True\n    "
    wua = salt.utils.win_update.WindowsUpdateAgent(online=False)
    updates = wua.installed()
    results = updates.list()
    if kbs_only:
        list_kbs = set()
        for item in results:
            list_kbs.update(results[item]['KBs'])
        return sorted(list_kbs)
    return updates.summary() if summary else results

def download(names):
    if False:
        return 10
    "\n    .. versionadded:: 2017.7.0\n\n    Downloads updates that match the list of passed identifiers. It's easier to\n    use this function by using list_updates and setting ``download=True``.\n\n    Args:\n\n        names (str, list):\n            A single update or a list of updates to download. This can be any\n            combination of GUIDs, KB numbers, or names. GUIDs or KBs are\n            preferred.\n\n            .. note::\n\n                An error will be raised if there are more results than there are\n                items in the names parameter\n\n    Returns:\n\n        dict: A dictionary containing the details about the downloaded updates\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Normal Usage\n        salt '*' win_wua.download names=['12345678-abcd-1234-abcd-1234567890ab', 'KB2131233']\n    "
    wua = salt.utils.win_update.WindowsUpdateAgent()
    updates = wua.search(names)
    if updates.count() == 0:
        raise CommandExecutionError('No updates found')
    if isinstance(names, str):
        names = [names]
    if isinstance(names, int):
        names = [str(names)]
    if updates.count() > len(names):
        raise CommandExecutionError('Multiple updates found, names need to be more specific')
    return wua.download(updates)

def install(names):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2017.7.0\n\n    Installs updates that match the list of identifiers. It may be easier to use\n    the list_updates function and set ``install=True``.\n\n    Args:\n\n        names (str, list):\n            A single update or a list of updates to install. This can be any\n            combination of GUIDs, KB numbers, or names. GUIDs or KBs are\n            preferred.\n\n    .. note::\n\n        An error will be raised if there are more results than there are items\n        in the names parameter\n\n    Returns:\n\n        dict: A dictionary containing the details about the installed updates\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        # Normal Usage\n        salt '*' win_wua.install KB12323211\n    "
    wua = salt.utils.win_update.WindowsUpdateAgent()
    updates = wua.search(names)
    if updates.count() == 0:
        raise CommandExecutionError('No updates found')
    if isinstance(names, str):
        names = [names]
    if isinstance(names, int):
        names = [str(names)]
    if updates.count() > len(names):
        raise CommandExecutionError('Multiple updates found, names need to be more specific')
    return wua.install(updates)

def uninstall(names):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2017.7.0\n\n    Uninstall updates.\n\n    Args:\n\n        names (str, list):\n            A single update or a list of updates to uninstall. This can be any\n            combination of GUIDs, KB numbers, or names. GUIDs or KBs are\n            preferred.\n\n    Returns:\n\n        dict: A dictionary containing the details about the uninstalled updates\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        # Normal Usage\n        salt '*' win_wua.uninstall KB3121212\n\n        # As a list\n        salt '*' win_wua.uninstall guid=['12345678-abcd-1234-abcd-1234567890ab', 'KB1231231']\n    "
    wua = salt.utils.win_update.WindowsUpdateAgent()
    updates = wua.search(names)
    if updates.count() == 0:
        raise CommandExecutionError('No updates found')
    return wua.uninstall(updates)

def set_wu_settings(level=None, recommended=None, featured=None, elevated=None, msupdate=None, day=None, time=None):
    if False:
        while True:
            i = 10
    "\n    Change Windows Update settings. If no parameters are passed, the current\n    value will be returned.\n\n    Supported:\n        - Windows Vista / Server 2008\n        - Windows 7 / Server 2008R2\n        - Windows 8 / Server 2012\n        - Windows 8.1 / Server 2012R2\n\n    .. note:\n        Microsoft began using the Unified Update Platform (UUP) starting with\n        Windows 10 / Server 2016. The Windows Update settings have changed and\n        the ability to 'Save' Windows Update settings has been removed. Windows\n        Update settings are read-only. See MSDN documentation:\n        https://msdn.microsoft.com/en-us/library/aa385829(v=vs.85).aspx\n\n    Args:\n\n        level (int):\n            Number from 1 to 4 indicating the update level:\n\n            1. Never check for updates\n            2. Check for updates but let me choose whether to download and\n               install them\n            3. Download updates but let me choose whether to install them\n            4. Install updates automatically\n\n        recommended (bool):\n            Boolean value that indicates whether to include optional or\n            recommended updates when a search for updates and installation of\n            updates is performed.\n\n        featured (bool):\n            Boolean value that indicates whether to display notifications for\n            featured updates.\n\n        elevated (bool):\n            Boolean value that indicates whether non-administrators can perform\n            some update-related actions without administrator approval.\n\n        msupdate (bool):\n            Boolean value that indicates whether to turn on Microsoft Update for\n            other Microsoft products\n\n        day (str):\n            Days of the week on which Automatic Updates installs or uninstalls\n            updates. Accepted values:\n\n            - Everyday\n            - Monday\n            - Tuesday\n            - Wednesday\n            - Thursday\n            - Friday\n            - Saturday\n\n        time (str):\n            Time at which Automatic Updates installs or uninstalls updates. Must\n            be in the ##:## 24hr format, eg. 3:00 PM would be 15:00. Must be in\n            1 hour increments.\n\n    Returns:\n\n        dict: Returns a dictionary containing the results.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' win_wua.set_wu_settings level=4 recommended=True featured=False\n    "
    ret = {'Success': True}
    with salt.utils.winapi.Com():
        obj_au = win32com.client.Dispatch('Microsoft.Update.AutoUpdate')
        obj_au_settings = obj_au.Settings
    if level is not None:
        obj_au_settings.NotificationLevel = int(level)
        result = obj_au_settings.Save()
        if result is None:
            ret['Level'] = level
        else:
            ret['Comment'] = 'Settings failed to save. Check permissions.'
            ret['Success'] = False
    if recommended is not None:
        obj_au_settings.IncludeRecommendedUpdates = recommended
        result = obj_au_settings.Save()
        if result is None:
            ret['Recommended'] = recommended
        else:
            ret['Comment'] = 'Settings failed to save. Check permissions.'
            ret['Success'] = False
    if featured is not None:
        obj_au_settings.FeaturedUpdatesEnabled = featured
        result = obj_au_settings.Save()
        if result is None:
            ret['Featured'] = featured
        else:
            ret['Comment'] = 'Settings failed to save. Check permissions.'
            ret['Success'] = False
    if elevated is not None:
        obj_au_settings.NonAdministratorsElevated = elevated
        result = obj_au_settings.Save()
        if result is None:
            ret['Elevated'] = elevated
        else:
            ret['Comment'] = 'Settings failed to save. Check permissions.'
            ret['Success'] = False
    if day is not None:
        days = {'Everyday': 0, 'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7}
        if day not in days:
            ret['Comment'] = 'Day needs to be one of the following: Everyday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday'
            ret['Success'] = False
        else:
            obj_au_settings.ScheduledInstallationDay = days[day]
            result = obj_au_settings.Save()
            if result is None:
                ret['Day'] = day
            else:
                ret['Comment'] = 'Settings failed to save. Check permissions.'
                ret['Success'] = False
    if time is not None:
        if not isinstance(time, str):
            ret['Comment'] = 'Time argument needs to be a string; it may need to be quoted. Passed {}. Time not set.'.format(time)
            ret['Success'] = False
        elif ':' not in time:
            ret['Comment'] = 'Time argument needs to be in 00:00 format. Passed {}. Time not set.'.format(time)
            ret['Success'] = False
        else:
            t = time.split(':')
            obj_au_settings.FeaturedUpdatesEnabled = t[0]
            result = obj_au_settings.Save()
            if result is None:
                ret['Time'] = time
            else:
                ret['Comment'] = 'Settings failed to save. Check permissions.'
                ret['Success'] = False
    if msupdate is not None:
        with salt.utils.winapi.Com():
            obj_sm = win32com.client.Dispatch('Microsoft.Update.ServiceManager')
            obj_sm.ClientApplicationID = 'My App'
            if msupdate:
                try:
                    obj_sm.AddService2('7971f918-a847-4430-9279-4a52d1efe18d', 7, '')
                    ret['msupdate'] = msupdate
                except Exception as error:
                    (hr, msg, exc, arg) = error.args
                    ret['Comment'] = 'Failed with failure code: {}'.format(exc[5])
                    ret['Success'] = False
            elif _get_msupdate_status():
                try:
                    obj_sm.RemoveService('7971f918-a847-4430-9279-4a52d1efe18d')
                    ret['msupdate'] = msupdate
                except Exception as error:
                    (hr, msg, exc, arg) = error.args
                    ret['Comment'] = 'Failed with failure code: {}'.format(exc[5])
                    ret['Success'] = False
            else:
                ret['msupdate'] = msupdate
    ret['Reboot'] = get_needs_reboot()
    return ret

def get_wu_settings():
    if False:
        i = 10
        return i + 15
    "\n    Get current Windows Update settings.\n\n    Returns:\n\n        dict: A dictionary of Windows Update settings:\n\n        Featured Updates:\n            Boolean value that indicates whether to display notifications for\n            featured updates.\n\n        Group Policy Required (Read-only):\n            Boolean value that indicates whether Group Policy requires the\n            Automatic Updates service.\n\n        Microsoft Update:\n            Boolean value that indicates whether to turn on Microsoft Update for\n            other Microsoft Products\n\n        Needs Reboot:\n            Boolean value that indicates whether the machine is in a reboot\n            pending state.\n\n        Non Admins Elevated:\n            Boolean value that indicates whether non-administrators can perform\n            some update-related actions without administrator approval.\n\n        Notification Level:\n\n            Number 1 to 4 indicating the update level:\n\n                1. Never check for updates\n                2. Check for updates but let me choose whether to download and\n                   install them\n                3. Download updates but let me choose whether to install them\n                4. Install updates automatically\n\n        Read Only (Read-only):\n            Boolean value that indicates whether the Automatic Update\n            settings are read-only.\n\n        Recommended Updates:\n            Boolean value that indicates whether to include optional or\n            recommended updates when a search for updates and installation of\n            updates is performed.\n\n        Scheduled Day:\n            Days of the week on which Automatic Updates installs or uninstalls\n            updates.\n\n        Scheduled Time:\n            Time at which Automatic Updates installs or uninstalls updates.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' win_wua.get_wu_settings\n    "
    ret = {}
    day = ['Every Day', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    with salt.utils.winapi.Com():
        obj_au = win32com.client.Dispatch('Microsoft.Update.AutoUpdate')
        obj_au_settings = obj_au.Settings
        ret['Featured Updates'] = obj_au_settings.FeaturedUpdatesEnabled
        ret['Group Policy Required'] = obj_au_settings.Required
        ret['Microsoft Update'] = _get_msupdate_status()
        ret['Needs Reboot'] = get_needs_reboot()
        ret['Non Admins Elevated'] = obj_au_settings.NonAdministratorsElevated
        ret['Notification Level'] = obj_au_settings.NotificationLevel
        ret['Read Only'] = obj_au_settings.ReadOnly
        ret['Recommended Updates'] = obj_au_settings.IncludeRecommendedUpdates
        ret['Scheduled Day'] = day[obj_au_settings.ScheduledInstallationDay]
        if obj_au_settings.ScheduledInstallationTime < 10:
            ret['Scheduled Time'] = '0{}:00'.format(obj_au_settings.ScheduledInstallationTime)
        else:
            ret['Scheduled Time'] = '{}:00'.format(obj_au_settings.ScheduledInstallationTime)
    return ret

def _get_msupdate_status():
    if False:
        for i in range(10):
            print('nop')
    '\n    Check to see if Microsoft Update is Enabled\n    Return Boolean\n    '
    with salt.utils.winapi.Com():
        obj_sm = win32com.client.Dispatch('Microsoft.Update.ServiceManager')
        col_services = obj_sm.Services
        for service in col_services:
            if service.name == 'Microsoft Update':
                return True
    return False

def get_needs_reboot():
    if False:
        i = 10
        return i + 15
    "\n    Determines if the system needs to be rebooted.\n\n    Returns:\n\n        bool: ``True`` if the system requires a reboot, otherwise ``False``\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' win_wua.get_needs_reboot\n    "
    return salt.utils.win_update.needs_reboot()