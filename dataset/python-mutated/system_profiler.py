"""
System Profiler Module

Interface with macOS's command-line System Profiler utility to get
information about package receipts and installed applications.

.. versionadded:: 2015.5.0

"""
import plistlib
import subprocess
import salt.utils.path
PROFILER_BINARY = '/usr/sbin/system_profiler'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Check to see if the system_profiler binary is available\n    '
    PROFILER_BINARY = salt.utils.path.which('system_profiler')
    if PROFILER_BINARY:
        return True
    return (False, 'The system_profiler execution module cannot be loaded: system_profiler unavailable.')

def _call_system_profiler(datatype):
    if False:
        i = 10
        return i + 15
    '\n    Call out to system_profiler.  Return a dictionary\n    of the stuff we are interested in.\n    '
    p = subprocess.Popen([PROFILER_BINARY, '-detailLevel', 'full', '-xml', datatype], stdout=subprocess.PIPE)
    (sysprofresults, sysprof_stderr) = p.communicate(input=None)
    plist = plistlib.readPlistFromBytes(sysprofresults)
    try:
        apps = plist[0]['_items']
    except (IndexError, KeyError):
        apps = []
    return apps

def receipts():
    if False:
        while True:
            i = 10
    "\n    Return the results of a call to\n    ``system_profiler -xml -detail full SPInstallHistoryDataType``\n    as a dictionary.  Top-level keys of the dictionary\n    are the names of each set of install receipts, since\n    there can be multiple receipts with the same name.\n    Contents of each key are a list of dictionaries.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' systemprofiler.receipts\n    "
    apps = _call_system_profiler('SPInstallHistoryDataType')
    appdict = {}
    for a in apps:
        details = dict(a)
        details.pop('_name')
        if 'install_date' in details:
            details['install_date'] = details['install_date'].strftime('%Y-%m-%d %H:%M:%S')
        if 'info' in details:
            try:
                details['info'] = '{}: {}'.format(details['info'][0], details['info'][1].strftime('%Y-%m-%d %H:%M:%S'))
            except (IndexError, AttributeError):
                pass
        if a['_name'] not in appdict:
            appdict[a['_name']] = []
        appdict[a['_name']].append(details)
    return appdict

def applications():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the results of a call to\n    ``system_profiler -xml -detail full SPApplicationsDataType``\n    as a dictionary.  Top-level keys of the dictionary\n    are the names of each set of install receipts, since\n    there can be multiple receipts with the same name.\n    Contents of each key are a list of dictionaries.\n\n    Note that this can take a long time depending on how many\n    applications are installed on the target Mac.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' systemprofiler.applications\n    "
    apps = _call_system_profiler('SPApplicationsDataType')
    appdict = {}
    for a in apps:
        details = dict(a)
        details.pop('_name')
        if 'lastModified' in details:
            details['lastModified'] = details['lastModified'].strftime('%Y-%m-%d %H:%M:%S')
        if 'info' in details:
            try:
                details['info'] = '{}: {}'.format(details['info'][0], details['info'][1].strftime('%Y-%m-%d %H:%M:%S'))
            except (IndexError, AttributeError):
                pass
        if a['_name'] not in appdict:
            appdict[a['_name']] = []
        appdict[a['_name']].append(details)
    return appdict