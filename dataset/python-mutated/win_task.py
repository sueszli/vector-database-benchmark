"""
Windows Task Scheduler Module
.. versionadded:: 2016.3.0

A module for working with the Windows Task Scheduler.
You can add and edit existing tasks.
You can add and clear triggers and actions.
You can list all tasks, folders, triggers, and actions.
"""
import logging
import time
from datetime import datetime
import salt.utils.platform
import salt.utils.winapi
from salt.exceptions import ArgumentValueError, CommandExecutionError
try:
    import pythoncom
    import pywintypes
    import win32com.client
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
log = logging.getLogger(__name__)
__virtualname__ = 'task'
TASK_ACTION_EXEC = 0
TASK_ACTION_COM_HANDLER = 5
TASK_ACTION_SEND_EMAIL = 6
TASK_ACTION_SHOW_MESSAGE = 7
TASK_COMPATIBILITY_AT = 0
TASK_COMPATIBILITY_V1 = 1
TASK_COMPATIBILITY_V2 = 2
TASK_COMPATIBILITY_V3 = 3
TASK_VALIDATE_ONLY = 1
TASK_CREATE = 2
TASK_UPDATE = 4
TASK_CREATE_OR_UPDATE = 6
TASK_DISABLE = 8
TASK_DONT_ADD_PRINCIPAL_ACE = 16
TASK_IGNORE_REGISTRATION_TRIGGERS = 32
TASK_INSTANCES_PARALLEL = 0
TASK_INSTANCES_QUEUE = 1
TASK_INSTANCES_IGNORE_NEW = 2
TASK_INSTANCES_STOP_EXISTING = 3
TASK_LOGON_NONE = 0
TASK_LOGON_PASSWORD = 1
TASK_LOGON_S4U = 2
TASK_LOGON_INTERACTIVE_TOKEN = 3
TASK_LOGON_GROUP = 4
TASK_LOGON_SERVICE_ACCOUNT = 5
TASK_LOGON_INTERACTIVE_TOKEN_OR_PASSWORD = 6
TASK_RUNLEVEL_LUA = 0
TASK_RUNLEVEL_HIGHEST = 1
TASK_STATE_UNKNOWN = 0
TASK_STATE_DISABLED = 1
TASK_STATE_QUEUED = 2
TASK_STATE_READY = 3
TASK_STATE_RUNNING = 4
TASK_TRIGGER_EVENT = 0
TASK_TRIGGER_TIME = 1
TASK_TRIGGER_DAILY = 2
TASK_TRIGGER_WEEKLY = 3
TASK_TRIGGER_MONTHLY = 4
TASK_TRIGGER_MONTHLYDOW = 5
TASK_TRIGGER_IDLE = 6
TASK_TRIGGER_REGISTRATION = 7
TASK_TRIGGER_BOOT = 8
TASK_TRIGGER_LOGON = 9
TASK_TRIGGER_SESSION_STATE_CHANGE = 11
duration = {'Immediately': 'PT0M', 'Indefinitely': '', 'Do not wait': 'PT0M', '15 seconds': 'PT15S', '30 seconds': 'PT30S', '1 minute': 'PT1M', '5 minutes': 'PT5M', '10 minutes': 'PT10M', '15 minutes': 'PT15M', '30 minutes': 'PT30M', '1 hour': 'PT1H', '2 hours': 'PT2H', '4 hours': 'PT4H', '8 hours': 'PT8H', '12 hours': 'PT12H', '1 day': ['P1D', 'PT24H'], '3 days': ['P3D', 'PT72H'], '30 days': 'P30D', '90 days': 'P90D', '180 days': 'P180D', '365 days': 'P365D'}
action_types = {'Execute': TASK_ACTION_EXEC, 'Email': TASK_ACTION_SEND_EMAIL, 'Message': TASK_ACTION_SHOW_MESSAGE}
trigger_types = {'Event': TASK_TRIGGER_EVENT, 'Once': TASK_TRIGGER_TIME, 'Daily': TASK_TRIGGER_DAILY, 'Weekly': TASK_TRIGGER_WEEKLY, 'Monthly': TASK_TRIGGER_MONTHLY, 'MonthlyDay': TASK_TRIGGER_MONTHLYDOW, 'OnIdle': TASK_TRIGGER_IDLE, 'OnTaskCreation': TASK_TRIGGER_REGISTRATION, 'OnBoot': TASK_TRIGGER_BOOT, 'OnLogon': TASK_TRIGGER_LOGON, 'OnSessionChange': TASK_TRIGGER_SESSION_STATE_CHANGE}
states = {TASK_STATE_UNKNOWN: 'Unknown', TASK_STATE_DISABLED: 'Disabled', TASK_STATE_QUEUED: 'Queued', TASK_STATE_READY: 'Ready', TASK_STATE_RUNNING: 'Running'}
instances = {'Parallel': TASK_INSTANCES_PARALLEL, 'Queue': TASK_INSTANCES_QUEUE, 'No New Instance': TASK_INSTANCES_IGNORE_NEW, 'Stop Existing': TASK_INSTANCES_STOP_EXISTING}
results = {0: 'The operation completed successfully', 1: 'Incorrect or unknown function called', 2: 'File not found', 10: 'The environment is incorrect', 267008: 'Task is ready to run at its next scheduled time', 267009: 'Task is currently running', 267010: 'Task is disabled', 267011: 'Task has not yet run', 267012: 'There are no more runs scheduled for this task', 267014: 'Task was terminated by the user', 2147750671: 'Credentials became corrupted', 2147750687: 'An instance of this task is already running', 2147946720: 'The operator or administrator has refused the request', 2147943645: 'The service is not available (Run only when logged in?)', 3221225786: 'The application terminated as a result of CTRL+C', 3228369022: 'Unknown software exception'}

def __virtual__():
    if False:
        return 10
    '\n    Only works on Windows systems\n    '
    if salt.utils.platform.is_windows():
        if not HAS_DEPENDENCIES:
            log.warning('Could not load dependencies for %s', __virtualname__)
        return __virtualname__
    return (False, 'Module win_task: module only works on Windows systems')

def _get_date_time_format(dt_string):
    if False:
        for i in range(10):
            print('nop')
    '\n    Copied from win_system.py (_get_date_time_format)\n\n    Function that detects the date/time format for the string passed.\n\n    :param str dt_string:\n        A date/time string\n\n    :return: The format of the passed dt_string\n    :rtype: str\n    '
    valid_formats = ['%I:%M:%S %p', '%I:%M %p', '%H:%M:%S', '%H:%M', '%Y-%m-%d', '%m-%d-%y', '%m-%d-%Y', '%m/%d/%y', '%m/%d/%Y', '%Y/%m/%d']
    for dt_format in valid_formats:
        try:
            datetime.strptime(dt_string, dt_format)
            return dt_format
        except ValueError:
            continue
    return False

def _get_date_value(date):
    if False:
        for i in range(10):
            print('nop')
    '\n    Function for dealing with PyTime values with invalid dates. ie: 12/30/1899\n    which is the windows task scheduler value for Never\n\n    :param obj date: A PyTime object\n\n    :return: A string value representing the date or the word "Never" for\n    invalid date strings\n    :rtype: str\n    '
    try:
        return '{}'.format(date)
    except ValueError:
        return 'Never'

def _reverse_lookup(dictionary, value):
    if False:
        while True:
            i = 10
    '\n    Lookup the key in a dictionary by its value. Will return the first match.\n\n    :param dict dictionary: The dictionary to search\n\n    :param str value: The value to search for.\n\n    :return: Returns the first key to match the value\n    :rtype: str\n    '
    value_index = -1
    for (idx, dict_value) in enumerate(dictionary.values()):
        if type(dict_value) == list:
            if value in dict_value:
                value_index = idx
                break
        elif value == dict_value:
            value_index = idx
            break
    return list(dictionary)[value_index]

def _lookup_first(dictionary, key):
    if False:
        return 10
    '\n    Lookup the first value given a key. Returns the first value if the key\n    refers to a list or the value itself.\n\n    :param dict dictionary: The dictionary to search\n\n    :param str key: The key to get\n\n    :return: Returns the first value available for the key\n    :rtype: str\n    '
    value = dictionary[key]
    if type(value) == list:
        return value[0]
    else:
        return value

def _save_task_definition(name, task_folder, task_definition, user_name, password, logon_type):
    if False:
        print('Hello World!')
    '\n    Internal function to save the task definition.\n\n    :param str name: The name of the task.\n\n    :param str task_folder: The object representing the folder in which to save\n    the task\n\n    :param str task_definition: The object representing the task to be saved\n\n    :param str user_name: The user_account under which to run the task\n\n    :param str password: The password that corresponds to the user account\n\n    :param int logon_type: The logon type for the task.\n\n    :return: True if successful, False if not\n    :rtype: bool\n    '
    try:
        task_folder.RegisterTaskDefinition(name, task_definition, TASK_CREATE_OR_UPDATE, user_name, password, logon_type)
        return True
    except pythoncom.com_error as error:
        (hr, msg, exc, arg) = error.args
        fc = {-2147024773: 'The filename, directory name, or volume label syntax is incorrect', -2147024894: 'The system cannot find the file specified', -2147216615: 'Required element or attribute missing', -2147216616: 'Value incorrectly formatted or out of range', -2147352571: 'Access denied'}
        try:
            failure_code = fc[exc[5]]
        except KeyError:
            failure_code = 'Unknown Failure: {}'.format(error)
        log.debug('Failed to modify task: %s', failure_code)
        return 'Failed to modify task: {}'.format(failure_code)

def list_tasks(location='\\'):
    if False:
        return 10
    "\n    List all tasks located in a specific location in the task scheduler.\n\n    Args:\n\n        location (str):\n            A string value representing the folder from which you want to list\n            tasks. Default is ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        list: Returns a list of tasks\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # List all tasks in the default location\n        salt 'minion-id' task.list_tasks\n\n        # List all tasks in the Microsoft\\XblGameSave Directory\n        salt 'minion-id' task.list_tasks Microsoft\\XblGameSave\n    "
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        try:
            task_folder = task_service.GetFolder(location)
        except pywintypes.com_error:
            msg = 'Unable to load location: {}'.format(location)
            log.error(msg)
            raise CommandExecutionError(msg)
        tasks = task_folder.GetTasks(0)
        ret = []
        for task in tasks:
            ret.append(task.Name)
    return ret

def list_folders(location='\\'):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all folders located in a specific location in the task scheduler.\n\n    Args:\n\n        location (str):\n            A string value representing the folder from which you want to list\n            tasks. Default is ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        list: Returns a list of folders.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # List all folders in the default location\n        salt 'minion-id' task.list_folders\n\n        # List all folders in the Microsoft directory\n        salt 'minion-id' task.list_folders Microsoft\n    "
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        folders = task_folder.GetFolders(0)
        ret = []
        for folder in folders:
            ret.append(folder.Name)
    return ret

def list_triggers(name, location='\\'):
    if False:
        print('Hello World!')
    "\n    List all triggers that pertain to a task in the specified location.\n\n    Args:\n\n        name (str):\n            The name of the task for which list triggers.\n\n        location (str):\n            A string value representing the location of the task from which to\n            list triggers. Default is ``\\`` which is the root for the task\n            scheduler (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        list: Returns a list of triggers.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # List all triggers for a task in the default location\n        salt 'minion-id' task.list_triggers <task_name>\n\n        # List all triggers for the XblGameSaveTask in the Microsoft\\XblGameSave\n        # location\n        salt '*' task.list_triggers XblGameSaveTask Microsoft\\XblGameSave\n    "
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task_definition = task_folder.GetTask(name).Definition
        triggers = task_definition.Triggers
        ret = []
        for trigger in triggers:
            ret.append(trigger.Id)
    return ret

def list_actions(name, location='\\'):
    if False:
        print('Hello World!')
    "\n    List all actions that pertain to a task in the specified location.\n\n    Args:\n\n        name (str):\n            The name of the task for which list actions.\n\n        location (str):\n            A string value representing the location of the task from which to\n            list actions. Default is ``\\`` which is the root for the task\n            scheduler (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        list: Returns a list of actions.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # List all actions for a task in the default location\n        salt 'minion-id' task.list_actions <task_name>\n\n        # List all actions for the XblGameSaveTask in the Microsoft\\XblGameSave\n        # location\n        salt 'minion-id' task.list_actions XblGameSaveTask Microsoft\\XblGameSave\n    "
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task_definition = task_folder.GetTask(name).Definition
        actions = task_definition.Actions
        ret = []
        for action in actions:
            ret.append(action.Id)
    return ret

def create_task(name, location='\\', user_name='System', password=None, force=False, **kwargs):
    if False:
        print('Hello World!')
    '\n    Create a new task in the designated location. This function has many keyword\n    arguments that are not listed here. For additional arguments see:\n\n        - :py:func:`edit_task`\n        - :py:func:`add_action`\n        - :py:func:`add_trigger`\n\n    Args:\n\n        name (str):\n            The name of the task. This will be displayed in the task scheduler.\n\n        location (str):\n            A string value representing the location in which to create the\n            task. Default is ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n        user_name (str):\n            The user account under which to run the task. To specify the\n            \'System\' account, use \'System\'. The password will be ignored.\n\n        password (str):\n            The password to use for authentication. This should set the task to\n            run whether the user is logged in or not, but is currently not\n            working.\n\n        force (bool):\n            If the task exists, overwrite the existing task.\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'minion-id\' task.create_task <task_name> user_name=System force=True action_type=Execute cmd=\'del /Q /S C:\\\\Temp\' trigger_type=Once start_date=2016-12-1 start_time=\'"01:00"\'\n    '
    if name in list_tasks(location) and (not force):
        return '{} already exists'.format(name)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_definition = task_service.NewTask(0)
        edit_task(task_definition=task_definition, user_name=user_name, password=password, **kwargs)
        add_action(task_definition=task_definition, **kwargs)
        add_trigger(task_definition=task_definition, **kwargs)
        task_folder = task_service.GetFolder(location)
        _save_task_definition(name=name, task_folder=task_folder, task_definition=task_definition, user_name=task_definition.Principal.UserID, password=password, logon_type=task_definition.Principal.LogonType)
    return name in list_tasks(location)

def create_task_from_xml(name, location='\\', xml_text=None, xml_path=None, user_name='System', password=None):
    if False:
        return 10
    "\n    Create a task based on XML. Source can be a file or a string of XML.\n\n    Args:\n\n        name (str):\n            The name of the task. This will be displayed in the task scheduler.\n\n        location (str):\n            A string value representing the location in which to create the\n            task. Default is ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n        xml_text (str):\n            A string of xml representing the task to be created. This will be\n            overridden by ``xml_path`` if passed.\n\n        xml_path (str):\n            The path to an XML file on the local system containing the xml that\n            defines the task. This will override ``xml_text``\n\n        user_name (str):\n            The user account under which to run the task. To specify the\n            'System' account, use 'System'. The password will be ignored.\n\n        password (str):\n            The password to use for authentication. This should set the task to\n            run whether the user is logged in or not, but is currently not\n            working.\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n        str: A string with the error message if there is an error\n\n    Raises:\n        ArgumentValueError: If arguments are invalid\n        CommandExecutionError\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' task.create_task_from_xml <task_name> xml_path=C:\\task.xml\n    "
    if name in list_tasks(location):
        return '{} already exists'.format(name)
    if not xml_text and (not xml_path):
        raise ArgumentValueError('Must specify either xml_text or xml_path')
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        if xml_path:
            xml_text = xml_path
        task_folder = task_service.GetFolder(location)
        if user_name:
            if user_name.lower() == 'system':
                logon_type = TASK_LOGON_SERVICE_ACCOUNT
                user_name = 'SYSTEM'
                password = None
            elif password:
                logon_type = TASK_LOGON_PASSWORD
            else:
                logon_type = TASK_LOGON_INTERACTIVE_TOKEN
        else:
            password = None
            logon_type = TASK_LOGON_NONE
        try:
            task_folder.RegisterTask(name, xml_text, TASK_CREATE, user_name, password, logon_type)
        except pythoncom.com_error as error:
            (hr, msg, exc, arg) = error.args
            error_code = hex(exc[5] + 2 ** 32)
            fc = {2147750681: 'Required element or attribute missing', 2147750680: 'Value incorrectly formatted or out of range', 2147614725: 'Access denied', 2147750665: "A task's trigger is not found", 2147750666: 'One or more of the properties required to run this task have not been set', 2147750668: 'The Task Scheduler service is not installed on this computer', 2147750669: 'The task object could not be opened', 2147750670: 'The object is either an invalid task object or is not a task object', 2147750671: 'No account information could be found in the Task Scheduler security database for the task indicated', 2147750672: 'Unable to establish existence of the account specified', 2147750673: 'Corruption was detected in the Task Scheduler security database; the database has been reset', 2147750675: 'The task object version is either unsupported or invalid', 2147750676: 'The task has been configured with an unsupported combination of account settings and run time options', 2147750677: 'The Task Scheduler Service is not running', 2147750678: 'The task XML contains an unexpected node', 2147750679: 'The task XML contains an element or attribute from an unexpected namespace', 2147750682: 'The task XML is malformed', 267036: 'The task is registered, but may fail to start. Batch logon privilege needs to be enabled for the task principal', 2147750685: 'The task XML contains too many nodes of the same type'}
            try:
                failure_code = fc[error_code]
            except KeyError:
                failure_code = 'Unknown Failure: {}'.format(error_code)
            finally:
                log.debug('Failed to create task: %s', failure_code)
            raise CommandExecutionError(failure_code)
    return name in list_tasks(location)

def create_folder(name, location='\\'):
    if False:
        return 10
    "\n    Create a folder in which to create tasks.\n\n    Args:\n\n        name (str):\n            The name of the folder. This will be displayed in the task\n            scheduler.\n\n        location (str):\n            A string value representing the location in which to create the\n            folder. Default is ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion-id' task.create_folder <folder_name>\n    "
    if name in list_folders(location):
        return '{} already exists'.format(name)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task_folder.CreateFolder(name)
    return name in list_folders(location)

def edit_task(name=None, location='\\', user_name=None, password=None, description=None, enabled=None, hidden=None, run_if_idle=None, idle_duration=None, idle_wait_timeout=None, idle_stop_on_end=None, idle_restart=None, ac_only=None, stop_if_on_batteries=None, wake_to_run=None, run_if_network=None, network_id=None, network_name=None, allow_demand_start=None, start_when_available=None, restart_every=None, restart_count=3, execution_time_limit=None, force_stop=None, delete_after=None, multiple_instances=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Edit the parameters of a task. Triggers and Actions cannot be edited yet.\n\n    Args:\n\n        name (str):\n            The name of the task. This will be displayed in the task scheduler.\n\n        location (str):\n            A string value representing the location in which to create the\n            task. Default is ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n        user_name (str):\n            The user account under which to run the task. To specify the\n            'System' account, use 'System'. The password will be ignored.\n\n        password (str):\n            The password to use for authentication. This should set the task to\n            run whether the user is logged in or not, but is currently not\n            working.\n\n            .. note::\n\n                The combination of user_name and password determine how the\n                task runs. For example, if a username is passed without at\n                password the task will only run when the user is logged in. If a\n                password is passed as well the task will run whether the user is\n                logged on or not. If you pass 'System' as the username the task\n                will run as the system account (the password parameter is\n                ignored).\n\n        description (str):\n            A string representing the text that will be displayed in the\n            description field in the task scheduler.\n\n        enabled (bool):\n            A boolean value representing whether or not the task is enabled.\n\n        hidden (bool):\n            A boolean value representing whether or not the task is hidden.\n\n        run_if_idle (bool):\n            Boolean value that indicates that the Task Scheduler will run the\n            task only if the computer is in an idle state.\n\n        idle_duration (str):\n            A value that indicates the amount of time that the computer must be\n            in an idle state before the task is run. Valid values are:\n\n                - 1 minute\n                - 5 minutes\n                - 10 minutes\n                - 15 minutes\n                - 30 minutes\n                - 1 hour\n\n        idle_wait_timeout (str):\n            A value that indicates the amount of time that the Task Scheduler\n            will wait for an idle condition to occur. Valid values are:\n\n                - Do not wait\n                - 1 minute\n                - 5 minutes\n                - 10 minutes\n                - 15 minutes\n                - 30 minutes\n                - 1 hour\n                - 2 hours\n\n        idle_stop_on_end (bool):\n            Boolean value that indicates that the Task Scheduler will terminate\n            the task if the idle condition ends before the task is completed.\n\n        idle_restart (bool):\n            Boolean value that indicates whether the task is restarted when the\n            computer cycles into an idle condition more than once.\n\n        ac_only (bool):\n            Boolean value that indicates that the Task Scheduler will launch the\n            task only while on AC power.\n\n        stop_if_on_batteries (bool):\n            Boolean value that indicates that the task will be stopped if the\n            computer begins to run on battery power.\n\n        wake_to_run (bool):\n            Boolean value that indicates that the Task Scheduler will wake the\n            computer when it is time to run the task.\n\n        run_if_network (bool):\n            Boolean value that indicates that the Task Scheduler will run the\n            task only when a network is available.\n\n        network_id (guid):\n            GUID value that identifies a network profile.\n\n        network_name (str):\n            Sets the name of a network profile. The name is used for display\n            purposes.\n\n        allow_demand_start (bool):\n            Boolean value that indicates that the task can be started by using\n            either the Run command or the Context menu.\n\n        start_when_available (bool):\n            Boolean value that indicates that the Task Scheduler can start the\n            task at any time after its scheduled time has passed.\n\n        restart_every (str):\n            A value that specifies the interval between task restart attempts.\n            Valid values are:\n\n                - False (to disable)\n                - 1 minute\n                - 5 minutes\n                - 10 minutes\n                - 15 minutes\n                - 30 minutes\n                - 1 hour\n                - 2 hours\n\n        restart_count (int):\n            The number of times the Task Scheduler will attempt to restart the\n            task. Valid values are integers 1 - 999.\n\n        execution_time_limit (bool, str):\n            The amount of time allowed to complete the task. Valid values are:\n\n                - False (to disable)\n                - 1 hour\n                - 2 hours\n                - 4 hours\n                - 8 hours\n                - 12 hours\n                - 1 day\n                - 3 days\n\n        force_stop (bool):\n            Boolean value that indicates that the task may be terminated by\n            using TerminateProcess.\n\n        delete_after (bool, str):\n            The amount of time that the Task Scheduler will wait before deleting\n            the task after it expires. Requires a trigger with an expiration\n            date. Valid values are:\n\n                - False (to disable)\n                - Immediately\n                - 30 days\n                - 90 days\n                - 180 days\n                - 365 days\n\n        multiple_instances (str):\n            Sets the policy that defines how the Task Scheduler deals with\n            multiple instances of the task. Valid values are:\n\n                - Parallel\n                - Queue\n                - No New Instance\n                - Stop Existing\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' task.edit_task <task_name> description='This task is awesome'\n    "
    with salt.utils.winapi.Com():
        save_definition = False
        if kwargs.get('task_definition', False):
            task_definition = kwargs.get('task_definition')
        else:
            save_definition = True
            if not name:
                return 'Required parameter "name" not passed'
            if name in list_tasks(location):
                task_service = win32com.client.Dispatch('Schedule.Service')
                task_service.Connect()
                task_folder = task_service.GetFolder(location)
                task_definition = task_folder.GetTask(name).Definition
            else:
                return '{} not found'.format(name)
        if save_definition:
            task_definition.RegistrationInfo.Author = 'Salt Minion'
            task_definition.RegistrationInfo.Source = 'Salt Minion Daemon'
        if description is not None:
            task_definition.RegistrationInfo.Description = description
        if user_name:
            if user_name.lower() == 'system':
                logon_type = TASK_LOGON_SERVICE_ACCOUNT
                user_name = 'SYSTEM'
                password = None
            else:
                task_definition.Principal.Id = user_name
                if password:
                    logon_type = TASK_LOGON_PASSWORD
                else:
                    logon_type = TASK_LOGON_INTERACTIVE_TOKEN
            task_definition.Principal.UserID = user_name
            task_definition.Principal.DisplayName = user_name
            task_definition.Principal.LogonType = logon_type
            task_definition.Principal.RunLevel = TASK_RUNLEVEL_HIGHEST
        else:
            user_name = None
            password = None
        if enabled is not None:
            task_definition.Settings.Enabled = enabled
        if hidden is not None:
            task_definition.Settings.Hidden = hidden
        if run_if_idle is not None:
            task_definition.Settings.RunOnlyIfIdle = run_if_idle
        if task_definition.Settings.RunOnlyIfIdle:
            if idle_stop_on_end is not None:
                task_definition.Settings.IdleSettings.StopOnIdleEnd = idle_stop_on_end
            if idle_restart is not None:
                task_definition.Settings.IdleSettings.RestartOnIdle = idle_restart
            if idle_duration is not None:
                if idle_duration in duration:
                    task_definition.Settings.IdleSettings.IdleDuration = _lookup_first(duration, idle_duration)
                else:
                    return 'Invalid value for "idle_duration"'
            if idle_wait_timeout is not None:
                if idle_wait_timeout in duration:
                    task_definition.Settings.IdleSettings.WaitTimeout = _lookup_first(duration, idle_wait_timeout)
                else:
                    return 'Invalid value for "idle_wait_timeout"'
        if ac_only is not None:
            task_definition.Settings.DisallowStartIfOnBatteries = ac_only
        if stop_if_on_batteries is not None:
            task_definition.Settings.StopIfGoingOnBatteries = stop_if_on_batteries
        if wake_to_run is not None:
            task_definition.Settings.WakeToRun = wake_to_run
        if run_if_network is not None:
            task_definition.Settings.RunOnlyIfNetworkAvailable = run_if_network
        if task_definition.Settings.RunOnlyIfNetworkAvailable:
            if network_id:
                task_definition.Settings.NetworkSettings.Id = network_id
            if network_name:
                task_definition.Settings.NetworkSettings.Name = network_name
        if allow_demand_start is not None:
            task_definition.Settings.AllowDemandStart = allow_demand_start
        if start_when_available is not None:
            task_definition.Settings.StartWhenAvailable = start_when_available
        if restart_every is not None:
            if restart_every is False:
                task_definition.Settings.RestartInterval = ''
            elif restart_every in duration:
                task_definition.Settings.RestartInterval = _lookup_first(duration, restart_every)
            else:
                return 'Invalid value for "restart_every"'
        if task_definition.Settings.RestartInterval:
            if restart_count is not None:
                if restart_count in range(1, 999):
                    task_definition.Settings.RestartCount = restart_count
                else:
                    return '"restart_count" must be a value between 1 and 999'
        if execution_time_limit is not None:
            if execution_time_limit is False:
                task_definition.Settings.ExecutionTimeLimit = 'PT0S'
            elif execution_time_limit in duration:
                task_definition.Settings.ExecutionTimeLimit = _lookup_first(duration, execution_time_limit)
            else:
                return 'Invalid value for "execution_time_limit"'
        if force_stop is not None:
            task_definition.Settings.AllowHardTerminate = force_stop
        if delete_after is not None:
            if delete_after is False:
                task_definition.Settings.DeleteExpiredTaskAfter = ''
            elif delete_after in duration:
                task_definition.Settings.DeleteExpiredTaskAfter = _lookup_first(duration, delete_after)
            else:
                return 'Invalid value for "delete_after"'
        if multiple_instances is not None:
            task_definition.Settings.MultipleInstances = instances[multiple_instances]
        if save_definition:
            return _save_task_definition(name=name, task_folder=task_folder, task_definition=task_definition, user_name=user_name, password=password, logon_type=task_definition.Principal.LogonType)

def delete_task(name, location='\\'):
    if False:
        print('Hello World!')
    "\n    Delete a task from the task scheduler.\n\n    Args:\n        name (str):\n            The name of the task to delete.\n\n        location (str):\n            A string value representing the location of the task. Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion-id' task.delete_task <task_name>\n    "
    if name not in list_tasks(location):
        return '{} not found in {}'.format(name, location)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task_folder.DeleteTask(name, 0)
    return name not in list_tasks(location)

def delete_folder(name, location='\\'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete a folder from the task scheduler.\n\n    Args:\n\n        name (str):\n            The name of the folder to delete.\n\n        location (str):\n            A string value representing the location of the folder.  Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion-id' task.delete_folder <folder_name>\n    "
    if name not in list_folders(location):
        return '{} not found in {}'.format(name, location)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task_folder.DeleteFolder(name, 0)
    return name not in list_folders(location)

def run(name, location='\\'):
    if False:
        return 10
    "\n    Run a scheduled task manually.\n\n    Args:\n\n        name (str):\n            The name of the task to run.\n\n        location (str):\n            A string value representing the location of the task. Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion-id' task.run <task_name>\n    "
    if name not in list_tasks(location):
        return '{} not found in {}'.format(name, location)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task = task_folder.GetTask(name)
        try:
            task.Run('')
            return True
        except pythoncom.com_error:
            return False

def run_wait(name, location='\\'):
    if False:
        return 10
    "\n    Run a scheduled task and return when the task finishes\n\n    Args:\n\n        name (str):\n            The name of the task to run.\n\n        location (str):\n            A string value representing the location of the task. Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion-id' task.run_wait <task_name>\n    "
    if name not in list_tasks(location):
        return '{} not found in {}'.format(name, location)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task = task_folder.GetTask(name)
        if task.State == TASK_STATE_RUNNING:
            return 'Task already running'
        try:
            task.Run('')
            time.sleep(1)
            running = True
        except pythoncom.com_error:
            return False
        while running:
            running = False
            try:
                running_tasks = task_service.GetRunningTasks(0)
                if running_tasks.Count:
                    for item in running_tasks:
                        if item.Name == name:
                            running = True
            except pythoncom.com_error:
                running = False
    return True

def stop(name, location='\\'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Stop a scheduled task.\n\n    Args:\n\n        name (str):\n            The name of the task to stop.\n\n        location (str):\n            A string value representing the location of the task. Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion-id' task.list_stop <task_name>\n    "
    if name not in list_tasks(location):
        return '{} not found in {}'.format(name, location)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task = task_folder.GetTask(name)
        try:
            task.Stop(0)
            return True
        except pythoncom.com_error:
            return False

def status(name, location='\\'):
    if False:
        return 10
    "\n    Determine the status of a task. Is it Running, Queued, Ready, etc.\n\n    Args:\n\n        name (str):\n            The name of the task for which to return the status\n\n        location (str):\n            A string value representing the location of the task. Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        str: The current status of the task. Will be one of the following:\n\n            - Unknown\n            - Disabled\n            - Queued\n            - Ready\n            - Running\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion-id' task.list_status <task_name>\n    "
    if name not in list_tasks(location):
        return '{} not found in {}'.format(name, location)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task = task_folder.GetTask(name)
        return states[task.State]

def info(name, location='\\'):
    if False:
        while True:
            i = 10
    "\n    Get the details about a task in the task scheduler.\n\n    Args:\n\n        name (str):\n            The name of the task for which to return the status\n\n        location (str):\n            A string value representing the location of the task. Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        dict: A dictionary containing the task configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion-id' task.info <task_name>\n    "
    if name not in list_tasks(location):
        return '{} not found in {}'.format(name, location)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task = task_folder.GetTask(name)
        properties = {'enabled': task.Enabled, 'last_run': _get_date_value(task.LastRunTime), 'last_run_result': results[task.LastTaskResult], 'missed_runs': task.NumberOfMissedRuns, 'next_run': _get_date_value(task.NextRunTime), 'status': states[task.State]}
        def_set = task.Definition.Settings
        settings = {'allow_demand_start': def_set.AllowDemandStart, 'force_stop': def_set.AllowHardTerminate}
        if def_set.DeleteExpiredTaskAfter == '':
            settings['delete_after'] = False
        elif def_set.DeleteExpiredTaskAfter == 'PT0S':
            settings['delete_after'] = 'Immediately'
        else:
            settings['delete_after'] = _reverse_lookup(duration, def_set.DeleteExpiredTaskAfter)
        if def_set.ExecutionTimeLimit == '':
            settings['execution_time_limit'] = False
        else:
            settings['execution_time_limit'] = _reverse_lookup(duration, def_set.ExecutionTimeLimit)
        settings['multiple_instances'] = _reverse_lookup(instances, def_set.MultipleInstances)
        if def_set.RestartInterval == '':
            settings['restart_interval'] = False
        else:
            settings['restart_interval'] = _reverse_lookup(duration, def_set.RestartInterval)
        if settings['restart_interval']:
            settings['restart_count'] = def_set.RestartCount
        settings['stop_if_on_batteries'] = def_set.StopIfGoingOnBatteries
        settings['wake_to_run'] = def_set.WakeToRun
        conditions = {'ac_only': def_set.DisallowStartIfOnBatteries, 'run_if_idle': def_set.RunOnlyIfIdle, 'run_if_network': def_set.RunOnlyIfNetworkAvailable, 'start_when_available': def_set.StartWhenAvailable}
        if conditions['run_if_idle']:
            idle_set = def_set.IdleSettings
            conditions['idle_duration'] = idle_set.IdleDuration
            conditions['idle_restart'] = idle_set.RestartOnIdle
            conditions['idle_stop_on_end'] = idle_set.StopOnIdleEnd
            conditions['idle_wait_timeout'] = idle_set.WaitTimeout
        if conditions['run_if_network']:
            net_set = def_set.NetworkSettings
            conditions['network_id'] = net_set.Id
            conditions['network_name'] = net_set.Name
        actions = []
        for actionObj in task.Definition.Actions:
            action = {'action_type': _reverse_lookup(action_types, actionObj.Type)}
            if actionObj.Path:
                action['cmd'] = actionObj.Path
            if actionObj.Arguments:
                action['arguments'] = actionObj.Arguments
            if actionObj.WorkingDirectory:
                action['working_dir'] = actionObj.WorkingDirectory
            actions.append(action)
        triggers = []
        for triggerObj in task.Definition.Triggers:
            trigger = {'trigger_type': _reverse_lookup(trigger_types, triggerObj.Type)}
            if triggerObj.ExecutionTimeLimit:
                trigger['execution_time_limit'] = _reverse_lookup(duration, triggerObj.ExecutionTimeLimit)
            if triggerObj.StartBoundary:
                (start_date, start_time) = triggerObj.StartBoundary.split('T', 1)
                trigger['start_date'] = start_date
                trigger['start_time'] = start_time
            if triggerObj.EndBoundary:
                (end_date, end_time) = triggerObj.EndBoundary.split('T', 1)
                trigger['end_date'] = end_date
                trigger['end_time'] = end_time
            trigger['enabled'] = triggerObj.Enabled
            if hasattr(triggerObj, 'RandomDelay'):
                if triggerObj.RandomDelay:
                    trigger['random_delay'] = _reverse_lookup(duration, triggerObj.RandomDelay)
                else:
                    trigger['random_delay'] = False
            if hasattr(triggerObj, 'Delay'):
                if triggerObj.Delay:
                    trigger['delay'] = _reverse_lookup(duration, triggerObj.Delay)
                else:
                    trigger['delay'] = False
            if hasattr(triggerObj, 'Repetition'):
                trigger['repeat_duration'] = _reverse_lookup(duration, triggerObj.Repetition.Duration)
                trigger['repeat_interval'] = _reverse_lookup(duration, triggerObj.Repetition.Interval)
                trigger['repeat_stop_at_duration_end'] = triggerObj.Repetition.StopAtDurationEnd
            triggers.append(trigger)
        properties['settings'] = settings
        properties['conditions'] = conditions
        properties['actions'] = actions
        properties['triggers'] = triggers
        ret = properties
    return ret

def add_action(name=None, location='\\', action_type='Execute', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add an action to a task.\n\n    Args:\n\n        name (str):\n            The name of the task to which to add the action.\n\n        location (str):\n            A string value representing the location of the task. Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n        action_type (str):\n            The type of action to add. There are three action types. Each one\n            requires its own set of Keyword Arguments (kwargs). Valid values\n            are:\n\n                - Execute\n                - Email\n                - Message\n\n    Required arguments for each action_type:\n\n    **Execute**\n\n        Execute a command or an executable\n\n            cmd (str):\n                (required) The command or executable to run.\n\n            arguments (str):\n                (optional) Arguments to be passed to the command or executable.\n                To launch a script the first command will need to be the\n                interpreter for the script. For example, to run a vbscript you\n                would pass ``cscript.exe`` in the ``cmd`` parameter and pass the\n                script in the ``arguments`` parameter as follows:\n\n                    - ``cmd=\'cscript.exe\' arguments=\'c:\\scripts\\myscript.vbs\'``\n\n                Batch files do not need an interpreter and may be passed to the\n                cmd parameter directly.\n\n            start_in (str):\n                (optional) The current working directory for the command.\n\n    **Email**\n\n        Send and email. Requires ``server``, ``from``, and ``to`` or ``cc``.\n\n            from (str): The sender\n\n            reply_to (str): Who to reply to\n\n            to (str): The recipient\n\n            cc (str): The CC recipient\n\n            bcc (str): The BCC recipient\n\n            subject (str): The subject of the email\n\n            body (str): The Message Body of the email\n\n            server (str): The server used to send the email\n\n            attachments (list):\n                A list of attachments. These will be the paths to the files to\n                attach. ie: ``attachments="[\'C:\\attachment1.txt\',\n                \'C:\\attachment2.txt\']"``\n\n    **Message**\n\n        Display a dialog box. The task must be set to "Run only when user is\n        logged on" in order for the dialog box to display. Both parameters are\n        required.\n\n            title (str):\n                The dialog box title.\n\n            message (str):\n                The dialog box message body\n\n    Returns:\n        dict: A dictionary containing the task configuration\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'minion-id\' task.add_action <task_name> cmd=\'del /Q /S C:\\\\Temp\'\n    '
    with salt.utils.winapi.Com():
        save_definition = False
        if kwargs.get('task_definition', False):
            task_definition = kwargs.get('task_definition')
        else:
            save_definition = True
            if not name:
                return 'Required parameter "name" not passed'
            if name in list_tasks(location):
                task_service = win32com.client.Dispatch('Schedule.Service')
                task_service.Connect()
                task_folder = task_service.GetFolder(location)
                task_definition = task_folder.GetTask(name).Definition
            else:
                return '{} not found'.format(name)
        task_action = task_definition.Actions.Create(action_types[action_type])
        if action_types[action_type] == TASK_ACTION_EXEC:
            task_action.Id = 'Execute_ID1'
            if kwargs.get('cmd', False):
                task_action.Path = kwargs.get('cmd')
            else:
                return 'Required parameter "cmd" not found'
            task_action.Arguments = kwargs.get('arguments', '')
            task_action.WorkingDirectory = kwargs.get('start_in', '')
        elif action_types[action_type] == TASK_ACTION_SEND_EMAIL:
            task_action.Id = 'Email_ID1'
            if kwargs.get('server', False):
                task_action.Server = kwargs.get('server')
            else:
                return 'Required parameter "server" not found'
            if kwargs.get('from', False):
                task_action.From = kwargs.get('from')
            else:
                return 'Required parameter "from" not found'
            if kwargs.get('to', False) or kwargs.get('cc', False):
                if kwargs.get('to'):
                    task_action.To = kwargs.get('to')
                if kwargs.get('cc'):
                    task_action.Cc = kwargs.get('cc')
            else:
                return 'Required parameter "to" or "cc" not found'
            if kwargs.get('reply_to'):
                task_action.ReplyTo = kwargs.get('reply_to')
            if kwargs.get('bcc'):
                task_action.Bcc = kwargs.get('bcc')
            if kwargs.get('subject'):
                task_action.Subject = kwargs.get('subject')
            if kwargs.get('body'):
                task_action.Body = kwargs.get('body')
            if kwargs.get('attachments'):
                task_action.Attachments = kwargs.get('attachments')
        elif action_types[action_type] == TASK_ACTION_SHOW_MESSAGE:
            task_action.Id = 'Message_ID1'
            if kwargs.get('title', False):
                task_action.Title = kwargs.get('title')
            else:
                return 'Required parameter "title" not found'
            if kwargs.get('message', False):
                task_action.MessageBody = kwargs.get('message')
            else:
                return 'Required parameter "message" not found'
        if save_definition:
            return _save_task_definition(name=name, task_folder=task_folder, task_definition=task_definition, user_name=task_definition.Principal.UserID, password=None, logon_type=task_definition.Principal.LogonType)

def _clear_actions(name, location='\\'):
    if False:
        i = 10
        return i + 15
    '\n    Remove all actions from the task.\n\n    :param str name: The name of the task from which to clear all actions.\n\n    :param str location: A string value representing the location of the task.\n    Default is ``\\`` which is the root for the task scheduler\n    (``C:\\Windows\\System32\\tasks``).\n\n    :return: True if successful, False if unsuccessful\n    :rtype: bool\n    '
    if name not in list_tasks(location):
        return '{} not found in {}'.format(name, location)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task_definition = task_folder.GetTask(name).Definition
        actions = task_definition.Actions
        actions.Clear()
        return _save_task_definition(name=name, task_folder=task_folder, task_definition=task_definition, user_name=task_definition.Principal.UserID, password=None, logon_type=task_definition.Principal.LogonType)

def add_trigger(name=None, location='\\', trigger_type=None, trigger_enabled=True, start_date=None, start_time=None, end_date=None, end_time=None, random_delay=None, repeat_interval=None, repeat_duration=None, repeat_stop_at_duration_end=False, execution_time_limit=None, delay=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add a trigger to a Windows Scheduled task\n\n    .. note::\n\n        Arguments are parsed by the YAML loader and are subject to\n        yaml\'s idiosyncrasies. Therefore, time values in some\n        formats (``%H:%M:%S`` and ``%H:%M``) should to be quoted.\n        See `YAML IDIOSYNCRASIES`_ for more details.\n\n    .. _`YAML IDIOSYNCRASIES`: https://docs.saltproject.io/en/latest/topics/troubleshooting/yaml_idiosyncrasies.html#time-expressions\n\n    Args:\n\n        name (str):\n            The name of the task to which to add the trigger.\n\n        location (str):\n            A string value representing the location of the task. Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n        trigger_type (str):\n            The type of trigger to create. This is defined when the trigger is\n            created and cannot be changed later. Options are as follows:\n\n                - Event\n                - Once\n                - Daily\n                - Weekly\n                - Monthly\n                - MonthlyDay\n                - OnIdle\n                - OnTaskCreation\n                - OnBoot\n                - OnLogon\n                - OnSessionChange\n\n        trigger_enabled (bool):\n            Boolean value that indicates whether the trigger is enabled.\n\n        start_date (str):\n            The date when the trigger is activated. If no value is passed, the\n            current date will be used. Can be one of the following formats:\n\n                - %Y-%m-%d\n                - %m-%d-%y\n                - %m-%d-%Y\n                - %m/%d/%y\n                - %m/%d/%Y\n                - %Y/%m/%d\n\n        start_time (str):\n            The time when the trigger is activated. If no value is passed,\n            midnight will be used. Can be one of the following formats:\n\n                - %I:%M:%S %p\n                - %I:%M %p\n                - %H:%M:%S\n                - %H:%M\n\n        end_date (str):\n            The date when the trigger is deactivated. The trigger cannot start\n            the task after it is deactivated. Can be one of the following\n            formats:\n\n                - %Y-%m-%d\n                - %m-%d-%y\n                - %m-%d-%Y\n                - %m/%d/%y\n                - %m/%d/%Y\n                - %Y/%m/%d\n\n        end_time (str):\n            The time when the trigger is deactivated. If this is not passed\n            with ``end_date`` it will be set to midnight. Can be one of the\n            following formats:\n\n                - %I:%M:%S %p\n                - %I:%M %p\n                - %H:%M:%S\n                - %H:%M\n\n        random_delay (str):\n            The delay time that is randomly added to the start time of the\n            trigger. Valid values are:\n\n                - 30 seconds\n                - 1 minute\n                - 30 minutes\n                - 1 hour\n                - 8 hours\n                - 1 day\n\n            .. note::\n\n                This parameter applies to the following trigger types\n\n                    - Once\n                    - Daily\n                    - Weekly\n                    - Monthly\n                    - MonthlyDay\n\n        repeat_interval (str):\n            The amount of time between each restart of the task. Valid values\n            are:\n\n                - 5 minutes\n                - 10 minutes\n                - 15 minutes\n                - 30 minutes\n                - 1 hour\n\n        repeat_duration (str):\n            How long the pattern is repeated. Valid values are:\n\n                - Indefinitely\n                - 15 minutes\n                - 30 minutes\n                - 1 hour\n                - 12 hours\n                - 1 day\n\n        repeat_stop_at_duration_end (bool):\n            Boolean value that indicates if a running instance of the task is\n            stopped at the end of the repetition pattern duration.\n\n        execution_time_limit (str):\n            The maximum amount of time that the task launched by the trigger is\n            allowed to run. Valid values are:\n\n                - 30 minutes\n                - 1 hour\n                - 2 hours\n                - 4 hours\n                - 8 hours\n                - 12 hours\n                - 1 day\n                - 3 days (default)\n\n        delay (str):\n            The time the trigger waits after its activation to start the task.\n            Valid values are:\n\n                - 15 seconds\n                - 30 seconds\n                - 1 minute\n                - 30 minutes\n                - 1 hour\n                - 8 hours\n                - 1 day\n\n            .. note::\n\n                This parameter applies to the following trigger types:\n\n                    - OnLogon\n                    - OnBoot\n                    - Event\n                    - OnTaskCreation\n                    - OnSessionChange\n\n    **kwargs**\n\n    There are optional keyword arguments determined by the type of trigger\n    being defined. They are as follows:\n\n    *Event*\n\n        The trigger will be fired by an event.\n\n            subscription (str):\n                An event definition in xml format that fires the trigger. The\n                easiest way to get this would is to create an event in Windows\n                Task Scheduler and then copy the xml text.\n\n    *Once*\n\n        No special parameters required.\n\n    *Daily*\n\n        The task will run daily.\n\n            days_interval (int):\n                The interval between days in the schedule. An interval of 1\n                produces a daily schedule. An interval of 2 produces an\n                every-other day schedule. If no interval is specified, 1 is\n                used. Valid entries are 1 - 999.\n\n    *Weekly*\n\n        The task will run weekly.\n\n            weeks_interval (int):\n                The interval between weeks in the schedule. An interval of 1\n                produces a weekly schedule. An interval of 2 produces an\n                every-other week schedule. If no interval is specified, 1 is\n                used. Valid entries are 1 - 52.\n\n            days_of_week (list):\n                Sets the days of the week on which the task runs. Should be a\n                list. ie: ``[\'Monday\',\'Wednesday\',\'Friday\']``. Valid entries are\n                the names of the days of the week.\n\n    *Monthly*\n\n        The task will run monthly.\n\n            months_of_year (list):\n                Sets the months of the year during which the task runs. Should\n                be a list. ie: ``[\'January\',\'July\']``. Valid entries are the\n                full names of all the months.\n\n            days_of_month (list):\n                Sets the days of the month during which the task runs. Should be\n                a list. ie: ``[1, 15, \'Last\']``. Options are all days of the\n                month 1 - 31 and the word \'Last\' to indicate the last day of the\n                month.\n\n            last_day_of_month (bool):\n                Boolean value that indicates that the task runs on the last day\n                of the month regardless of the actual date of that day.\n\n                .. note::\n\n                    You can set the task to run on the last day of the month by\n                    either including the word \'Last\' in the list of days, or\n                    setting the parameter \'last_day_of_month\' equal to ``True``.\n\n    *MonthlyDay*\n\n        The task will run monthly on the specified day.\n\n            months_of_year (list):\n                Sets the months of the year during which the task runs. Should\n                be a list. ie: ``[\'January\',\'July\']``. Valid entries are the\n                full names of all the months.\n\n            weeks_of_month (list):\n                Sets the weeks of the month during which the task runs. Should\n                be a list. ie: ``[\'First\',\'Third\']``. Valid options are:\n\n                    - First\n                    - Second\n                    - Third\n                    - Fourth\n\n            last_week_of_month (bool):\n                Boolean value that indicates that the task runs on the last week\n                of the month.\n\n            days_of_week (list):\n                Sets the days of the week during which the task runs. Should be\n                a list. ie: ``[\'Monday\',\'Wednesday\',\'Friday\']``.  Valid entries\n                are the names of the days of the week.\n\n    *OnIdle*\n\n        No special parameters required.\n\n    *OnTaskCreation*\n\n        No special parameters required.\n\n    *OnBoot*\n\n        No special parameters required.\n\n    *OnLogon*\n\n        No special parameters required.\n\n    *OnSessionChange*\n\n        The task will be triggered by a session change.\n\n            session_user_name (str):\n                Sets the user for the Terminal Server session. When a session\n                state change is detected for this user, a task is started. To\n                detect session status change for any user, do not pass this\n                parameter.\n\n            state_change (str):\n                Sets the kind of Terminal Server session change that would\n                trigger a task launch. Valid options are:\n\n                    - ConsoleConnect: When you connect to a user session (switch\n                      users)\n                    - ConsoleDisconnect: When you disconnect a user session\n                      (switch users)\n                    - RemoteConnect: When a user connects via Remote Desktop\n                    - RemoteDisconnect: When a user disconnects via Remote\n                      Desktop\n                    - SessionLock: When the workstation is locked\n                    - SessionUnlock: When the workstation is unlocked\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'minion-id\' task.add_trigger <task_name> trigger_type=Once trigger_enabled=True start_date=2016/12/1 start_time=\'"12:01"\'\n    '
    if not trigger_type:
        return 'Required parameter "trigger_type" not specified'
    state_changes = {'ConsoleConnect': 1, 'ConsoleDisconnect': 2, 'RemoteConnect': 3, 'RemoteDisconnect': 4, 'SessionLock': 7, 'SessionUnlock': 8}
    days = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32, 7: 64, 8: 128, 9: 256, 10: 512, 11: 1024, 12: 2048, 13: 4096, 14: 8192, 15: 16384, 16: 32768, 17: 65536, 18: 131072, 19: 262144, 20: 524288, 21: 1048576, 22: 2097152, 23: 4194304, 24: 8388608, 25: 16777216, 26: 33554432, 27: 67108864, 28: 134217728, 29: 268435456, 30: 536870912, 31: 1073741824, 'Last': 2147483648}
    weekdays = {'Sunday': 1, 'Monday': 2, 'Tuesday': 4, 'Wednesday': 8, 'Thursday': 16, 'Friday': 32, 'Saturday': 64}
    weeks = {'First': 1, 'Second': 2, 'Third': 4, 'Fourth': 8}
    months = {'January': 1, 'February': 2, 'March': 4, 'April': 8, 'May': 16, 'June': 32, 'July': 64, 'August': 128, 'September': 256, 'October': 512, 'November': 1024, 'December': 2048}
    if start_date:
        date_format = _get_date_time_format(start_date)
        if date_format:
            dt_obj = datetime.strptime(start_date, date_format)
        else:
            return 'Invalid start_date'
    else:
        dt_obj = datetime.now()
    if start_time:
        time_format = _get_date_time_format(start_time)
        if time_format:
            tm_obj = datetime.strptime(start_time, time_format)
        else:
            return 'Invalid start_time'
    else:
        tm_obj = datetime.strptime('00:00:00', '%H:%M:%S')
    start_boundary = '{}T{}'.format(dt_obj.strftime('%Y-%m-%d'), tm_obj.strftime('%H:%M:%S'))
    dt_obj = None
    if end_date:
        date_format = _get_date_time_format(end_date)
        if date_format:
            dt_obj = datetime.strptime(end_date, date_format)
        else:
            return 'Invalid end_date'
    if end_time:
        time_format = _get_date_time_format(end_time)
        if time_format:
            tm_obj = datetime.strptime(end_time, time_format)
        else:
            return 'Invalid end_time'
    else:
        tm_obj = datetime.strptime('00:00:00', '%H:%M:%S')
    end_boundary = None
    if dt_obj and tm_obj:
        end_boundary = '{}T{}'.format(dt_obj.strftime('%Y-%m-%d'), tm_obj.strftime('%H:%M:%S'))
    with salt.utils.winapi.Com():
        save_definition = False
        if kwargs.get('task_definition', False):
            task_definition = kwargs.get('task_definition')
        else:
            save_definition = True
            if not name:
                return 'Required parameter "name" not passed'
            if name in list_tasks(location):
                task_service = win32com.client.Dispatch('Schedule.Service')
                task_service.Connect()
                task_folder = task_service.GetFolder(location)
                task_definition = task_folder.GetTask(name).Definition
            else:
                return '{} not found'.format(name)
        trigger = task_definition.Triggers.Create(trigger_types[trigger_type])
        trigger.StartBoundary = start_boundary
        if delay:
            trigger.Delay = _lookup_first(duration, delay)
        if random_delay:
            trigger.RandomDelay = _lookup_first(duration, random_delay)
        if repeat_interval:
            trigger.Repetition.Interval = _lookup_first(duration, repeat_interval)
            if repeat_duration:
                trigger.Repetition.Duration = _lookup_first(duration, repeat_duration)
            trigger.Repetition.StopAtDurationEnd = repeat_stop_at_duration_end
        if execution_time_limit:
            trigger.ExecutionTimeLimit = _lookup_first(duration, execution_time_limit)
        if end_boundary:
            trigger.EndBoundary = end_boundary
        trigger.Enabled = trigger_enabled
        if trigger_types[trigger_type] == TASK_TRIGGER_EVENT:
            if kwargs.get('subscription', False):
                trigger.Id = 'Event_ID1'
                trigger.Subscription = kwargs.get('subscription')
            else:
                return 'Required parameter "subscription" not passed'
        elif trigger_types[trigger_type] == TASK_TRIGGER_TIME:
            trigger.Id = 'Once_ID1'
        elif trigger_types[trigger_type] == TASK_TRIGGER_DAILY:
            trigger.Id = 'Daily_ID1'
            trigger.DaysInterval = kwargs.get('days_interval', 1)
        elif trigger_types[trigger_type] == TASK_TRIGGER_WEEKLY:
            trigger.Id = 'Weekly_ID1'
            trigger.WeeksInterval = kwargs.get('weeks_interval', 1)
            if kwargs.get('days_of_week', False):
                bits_days = 0
                for weekday in kwargs.get('days_of_week'):
                    bits_days |= weekdays[weekday]
                trigger.DaysOfWeek = bits_days
            else:
                return 'Required parameter "days_of_week" not passed'
        elif trigger_types[trigger_type] == TASK_TRIGGER_MONTHLY:
            trigger.Id = 'Monthly_ID1'
            if kwargs.get('months_of_year', False):
                bits_months = 0
                for month in kwargs.get('months_of_year'):
                    bits_months |= months[month]
                trigger.MonthsOfYear = bits_months
            else:
                return 'Required parameter "months_of_year" not passed'
            if kwargs.get('days_of_month', False) or kwargs.get('last_day_of_month', False):
                if kwargs.get('days_of_month', False):
                    bits_days = 0
                    for day in kwargs.get('days_of_month'):
                        bits_days |= days[day]
                    trigger.DaysOfMonth = bits_days
                trigger.RunOnLastDayOfMonth = kwargs.get('last_day_of_month', False)
            else:
                return 'Monthly trigger requires "days_of_month" or "last_day_of_month" parameters'
        elif trigger_types[trigger_type] == TASK_TRIGGER_MONTHLYDOW:
            trigger.Id = 'Monthly_DOW_ID1'
            if kwargs.get('months_of_year', False):
                bits_months = 0
                for month in kwargs.get('months_of_year'):
                    bits_months |= months[month]
                trigger.MonthsOfYear = bits_months
            else:
                return 'Required parameter "months_of_year" not passed'
            if kwargs.get('weeks_of_month', False) or kwargs.get('last_week_of_month', False):
                if kwargs.get('weeks_of_month', False):
                    bits_weeks = 0
                    for week in kwargs.get('weeks_of_month'):
                        bits_weeks |= weeks[week]
                    trigger.WeeksOfMonth = bits_weeks
                trigger.RunOnLastWeekOfMonth = kwargs.get('last_week_of_month', False)
            else:
                return 'Monthly DOW trigger requires "weeks_of_month" or "last_week_of_month" parameters'
            if kwargs.get('days_of_week', False):
                bits_days = 0
                for weekday in kwargs.get('days_of_week'):
                    bits_days |= weekdays[weekday]
                trigger.DaysOfWeek = bits_days
            else:
                return 'Required parameter "days_of_week" not passed'
        elif trigger_types[trigger_type] == TASK_TRIGGER_IDLE:
            trigger.Id = 'OnIdle_ID1'
        elif trigger_types[trigger_type] == TASK_TRIGGER_REGISTRATION:
            trigger.Id = 'OnTaskCreation_ID1'
        elif trigger_types[trigger_type] == TASK_TRIGGER_BOOT:
            trigger.Id = 'OnBoot_ID1'
        elif trigger_types[trigger_type] == TASK_TRIGGER_LOGON:
            trigger.Id = 'OnLogon_ID1'
        elif trigger_types[trigger_type] == TASK_TRIGGER_SESSION_STATE_CHANGE:
            trigger.Id = 'OnSessionStateChange_ID1'
            if kwargs.get('session_user_name', False):
                trigger.UserId = kwargs.get('session_user_name')
            if kwargs.get('state_change', False):
                trigger.StateChange = state_changes[kwargs.get('state_change')]
            else:
                return 'Required parameter "state_change" not passed'
        if save_definition:
            return _save_task_definition(name=name, task_folder=task_folder, task_definition=task_definition, user_name=task_definition.Principal.UserID, password=None, logon_type=task_definition.Principal.LogonType)

def clear_triggers(name, location='\\'):
    if False:
        return 10
    "\n    Remove all triggers from the task.\n\n    Args:\n\n        name (str):\n            The name of the task from which to clear all triggers.\n\n        location (str):\n            A string value representing the location of the task. Default is\n            ``\\`` which is the root for the task scheduler\n            (``C:\\Windows\\System32\\tasks``).\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion-id' task.clear_trigger <task_name>\n    "
    if name not in list_tasks(location):
        return '{} not found in {}'.format(name, location)
    with salt.utils.winapi.Com():
        task_service = win32com.client.Dispatch('Schedule.Service')
        task_service.Connect()
        task_folder = task_service.GetFolder(location)
        task_definition = task_folder.GetTask(name).Definition
        triggers = task_definition.Triggers
        triggers.Clear()
        return _save_task_definition(name=name, task_folder=task_folder, task_definition=task_definition, user_name=task_definition.Principal.UserID, password=None, logon_type=task_definition.Principal.LogonType)