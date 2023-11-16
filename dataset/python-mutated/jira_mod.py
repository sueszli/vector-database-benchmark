"""
JIRA Execution module
=====================

.. versionadded:: 2019.2.0

Execution module to manipulate JIRA tickets via Salt.

This module requires the ``jira`` Python library to be installed.

Configuration example:

.. code-block:: yaml

  jira:
    server: https://jira.atlassian.org
    username: salt
    password: pass
"""
import logging
import salt.utils.args
try:
    import jira
    HAS_JIRA = True
except ImportError:
    HAS_JIRA = False
log = logging.getLogger(__name__)
__virtualname__ = 'jira'
__proxyenabled__ = ['*']
JIRA = None

def __virtual__():
    if False:
        print('Hello World!')
    return __virtualname__ if HAS_JIRA else (False, 'Please install the jira Python library from PyPI')

def _get_credentials(server=None, username=None, password=None):
    if False:
        while True:
            i = 10
    '\n    Returns the credentials merged with the config data (opts + pillar).\n    '
    jira_cfg = __salt__['config.merge']('jira', default={})
    if not server:
        server = jira_cfg.get('server')
    if not username:
        username = jira_cfg.get('username')
    if not password:
        password = jira_cfg.get('password')
    return (server, username, password)

def _get_jira(server=None, username=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    global JIRA
    if not JIRA:
        (server, username, password) = _get_credentials(server=server, username=username, password=password)
        JIRA = jira.JIRA(basic_auth=(username, password), server=server, logging=True)
    return JIRA

def create_issue(project, summary, description, template_engine='jinja', context=None, defaults=None, saltenv='base', issuetype='Bug', priority='Normal', labels=None, assignee=None, server=None, username=None, password=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a JIRA issue using the named settings. Return the JIRA ticket ID.\n\n    project\n        The name of the project to attach the JIRA ticket to.\n\n    summary\n        The summary (title) of the JIRA ticket. When the ``template_engine``\n        argument is set to a proper value of an existing Salt template engine\n        (e.g., ``jinja``, ``mako``, etc.) it will render the ``summary`` before\n        creating the ticket.\n\n    description\n        The full body description of the JIRA ticket. When the ``template_engine``\n        argument is set to a proper value of an existing Salt template engine\n        (e.g., ``jinja``, ``mako``, etc.) it will render the ``description`` before\n        creating the ticket.\n\n    template_engine: ``jinja``\n        The name of the template engine to be used to render the values of the\n        ``summary`` and ``description`` arguments. Default: ``jinja``.\n\n    context: ``None``\n        The context to pass when rendering the ``summary`` and ``description``.\n        This argument is ignored when ``template_engine`` is set as ``None``\n\n    defaults: ``None``\n        Default values to pass to the Salt rendering pipeline for the\n        ``summary`` and ``description`` arguments.\n        This argument is ignored when ``template_engine`` is set as ``None``.\n\n    saltenv: ``base``\n        The Salt environment name (for the rendering system).\n\n    issuetype: ``Bug``\n        The type of the JIRA ticket. Default: ``Bug``.\n\n    priority: ``Normal``\n        The priority of the JIRA ticket. Default: ``Normal``.\n\n    labels: ``None``\n        A list of labels to add to the ticket.\n\n    assignee: ``None``\n        The name of the person to assign the ticket to.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' jira.create_issue NET 'Ticket title' 'Ticket description'\n        salt '*' jira.create_issue NET 'Issue on {{ opts.id }}' 'Error detected on {{ opts.id }}' template_engine=jinja\n    "
    if template_engine:
        summary = __salt__['file.apply_template_on_contents'](summary, template=template_engine, context=context, defaults=defaults, saltenv=saltenv)
        description = __salt__['file.apply_template_on_contents'](description, template=template_engine, context=context, defaults=defaults, saltenv=saltenv)
    jira_ = _get_jira(server=server, username=username, password=password)
    if not labels:
        labels = []
    data = {'project': {'key': project}, 'summary': summary, 'description': description, 'issuetype': {'name': issuetype}, 'priority': {'name': priority}, 'labels': labels}
    data.update(salt.utils.args.clean_kwargs(**kwargs))
    issue = jira_.create_issue(data)
    issue_key = str(issue)
    if assignee:
        assign_issue(issue_key, assignee)
    return issue_key

def assign_issue(issue_key, assignee, server=None, username=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Assign the issue to an existing user. Return ``True`` when the issue has\n    been properly assigned.\n\n    issue_key\n        The JIRA ID of the ticket to manipulate.\n\n    assignee\n        The name of the user to assign the ticket to.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jira.assign_issue NET-123 example_user\n    "
    jira_ = _get_jira(server=server, username=username, password=password)
    assigned = jira_.assign_issue(issue_key, assignee)
    return assigned

def add_comment(issue_key, comment, visibility=None, is_internal=False, server=None, username=None, password=None):
    if False:
        print('Hello World!')
    "\n    Add a comment to an existing ticket. Return ``True`` when it successfully\n    added the comment.\n\n    issue_key\n        The issue ID to add the comment to.\n\n    comment\n        The body of the comment to be added.\n\n    visibility: ``None``\n        A dictionary having two keys:\n\n        - ``type``: is ``role`` (or ``group`` if the JIRA server has configured\n          comment visibility for groups).\n        - ``value``: the name of the role (or group) to which viewing of this\n          comment will be restricted.\n\n    is_internal: ``False``\n        Whether a comment has to be marked as ``Internal`` in Jira Service Desk.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jira.add_comment NE-123 'This is a comment'\n    "
    jira_ = _get_jira(server=server, username=username, password=password)
    comm = jira_.add_comment(issue_key, comment, visibility=visibility, is_internal=is_internal)
    return True

def issue_closed(issue_key, server=None, username=None, password=None):
    if False:
        print('Hello World!')
    "\n    Check if the issue is closed.\n\n    issue_key\n        The JIRA iD of the ticket to close.\n\n    Returns:\n\n    - ``True``: the ticket exists and it is closed.\n    - ``False``: the ticket exists and it has not been closed.\n    - ``None``: the ticket does not exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' jira.issue_closed NE-123\n    "
    if not issue_key:
        return None
    jira_ = _get_jira(server=server, username=username, password=password)
    try:
        ticket = jira_.issue(issue_key)
    except jira.exceptions.JIRAError:
        return None
    return ticket.fields().status.name == 'Closed'