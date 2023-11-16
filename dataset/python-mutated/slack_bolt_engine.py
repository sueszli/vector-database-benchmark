"""
An engine that reads messages from Slack and can act on them

.. versionadded:: 3006.0

:depends: `slack_bolt <https://pypi.org/project/slack_bolt/>`_ Python module

.. important::
    This engine requires a Slack app and a Slack Bot user. To create a
    bot user, first go to the **Custom Integrations** page in your
    Slack Workspace. Copy and paste the following URL, and log in with
    account credentials with administrative privileges:

    ``https://api.slack.com/apps/new``

    Next, click on the ``From scratch`` option from the ``Create an app`` popup.
    Give your new app a unique name, eg. ``SaltSlackEngine``, select the workspace
    where your app will be running, and click ``Create App``.

    Next, click on ``Socket Mode`` and then click on the toggle button for
    ``Enable Socket Mode``. In the dialog give your Socket Mode Token a unique
    name and then copy and save the app level token.  This will be used
    as the ``app_token`` parameter in the Slack engine configuration.

    Next, click on ``Event Subscriptions`` and ensure that ``Enable Events`` is in
    the on position.  Then  add the following bot events, ``message.channel``
    and ``message.im`` to the ``Subcribe to bot events`` list.

    Next, click on ``OAuth & Permissions`` and then under ``Bot Token Scope``, click
    on ``Add an OAuth Scope``.  Ensure the following scopes are included:

        - ``channels:history``
        - ``channels:read``
        - ``chat:write``
        - ``commands``
        - ``files:read``
        - ``files:write``
        - ``im:history``
        - ``mpim:history``
        - ``usergroups:read``
        - ``users:read``

    Once all the scopes have been added, click the ``Install to Workspace`` button
    under ``OAuth Tokens for Your Workspace``, then click ``Allow``.  Copy and save
    the ``Bot User OAuth Token``, this will be used as the ``bot_token`` parameter
    in the Slack engine configuration.

    Finally, add this bot user to a channel by switching to the channel and
    using ``/invite @mybotuser``. Keep in mind that this engine will process
    messages from each channel in which the bot is a member, so it is
    recommended to narrowly define the commands which can be executed, and the
    Slack users which are allowed to run commands.


This engine has two boolean configuration parameters that toggle specific
features (both default to ``False``):

1. ``control`` - If set to ``True``, then any message which starts with the
   trigger string (which defaults to ``!`` and can be overridden by setting the
   ``trigger`` option in the engine configuration) will be interpreted as a
   Salt CLI command and the engine will attempt to run it. The permissions
   defined in the various ``groups`` will determine if the Slack user is
   allowed to run the command. The ``targets`` and ``default_target`` options
   can be used to set targets for a given command, but the engine can also read
   the following two keyword arguments:

   - ``target`` - The target expression to use for the command

   - ``tgt_type`` - The match type, can be one of ``glob``, ``list``,
     ``pcre``, ``grain``, ``grain_pcre``, ``pillar``, ``nodegroup``, ``range``,
     ``ipcidr``, or ``compound``. The default value is ``glob``.

   Here are a few examples:

   .. code-block:: text

       !test.ping target=*
       !state.apply foo target=os:CentOS tgt_type=grain
       !pkg.version mypkg target=role:database tgt_type=pillar

2. ``fire_all`` - If set to ``True``, all messages which are not prefixed with
   the trigger string will fired as events onto Salt's ref:`event bus
   <event-system>`. The tag for these events will be prefixed with the string
   specified by the ``tag`` config option (default: ``salt/engines/slack``).


The ``groups_pillar_name`` config option can be used to pull group
configuration from the specified pillar key.

.. note::
    In order to use ``groups_pillar_name``, the engine must be running as a
    minion running on the master, so that the ``Caller`` client can be used to
    retrieve that minion's pillar data, because the master process does not have
    pillar data.


Configuration Examples
======================

.. versionchanged:: 2017.7.0
    Access control group support added

.. versionchanged:: 3006.0
    Updated to use slack_bolt Python library.

This example uses a single group called ``default``. In addition, other groups
are being loaded from pillar data. The users and commands defined within these
groups are used to determine whether the Slack user has permission to run
the desired command.

.. code-block:: text

    engines:
      - slack_bolt:
          app_token: "xapp-x-xxxxxxxxxxx-xxxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          bot_token: 'xoxb-xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx'
          control: True
          fire_all: False
          groups_pillar_name: 'slack_engine:groups_pillar'
          groups:
            default:
              users:
                - '*'
              commands:
                - test.ping
                - cmd.run
                - list_jobs
                - list_commands
              aliases:
                list_jobs:
                  cmd: jobs.list_jobs
                list_commands:
                  cmd: 'pillar.get salt:engines:slack:valid_commands target=saltmaster tgt_type=list'
              default_target:
                target: saltmaster
                tgt_type: glob
              targets:
                test.ping:
                  target: '*'
                  tgt_type: glob
                cmd.run:
                  target: saltmaster
                  tgt_type: list

This example shows multiple groups applying to different users, with all users
having access to run test.ping. Keep in mind that when using ``*``, the value
must be quoted, or else PyYAML will fail to load the configuration.

.. code-block:: text

    engines:
      - slack_bolt:
          groups_pillar: slack_engine_pillar
          app_token: "xapp-x-xxxxxxxxxxx-xxxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          bot_token: 'xoxb-xxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx'
          control: True
          fire_all: True
          tag: salt/engines/slack
          groups_pillar_name: 'slack_engine:groups_pillar'
          groups:
            default:
              users:
                - '*'
              commands:
                - test.ping
              aliases:
                list_jobs:
                  cmd: jobs.list_jobs
                list_commands:
                  cmd: 'pillar.get salt:engines:slack:valid_commands target=saltmaster tgt_type=list'
            gods:
              users:
                - garethgreenaway
              commands:
                - '*'

"""
import ast
import collections
import datetime
import itertools
import logging
import re
import time
import traceback
import salt.client
import salt.loader
import salt.minion
import salt.output
import salt.runner
import salt.utils.args
import salt.utils.event
import salt.utils.http
import salt.utils.json
import salt.utils.slack
import salt.utils.yaml
try:
    import slack_bolt
    import slack_bolt.adapter.socket_mode
    HAS_SLACKBOLT = True
except ImportError:
    HAS_SLACKBOLT = False
log = logging.getLogger(__name__)
__virtualname__ = 'slack_bolt'

def __virtual__():
    if False:
        print('Hello World!')
    if not HAS_SLACKBOLT:
        return (False, "The 'slack_bolt' Python module could not be loaded")
    return __virtualname__

class SlackClient:

    def __init__(self, app_token, bot_token, trigger_string):
        if False:
            i = 10
            return i + 15
        self.master_minion = salt.minion.MasterMinion(__opts__)
        self.app = slack_bolt.App(token=bot_token)
        self.handler = slack_bolt.adapter.socket_mode.SocketModeHandler(self.app, app_token)
        self.handler.connect()
        self.app_token = app_token
        self.bot_token = bot_token
        self.msg_queue = collections.deque()
        trigger_pattern = f'(^{trigger_string}.*)'
        self.app.message(re.compile(trigger_pattern))(self.message_trigger)

    def _run_until(self):
        if False:
            print('Hello World!')
        return True

    def message_trigger(self, message):
        if False:
            return 10
        self.msg_queue.append(message)

    def get_slack_users(self, token):
        if False:
            return 10
        '\n        Get all users from Slack\n\n        :type user: str\n        :param token: The Slack token being used to allow Salt to interact with Slack.\n        '
        ret = salt.utils.slack.query(function='users', api_key=token, opts=__opts__)
        users = {}
        if 'message' in ret:
            for item in ret['message']:
                if 'is_bot' in item:
                    if not item['is_bot']:
                        users[item['name']] = item['id']
                        users[item['id']] = item['name']
        return users

    def get_slack_channels(self, token):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get all channel names from Slack\n\n        :type token: str\n        :param token: The Slack token being used to allow Salt to interact with Slack.\n        '
        ret = salt.utils.slack.query(function='rooms', api_key=token, opts={'exclude_archived': True, 'exclude_members': True})
        channels = {}
        if 'message' in ret:
            for item in ret['message']:
                channels[item['id']] = item['name']
        return channels

    def get_config_groups(self, groups_conf, groups_pillar_name):
        if False:
            i = 10
            return i + 15
        '\n        get info from groups in config, and from the named pillar\n\n        :type group_conf: dict\n        :param group_conf:\n            The dictionary containing the groups, group members,\n            and the commands those group members have access to.\n\n        :type groups_pillar_name: str\n        :param groups_pillar_name:\n            can be used to pull group configuration from the specified pillar key.\n        '
        ret_groups = {'default': {'users': set(), 'commands': set(), 'aliases': {}, 'default_target': {}, 'targets': {}}}
        if not groups_conf:
            use_groups = {}
        else:
            use_groups = groups_conf
        log.debug('use_groups %s', use_groups)
        try:
            groups_gen = itertools.chain(self._groups_from_pillar(groups_pillar_name).items(), use_groups.items())
        except AttributeError:
            log.warning('Failed to get groups from %s: %s or from config: %s', groups_pillar_name, self._groups_from_pillar(groups_pillar_name), use_groups)
            groups_gen = []
        for (name, config) in groups_gen:
            log.info('Trying to get %s and %s to be useful', name, config)
            ret_groups.setdefault(name, {'users': set(), 'commands': set(), 'aliases': {}, 'default_target': {}, 'targets': {}})
            try:
                ret_groups[name]['users'].update(set(config.get('users', [])))
                ret_groups[name]['commands'].update(set(config.get('commands', [])))
                ret_groups[name]['aliases'].update(config.get('aliases', {}))
                ret_groups[name]['default_target'].update(config.get('default_target', {}))
                ret_groups[name]['targets'].update(config.get('targets', {}))
            except (IndexError, AttributeError):
                log.warning("Couldn't use group %s. Check that targets is a dictionary and not a list", name)
        log.debug('Got the groups: %s', ret_groups)
        return ret_groups

    def _groups_from_pillar(self, pillar_name):
        if False:
            print('Hello World!')
        '\n\n        :type pillar_name: str\n        :param pillar_name: The pillar.get syntax for the pillar to be queried.\n\n        returns a dictionary (unless the pillar is mis-formatted)\n        '
        if pillar_name and __opts__['__role'] == 'minion':
            pillar_groups = __salt__['pillar.get'](pillar_name, {})
            log.debug('Got pillar groups %s from pillar %s', pillar_groups, pillar_name)
            log.debug('pillar groups is %s', pillar_groups)
            log.debug('pillar groups type is %s', type(pillar_groups))
        else:
            pillar_groups = {}
        return pillar_groups

    def fire(self, tag, msg):
        if False:
            return 10
        "\n        This replaces a function in main called 'fire'\n\n        It fires an event into the salt bus.\n\n        :type tag: str\n        :param tag: The tag to use when sending events to the Salt event bus.\n\n        :type msg: dict\n        :param msg: The msg dictionary to send to the Salt event bus.\n\n        "
        if __opts__.get('__role') == 'master':
            fire_master = salt.utils.event.get_master_event(__opts__, __opts__['sock_dir']).fire_master
        else:
            fire_master = None
        if fire_master:
            fire_master(msg, tag)
        else:
            __salt__['event.send'](tag, msg)

    def can_user_run(self, user, command, groups):
        if False:
            i = 10
            return i + 15
        "\n        Check whether a user is in any group, including whether a group has the '*' membership\n\n        :type user: str\n        :param user: The username being checked against\n\n        :type command: str\n        :param command: The command that is being invoked (e.g. test.ping)\n\n        :type groups: dict\n        :param groups: the dictionary with groups permissions structure.\n\n        :rtype: tuple\n        :returns: On a successful permitting match, returns 2-element tuple that contains\n            the name of the group that successfully matched, and a dictionary containing\n            the configuration of the group so it can be referenced.\n\n            On failure it returns an empty tuple\n\n        "
        log.info('%s wants to run %s with groups %s', user, command, groups)
        for (key, val) in groups.items():
            if user not in val['users']:
                if '*' not in val['users']:
                    continue
            if command not in val['commands'] and command not in val.get('aliases', {}).keys():
                if '*' not in val['commands']:
                    continue
            log.info('Slack user %s permitted to run %s', user, command)
            return (key, val)
        log.info('Slack user %s denied trying to run %s', user, command)
        return ()

    def commandline_to_list(self, cmdline_str, trigger_string):
        if False:
            print('Hello World!')
        '\n        cmdline_str is the string of the command line\n        trigger_string is the trigger string, to be removed\n        '
        cmdline = salt.utils.args.shlex_split(cmdline_str[len(trigger_string):])
        cmdlist = []
        for cmditem in cmdline:
            pattern = '(?P<begin>.*)(<.*\\|)(?P<url>.*)(>)(?P<remainder>.*)'
            mtch = re.match(pattern, cmditem)
            if mtch:
                origtext = mtch.group('begin') + mtch.group('url') + mtch.group('remainder')
                cmdlist.append(origtext)
            else:
                cmdlist.append(cmditem)
        return cmdlist

    def control_message_target(self, slack_user_name, text, loaded_groups, trigger_string):
        if False:
            while True:
                i = 10
        "Returns a tuple of (target, cmdline,) for the response\n\n        Raises IndexError if a user can't be looked up from all_slack_users\n\n        Returns (False, False) if the user doesn't have permission\n\n        These are returned together because the commandline and the targeting\n        interact with the group config (specifically aliases and targeting configuration)\n        so taking care of them together works out.\n\n        The cmdline that is returned is the actual list that should be\n        processed by salt, and not the alias.\n\n        "
        cmdline = self.commandline_to_list(text, trigger_string)
        permitted_group = self.can_user_run(slack_user_name, cmdline[0], loaded_groups)
        log.debug('slack_user_name is %s and the permitted group is %s', slack_user_name, permitted_group)
        if not permitted_group:
            return (False, None, cmdline[0])
        if not slack_user_name:
            return (False, None, cmdline[0])
        if cmdline[0] in permitted_group[1].get('aliases', {}).keys():
            use_cmdline = self.commandline_to_list(permitted_group[1]['aliases'][cmdline[0]].get('cmd', ''), '')
            use_cmdline.extend(cmdline[1:])
        else:
            use_cmdline = cmdline
        target = self.get_target(permitted_group, cmdline, use_cmdline)
        use_cmdline = [item for item in use_cmdline if all((not item.startswith(x) for x in ('target', 'tgt_type')))]
        return (True, target, use_cmdline)

    def message_text(self, m_data):
        if False:
            while True:
                i = 10
        "\n        Raises ValueError if a value doesn't work out, and TypeError if\n        this isn't a message type\n\n        :type m_data: dict\n        :param m_data: The message sent from Slack\n\n        "
        if m_data.get('type') != 'message':
            raise TypeError('This is not a message')
        _text = m_data.get('text', None) or m_data.get('message', {}).get('text', None)
        try:
            log.info('Message is %s', _text)
        except UnicodeEncodeError as uee:
            log.warning('Got a message that I could not log. The reason is: %s', uee)
        _text = salt.utils.json.dumps(_text)
        _text = salt.utils.yaml.safe_load(_text)
        if not _text:
            raise ValueError('_text has no value')
        return _text

    def generate_triggered_messages(self, token, trigger_string, groups, groups_pillar_name):
        if False:
            print('Hello World!')
        "\n        slack_token = string\n        trigger_string = string\n        input_valid_users = set\n        input_valid_commands = set\n\n        When the trigger_string prefixes the message text, yields a dictionary\n        of::\n\n            {\n                'message_data': m_data,\n                'cmdline': cmdline_list, # this is a list\n                'channel': channel,\n                'user': m_data['user'],\n                'slack_client': sc\n            }\n\n        else yields {'message_data': m_data} and the caller can handle that\n\n        When encountering an error (e.g. invalid message), yields {}, the caller can proceed to the next message\n\n        When the websocket being read from has given up all its messages, yields {'done': True} to\n        indicate that the caller has read all of the relevant data for now, and should continue\n        its own processing and check back for more data later.\n\n        This relies on the caller sleeping between checks, otherwise this could flood\n        "
        all_slack_users = self.get_slack_users(token)
        all_slack_channels = self.get_slack_channels(token)

        def just_data(m_data):
            if False:
                i = 10
                return i + 15
            'Always try to return the user and channel anyway'
            if 'user' not in m_data:
                if 'message' in m_data and 'user' in m_data['message']:
                    log.debug('Message was edited, so we look for user in the original message.')
                    user_id = m_data['message']['user']
                elif 'comment' in m_data and 'user' in m_data['comment']:
                    log.debug('Comment was added, so we look for user in the comment.')
                    user_id = m_data['comment']['user']
            else:
                user_id = m_data.get('user')
            channel_id = m_data.get('channel')
            if channel_id.startswith('D'):
                channel_name = 'private chat'
            else:
                channel_name = all_slack_channels.get(channel_id)
            data = {'message_data': m_data, 'user_id': user_id, 'user_name': all_slack_users.get(user_id), 'channel_name': channel_name}
            if not data['user_name']:
                all_slack_users.clear()
                all_slack_users.update(self.get_slack_users(token))
                data['user_name'] = all_slack_users.get(user_id)
            if not data['channel_name']:
                all_slack_channels.clear()
                all_slack_channels.update(self.get_slack_channels(token))
                data['channel_name'] = all_slack_channels.get(channel_id)
            return data
        for sleeps in (5, 10, 30, 60):
            if self.handler:
                break
            else:
                log.warning('Slack connection is invalid, sleeping %s', sleeps)
                time.sleep(sleeps)
        else:
            raise UserWarning('Connection to slack is still invalid, giving up: {}'.format(self.handler))
        while self._run_until():
            while self.msg_queue:
                msg = self.msg_queue.popleft()
                try:
                    msg_text = self.message_text(msg)
                except (ValueError, TypeError) as msg_err:
                    log.debug('Got an error trying to get the message text %s', msg_err)
                    yield {'message_data': msg}
                    continue
                channel = msg['channel']
                data = just_data(msg)
                if msg_text.startswith(trigger_string):
                    loaded_groups = self.get_config_groups(groups, groups_pillar_name)
                    if not data.get('user_name'):
                        log.error('The user %s can not be looked up via slack. What has happened here?', msg.get('user'))
                        channel.send_message('The user {} can not be looked up via slack.  Not running {}'.format(data['user_id'], msg_text))
                        yield {'message_data': msg}
                        continue
                    (allowed, target, cmdline) = self.control_message_target(data['user_name'], msg_text, loaded_groups, trigger_string)
                    if allowed:
                        ret = {'message_data': msg, 'channel': msg['channel'], 'user': data['user_id'], 'user_name': data['user_name'], 'cmdline': cmdline, 'target': target}
                        yield ret
                        continue
                    else:
                        channel.send_message('{} is not allowed to use command {}.'.format(data['user_name'], cmdline))
                        yield data
                        continue
                else:
                    yield data
                    continue
            yield {'done': True}

    def get_target(self, permitted_group, cmdline, alias_cmdline):
        if False:
            print('Hello World!')
        "\n        When we are permitted to run a command on a target, look to see\n        what the default targeting is for that group, and for that specific\n        command (if provided).\n\n        It's possible for ``None`` or ``False`` to be the result of either, which means\n        that it's expected that the caller provide a specific target.\n\n        If no configured target is provided, the command line will be parsed\n        for target=foo and tgt_type=bar\n\n        Test for this::\n\n            h = {'aliases': {}, 'commands': {'cmd.run', 'pillar.get'},\n                'default_target': {'target': '*', 'tgt_type': 'glob'},\n                'targets': {'pillar.get': {'target': 'you_momma', 'tgt_type': 'list'}},\n                'users': {'dmangot', 'jmickle', 'pcn'}}\n            f = {'aliases': {}, 'commands': {'cmd.run', 'pillar.get'},\n                 'default_target': {}, 'targets': {},'users': {'dmangot', 'jmickle', 'pcn'}}\n\n            g = {'aliases': {}, 'commands': {'cmd.run', 'pillar.get'},\n                 'default_target': {'target': '*', 'tgt_type': 'glob'},\n                 'targets': {}, 'users': {'dmangot', 'jmickle', 'pcn'}}\n\n        Run each of them through ``get_configured_target(('foo', f), 'pillar.get')`` and confirm a valid target\n\n        :type permitted_group: tuple\n        :param permitted_group: A tuple containing the group name and group configuration to check for permission.\n\n        :type cmdline: list\n        :param cmdline: The command sent from Slack formatted as a list.\n\n        :type alias_cmdline: str\n        :param alias_cmdline: An alias to a cmdline.\n\n        "
        null_target = {'target': '*', 'tgt_type': 'glob'}

        def check_cmd_against_group(cmd):
            if False:
                print('Hello World!')
            '\n            Validate cmd against the group to return the target, or a null target\n\n            :type cmd: list\n            :param cmd: The command sent from Slack formatted as a list.\n            '
            (name, group_config) = permitted_group
            target = group_config.get('default_target')
            if not target:
                target = null_target
            if group_config.get('targets'):
                if group_config['targets'].get(cmd):
                    target = group_config['targets'][cmd]
            if not target.get('target'):
                log.debug('Group %s is not configured to have a target for cmd %s.', name, cmd)
            return target
        for this_cl in (cmdline, alias_cmdline):
            (_, kwargs) = self.parse_args_and_kwargs(this_cl)
            if 'target' in kwargs:
                log.debug('target is in kwargs %s.', kwargs)
                if 'tgt_type' in kwargs:
                    log.debug('tgt_type is in kwargs %s.', kwargs)
                    return {'target': kwargs['target'], 'tgt_type': kwargs['tgt_type']}
                return {'target': kwargs['target'], 'tgt_type': 'glob'}
        for this_cl in (cmdline, alias_cmdline):
            checked = check_cmd_against_group(this_cl[0])
            log.debug('this cmdline has target %s.', this_cl)
            if checked.get('target'):
                return checked
        return null_target

    def format_return_text(self, data, function, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Print out YAML using the block mode\n\n        :type user: dict\n        :param token: The return data that needs to be formatted.\n\n        :type user: str\n        :param token: The function that was used to generate the return data.\n        '
        try:
            try:
                outputter = data[next(iter(data))].get('out')
            except (StopIteration, AttributeError):
                outputter = None
            return salt.output.string_format({x: y['return'] for (x, y) in data.items()}, out=outputter, opts=__opts__)
        except Exception as exc:
            import pprint
            log.exception('Exception encountered when trying to serialize %s', pprint.pformat(data))
            return 'Got an error trying to serialze/clean up the response'

    def parse_args_and_kwargs(self, cmdline):
        if False:
            while True:
                i = 10
        '\n\n        :type cmdline: list\n        :param cmdline: The command sent from Slack formatted as a list.\n\n        returns tuple of: args (list), kwargs (dict)\n        '
        args = []
        kwargs = {}
        if len(cmdline) > 1:
            for item in cmdline[1:]:
                if '=' in item:
                    (key, value) = item.split('=', 1)
                    kwargs[key] = value
                else:
                    args.append(item)
        return (args, kwargs)

    def get_jobs_from_runner(self, outstanding_jids):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a list of job_ids, return a dictionary of those job_ids that have\n        completed and their results.\n\n        Query the salt event bus via the jobs runner. jobs.list_job will show\n        a job in progress, jobs.lookup_jid will return a job that has\n        completed.\n\n        :type outstanding_jids: list\n        :param outstanding_jids: The list of job ids to check for completion.\n\n        returns a dictionary of job id: result\n        '
        runner = salt.runner.RunnerClient(__opts__)
        source = __opts__.get('ext_job_cache')
        if not source:
            source = __opts__.get('master_job_cache')
        results = {}
        for jid in outstanding_jids:
            if self.master_minion.returners[f'{source}.get_jid'](jid):
                job_result = runner.cmd('jobs.list_job', [jid])
                jid_result = job_result.get('Result', {})
                jid_function = job_result.get('Function', {})
                results[jid] = {'data': salt.utils.json.loads(salt.utils.json.dumps(jid_result)), 'function': jid_function}
        return results

    def run_commands_from_slack_async(self, message_generator, fire_all, tag, control, interval=1):
        if False:
            i = 10
            return i + 15
        '\n        Pull any pending messages from the message_generator, sending each\n        one to either the event bus, the command_async or both, depending on\n        the values of fire_all and command\n\n        :type message_generator: generator of dict\n        :param message_generator: Generates messages from slack that should be run\n\n        :type fire_all: bool\n        :param fire_all: Whether to also fire messages to the event bus\n\n        :type control: bool\n        :param control: If set to True, whether Slack is allowed to control Salt.\n\n        :type tag: str\n        :param tag: The tag to send to use to send to the event bus\n\n        :type interval: int\n        :param interval: time to wait between ending a loop and beginning the next\n        '
        outstanding = {}
        while self._run_until():
            log.trace('Sleeping for interval of %s', interval)
            time.sleep(interval)
            count = 0
            for msg in message_generator:
                if msg:
                    log.trace('Got a message from the generator: %s', msg.keys())
                    if count > 10:
                        log.warning('Breaking in getting messages because count is exceeded')
                        break
                    if not msg:
                        count += 1
                        log.warning('Skipping an empty message.')
                        continue
                    if msg.get('done'):
                        log.trace('msg is done')
                        break
                    if fire_all:
                        log.debug('Firing message to the bus with tag: %s', tag)
                        log.debug('%s %s', tag, msg)
                        self.fire('{}/{}'.format(tag, msg['message_data'].get('type')), msg)
                    if control and len(msg) > 1 and msg.get('cmdline'):
                        jid = self.run_command_async(msg)
                        log.debug('Submitted a job and got jid: %s', jid)
                        outstanding[jid] = msg
                        text_msg = "@{}'s job is submitted as salt jid {}".format(msg['user_name'], jid)
                        self.app.client.chat_postMessage(channel=msg['channel'], text=text_msg)
                    count += 1
            start_time = time.time()
            job_status = self.get_jobs_from_runner(outstanding.keys())
            log.trace('Getting %s jobs status took %s seconds', len(job_status), time.time() - start_time)
            for jid in job_status:
                result = job_status[jid]['data']
                function = job_status[jid]['function']
                if result:
                    log.debug('ret to send back is %s', result)
                    this_job = outstanding[jid]
                    channel = this_job['channel']
                    return_text = self.format_return_text(result, function)
                    return_prefix = "@{}'s job `{}` (id: {}) (target: {}) returned".format(this_job['user_name'], this_job['cmdline'], jid, this_job['target'])
                    self.app.client.chat_postMessage(channel=channel, text=return_prefix)
                    ts = time.time()
                    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S%f')
                    filename = f'salt-results-{st}.yaml'
                    resp = self.app.client.files_upload(channels=channel, filename=filename, content=return_text)
                    log.debug('Got back %s via the slack client', resp)
                    if 'ok' in resp and resp['ok'] is False:
                        this_job['channel'].send_message('Error: {}'.format(resp['error']))
                    del outstanding[jid]

    def run_command_async(self, msg):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type msg: dict\n        :param msg: The message dictionary that contains the command and all information.\n\n        '
        log.debug('Going to run a command asynchronous')
        runner_functions = sorted(salt.runner.Runner(__opts__).functions)
        cmd = msg['cmdline'][0]
        (args, kwargs) = self.parse_args_and_kwargs(msg['cmdline'])
        if 'pillar' in kwargs:
            kwargs.update(pillar=ast.literal_eval(kwargs['pillar']))
        target = msg['target']['target']
        tgt_type = msg['target']['tgt_type']
        log.debug('target_type is: %s', tgt_type)
        if cmd in runner_functions:
            runner = salt.runner.RunnerClient(__opts__)
            log.debug('Command %s will run via runner_functions', cmd)
            job_id_dict = runner.asynchronous(cmd, {'arg': args, 'kwarg': kwargs})
            job_id = job_id_dict['jid']
        else:
            log.debug('Command %s will run via local.cmd_async, targeting %s', cmd, target)
            log.debug('Running %s, %s, %s, %s, %s', target, cmd, args, kwargs, tgt_type)
            with salt.client.LocalClient() as local:
                job_id = local.cmd_async(str(target), cmd, arg=args, kwarg=kwargs, tgt_type=str(tgt_type))
            log.info('ret from local.cmd_async is %s', job_id)
        return job_id

def start(app_token, bot_token, control=False, trigger='!', groups=None, groups_pillar_name=None, fire_all=False, tag='salt/engines/slack'):
    if False:
        i = 10
        return i + 15
    "\n    Listen to slack events and forward them to salt, new version\n\n    :type app_token: str\n    :param app_token: The Slack application token used by Salt to communicate with Slack.\n\n    :type bot_token: str\n    :param bot_token: The Slack bot token used by Salt to communicate with Slack.\n\n    :type control: bool\n    :param control: Determines whether or not commands sent from Slack with the trigger string will control Salt, defaults to False.\n\n    :type trigger: str\n    :param trigger: The string that should preface all messages in Slack that should be treated as commands to send to Salt.\n\n    :type group: str\n    :param group: The string that should preface all messages in Slack that should be treated as commands to send to Salt.\n\n    :type groups_pillar: str\n    :param group_pillars: A pillar key that can be used to pull group configuration.\n\n    :type fire_all: bool\n    :param fire_all:\n        If set to ``True``, all messages which are not prefixed with\n        the trigger string will fired as events onto Salt's ref:`event bus\n        <event-system>`. The tag for these events will be prefixed with the string\n        specified by the ``tag`` config option (default: ``salt/engines/slack``).\n\n    :type tag: str\n    :param tag: The tag to prefix all events sent to the Salt event bus.\n    "
    if not bot_token or not bot_token.startswith('xoxb'):
        time.sleep(2)
        log.error('Slack bot token not found, bailing...')
        raise UserWarning('Slack Engine bot token not configured')
    try:
        client = SlackClient(app_token=app_token, bot_token=bot_token, trigger_string=trigger)
        message_generator = client.generate_triggered_messages(bot_token, trigger, groups, groups_pillar_name)
        client.run_commands_from_slack_async(message_generator, fire_all, tag, control)
    except Exception:
        raise Exception(f'{traceback.format_exc()}')