"""
Control the Salt command interface
==================================

This state is intended for use from the Salt Master. It provides access to
sending commands down to minions as well as access to executing master-side
modules. These state functions wrap Salt's :ref:`Python API <python-api>`.

    .. versionadded:: 2016.11.0

    Support for masterless minions was added to the ``salt.state`` function,
    so they can run orchestration sls files. This is particularly useful when
    the rendering of a state is dependent on the execution of another state.
    Orchestration will render and execute each orchestration block
    independently, while honoring requisites to ensure the states are applied
    in the correct order.

.. seealso:: More Orchestrate documentation

    * :ref:`Full Orchestrate Tutorial <orchestrate-runner>`
    * :py:func:`The Orchestrate runner <salt.runners.state.orchestrate>`
"""
import fnmatch
import logging
import sys
import threading
import time
import salt.exceptions
import salt.output
import salt.syspaths
import salt.utils.data
import salt.utils.event
import salt.utils.versions
from salt.features import features
log = logging.getLogger(__name__)
__virtualname__ = 'salt'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Named salt\n    '
    return __virtualname__

def _fire_args(tag_data):
    if False:
        for i in range(10):
            print('nop')
    try:
        salt.utils.event.fire_args(__opts__, __orchestration_jid__, tag_data, 'run')
    except NameError:
        log.debug('Unable to fire args event due to missing __orchestration_jid__')

def _parallel_map(func, inputs):
    if False:
        while True:
            i = 10
    '\n    Applies a function to each element of a list, returning the resulting list.\n\n    A separate thread is created for each element in the input list and the\n    passed function is called for each of the elements. When all threads have\n    finished execution a list with the results corresponding to the inputs is\n    returned.\n\n    If one of the threads fails (because the function throws an exception),\n    that exception is reraised. If more than one thread fails, the exception\n    from the first thread (according to the index of the input element) is\n    reraised.\n\n    func:\n        function that is applied on each input element.\n    inputs:\n        list of elements that shall be processed. The length of this list also\n        defines the number of threads created.\n    '
    outputs = len(inputs) * [None]
    errors = len(inputs) * [None]

    def create_thread(index):
        if False:
            for i in range(10):
                print('nop')

        def run_thread():
            if False:
                while True:
                    i = 10
            try:
                outputs[index] = func(inputs[index])
            except:
                errors[index] = sys.exc_info()
        thread = threading.Thread(target=run_thread)
        thread.start()
        return thread
    threads = list(map(create_thread, range(len(inputs))))
    for thread in threads:
        thread.join()
    for error in errors:
        if error is not None:
            (exc_type, exc_value, exc_traceback) = error
            raise exc_value.with_traceback(exc_traceback)
    return outputs

def state(name, tgt, ssh=False, tgt_type='glob', ret='', ret_config=None, ret_kwargs=None, highstate=None, sls=None, top=None, saltenv=None, test=None, pillar=None, pillarenv=None, expect_minions=True, exclude=None, fail_minions=None, allow_fail=0, concurrent=False, timeout=None, batch=None, queue=False, subset=None, orchestration_jid=None, failhard=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Invoke a state run on a given target\n\n    name\n        An arbitrary name used to track the state execution\n\n    tgt\n        The target specification for the state run.\n\n        .. versionadded:: 2016.11.0\n\n        Masterless support: When running on a masterless minion, the ``tgt``\n        is ignored and will always be the local minion.\n\n    tgt_type\n        The target type to resolve, defaults to ``glob``\n\n    ret\n        Optionally set a single or a list of returners to use\n\n    ret_config\n        Use an alternative returner configuration\n\n    ret_kwargs\n        Override individual returner configuration items\n\n    highstate\n        Defaults to None, if set to True the target systems will ignore any\n        sls references specified in the sls option and call state.highstate\n        on the targeted minions\n\n    top\n        Should be the name of a top file. If set state.top is called with this\n        top file instead of state.sls.\n\n    sls\n        A group of sls files to execute. This can be defined as a single string\n        containing a single sls file, or a list of sls files\n\n    test\n        Pass ``test=true`` or ``test=false`` through to the state function. This\n        can be used to override a test mode set in the minion's config file. If\n        left as the default of None and the 'test' mode is supplied on the\n        command line, that value is passed instead.\n\n    pillar\n        Pass the ``pillar`` kwarg through to the state function\n\n    pillarenv\n        The pillar environment to grab pillars from\n\n        .. versionadded:: 2017.7.0\n\n    saltenv\n        The default salt environment to pull sls files from\n\n    ssh\n        Set to `True` to use the ssh client instead of the standard salt client\n\n    roster\n        In the event of using salt-ssh, a roster system can be set\n\n    expect_minions\n        An optional boolean for failing if some minions do not respond\n\n    fail_minions\n        An optional list of targeted minions where failure is an option\n\n    allow_fail\n        Pass in the number of minions to allow for failure before setting\n        the result of the execution to False\n\n    exclude\n        Pass exclude kwarg to state\n\n    concurrent\n        Allow multiple state runs to occur at once.\n\n        WARNING: This flag is potentially dangerous. It is designed\n        for use when multiple state runs can safely be run at the same\n        Do not use this flag for performance optimization.\n\n    queue\n        Pass ``queue=true`` through to the state function\n\n    batch\n        Execute the command :ref:`in batches <targeting-batch>`. E.g.: ``10%``.\n\n        .. versionadded:: 2016.3.0\n\n    subset\n        Number of minions from the targeted set to randomly use\n\n        .. versionadded:: 2017.7.0\n\n    failhard\n        pass failhard down to the executing state\n\n        .. versionadded:: 2019.2.2\n\n    Examples:\n\n    Run a list of sls files via :py:func:`state.sls <salt.state.sls>` on target\n    minions:\n\n    .. code-block:: yaml\n\n        webservers:\n          salt.state:\n            - tgt: 'web*'\n            - sls:\n              - apache\n              - django\n              - core\n            - saltenv: prod\n\n    Run sls file via :py:func:`state.sls <salt.state.sls>` on target\n    minions with exclude:\n\n    .. code-block:: yaml\n\n        docker:\n          salt.state:\n            - tgt: 'docker*'\n            - sls: docker\n            - exclude: docker.swarm\n            - saltenv: prod\n\n    Run a full :py:func:`state.highstate <salt.state.highstate>` on target\n    mininons.\n\n    .. code-block:: yaml\n\n        databases:\n          salt.state:\n            - tgt: role:database\n            - tgt_type: grain\n            - highstate: True\n    "
    cmd_kw = {'arg': [], 'kwarg': {}, 'ret': ret, 'timeout': timeout}
    if ret_config:
        cmd_kw['ret_config'] = ret_config
    if ret_kwargs:
        cmd_kw['ret_kwargs'] = ret_kwargs
    state_ret = {'name': name, 'changes': {}, 'comment': '', 'result': True}
    try:
        allow_fail = int(allow_fail)
    except ValueError:
        state_ret['result'] = False
        state_ret['comment'] = "Passed invalid value for 'allow_fail', must be an int"
        return state_ret
    cmd_kw['tgt_type'] = tgt_type
    cmd_kw['ssh'] = ssh
    if 'roster' in kwargs:
        cmd_kw['roster'] = kwargs['roster']
    cmd_kw['expect_minions'] = expect_minions
    if highstate:
        fun = 'state.highstate'
    elif top:
        fun = 'state.top'
        cmd_kw['arg'].append(top)
    elif sls:
        fun = 'state.sls'
        if isinstance(sls, list):
            sls = ','.join(sls)
        cmd_kw['arg'].append(sls)
    else:
        state_ret['comment'] = 'No highstate or sls specified, no execution made'
        state_ret['result'] = False
        return state_ret
    if test is not None or __opts__.get('test'):
        cmd_kw['kwarg']['test'] = test if test is not None else __opts__.get('test')
    if pillar:
        cmd_kw['kwarg']['pillar'] = pillar
    if pillarenv is not None:
        cmd_kw['kwarg']['pillarenv'] = pillarenv
    if saltenv is not None:
        cmd_kw['kwarg']['saltenv'] = saltenv
    if exclude is not None:
        cmd_kw['kwarg']['exclude'] = exclude
    cmd_kw['kwarg']['queue'] = queue
    if isinstance(concurrent, bool):
        cmd_kw['kwarg']['concurrent'] = concurrent
    else:
        state_ret['comment'] = "Must pass in boolean for value of 'concurrent'"
        state_ret['result'] = False
        return state_ret
    if batch is not None:
        cmd_kw['batch'] = str(batch)
    if subset is not None:
        cmd_kw['subset'] = subset
    if failhard is True or __opts__.get('failhard'):
        cmd_kw['failhard'] = True
    masterless = __opts__['__role'] == 'minion' and __opts__['file_client'] == 'local'
    if not masterless:
        _fire_args({'type': 'state', 'tgt': tgt, 'name': name, 'args': cmd_kw})
        cmd_ret = __salt__['saltutil.cmd'](tgt, fun, **cmd_kw)
    else:
        if top:
            cmd_kw['topfn'] = ''.join(cmd_kw.pop('arg'))
        elif sls:
            cmd_kw['mods'] = ''.join(cmd_kw.pop('arg'))
        cmd_kw.update(cmd_kw.pop('kwarg'))
        tmp_ret = __salt__[fun](**cmd_kw)
        cmd_ret = {__opts__['id']: {'ret': tmp_ret, 'out': tmp_ret.get('out', 'highstate') if isinstance(tmp_ret, dict) else 'highstate'}}
    try:
        state_ret['__jid__'] = cmd_ret[next(iter(cmd_ret))]['jid']
    except (StopIteration, KeyError):
        pass
    changes = {}
    fail = set()
    no_change = set()
    if fail_minions is None:
        fail_minions = ()
    elif isinstance(fail_minions, str):
        fail_minions = [minion.strip() for minion in fail_minions.split(',')]
    elif not isinstance(fail_minions, list):
        state_ret.setdefault('warnings', []).append("'fail_minions' needs to be a list or a comma separated string. Ignored.")
        fail_minions = ()
    if not cmd_ret and expect_minions:
        state_ret['result'] = False
        state_ret['comment'] = 'No minions returned'
        return state_ret
    for (minion, mdata) in cmd_ret.items():
        if mdata.get('out', '') != 'highstate':
            log.warning('Output from salt state not highstate')
        m_ret = False
        if 'return' in mdata and 'ret' not in mdata:
            mdata['ret'] = mdata.pop('return')
        m_state = True
        if mdata.get('failed', False):
            m_state = False
        else:
            try:
                m_ret = mdata['ret']
            except KeyError:
                m_state = False
            if m_state:
                m_state = __utils__['state.check_result'](m_ret, recurse=True)
        if not m_state:
            if minion not in fail_minions:
                fail.add(minion)
            changes[minion] = m_ret
            continue
        try:
            for state_item in m_ret.values():
                if isinstance(state_item, dict):
                    if 'changes' in state_item and state_item['changes']:
                        changes[minion] = m_ret
                        break
            else:
                no_change.add(minion)
        except AttributeError:
            log.error('m_ret did not have changes %s %s', type(m_ret), m_ret)
            no_change.add(minion)
    if changes:
        state_ret['changes'] = {'out': 'highstate', 'ret': changes}
    if len(fail) > allow_fail:
        state_ret['result'] = False
        state_ret['comment'] = 'Run failed on minions: {}'.format(', '.join(fail))
    else:
        state_ret['comment'] = 'States ran successfully.'
        if changes:
            state_ret['comment'] += ' Updating {}.'.format(', '.join(changes))
        if no_change:
            state_ret['comment'] += ' No changes made to {}.'.format(', '.join(no_change))
    if test or __opts__.get('test'):
        if state_ret['changes'] and state_ret['result'] is True:
            state_ret['result'] = None
    return state_ret

def function(name, tgt, ssh=False, tgt_type='glob', ret='', ret_config=None, ret_kwargs=None, expect_minions=False, fail_minions=None, fail_function=None, arg=None, kwarg=None, timeout=None, batch=None, subset=None, failhard=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Execute a single module function on a remote minion via salt or salt-ssh\n\n    name\n        The name of the function to run, aka cmd.run or pkg.install\n\n    tgt\n        The target specification, aka '*' for all minions\n\n    tgt_type\n        The target type, defaults to ``glob``\n\n    arg\n        The list of arguments to pass into the function\n\n    kwarg\n        The dict (not a list) of keyword arguments to pass into the function\n\n    ret\n        Optionally set a single or a list of returners to use\n\n    ret_config\n        Use an alternative returner configuration\n\n    ret_kwargs\n        Override individual returner configuration items\n\n    expect_minions\n        An optional boolean for failing if some minions do not respond\n\n    fail_minions\n        An optional list of targeted minions where failure is an option\n\n    fail_function\n        An optional string that points to a salt module that returns True or False\n        based on the returned data dict for individual minions\n\n    ssh\n        Set to `True` to use the ssh client instead of the standard salt client\n\n    roster\n        In the event of using salt-ssh, a roster system can be set\n\n        .. versionadded:: 3005\n\n    batch\n        Execute the command :ref:`in batches <targeting-batch>`. E.g.: ``10%``.\n\n    subset\n        Number of minions from the targeted set to randomly use\n\n        .. versionadded:: 2017.7.0\n\n    failhard\n        pass failhard down to the executing state\n\n        .. versionadded:: 2019.2.2\n\n    "
    func_ret = {'name': name, 'changes': {}, 'comment': '', 'result': True}
    if kwarg is None:
        kwarg = {}
    if isinstance(arg, str):
        func_ret['warnings'] = ["Please specify 'arg' as a list of arguments."]
        arg = arg.split()
    cmd_kw = {'arg': arg or [], 'kwarg': kwarg, 'ret': ret, 'timeout': timeout}
    if batch is not None:
        cmd_kw['batch'] = str(batch)
    if subset is not None:
        cmd_kw['subset'] = subset
    cmd_kw['tgt_type'] = tgt_type
    cmd_kw['ssh'] = ssh
    if 'roster' in kwargs:
        cmd_kw['roster'] = kwargs['roster']
    cmd_kw['expect_minions'] = expect_minions
    cmd_kw['_cmd_meta'] = True
    if failhard is True or __opts__.get('failhard'):
        cmd_kw['failhard'] = True
    if ret_config:
        cmd_kw['ret_config'] = ret_config
    if ret_kwargs:
        cmd_kw['ret_kwargs'] = ret_kwargs
    fun = name
    if __opts__['test'] is True:
        func_ret['comment'] = 'Function {} would be executed on target {}'.format(fun, tgt)
        func_ret['result'] = None
        return func_ret
    try:
        _fire_args({'type': 'function', 'tgt': tgt, 'name': name, 'args': cmd_kw})
        cmd_ret = __salt__['saltutil.cmd'](tgt, fun, **cmd_kw)
    except Exception as exc:
        func_ret['result'] = False
        func_ret['comment'] = str(exc)
        return func_ret
    try:
        func_ret['__jid__'] = cmd_ret[next(iter(cmd_ret))]['jid']
    except (StopIteration, KeyError):
        pass
    changes = {}
    fail = set()
    if fail_minions is None:
        fail_minions = ()
    elif isinstance(fail_minions, str):
        fail_minions = [minion.strip() for minion in fail_minions.split(',')]
    elif not isinstance(fail_minions, list):
        func_ret.setdefault('warnings', []).append("'fail_minions' needs to be a list or a comma separated string. Ignored.")
        fail_minions = ()
    for (minion, mdata) in cmd_ret.items():
        m_ret = False
        if mdata.get('retcode'):
            func_ret['result'] = False
            fail.add(minion)
        if mdata.get('failed', False):
            m_func = False
        else:
            if 'return' in mdata and 'ret' not in mdata:
                mdata['ret'] = mdata.pop('return')
            m_ret = mdata['ret']
            m_func = not fail_function and True or __salt__[fail_function](m_ret)
            if m_ret is False:
                m_func = False
        if not m_func:
            if minion not in fail_minions:
                fail.add(minion)
        changes[minion] = m_ret
    if not cmd_ret:
        func_ret['result'] = False
        func_ret['comment'] = 'No minions responded'
    else:
        if changes:
            func_ret['changes'] = {'ret': changes}
        if fail:
            func_ret['result'] = False
            func_ret['comment'] = 'Running function {} failed on minions: {}'.format(name, ', '.join(fail))
        else:
            func_ret['comment'] = 'Function ran successfully.'
        if changes:
            func_ret['comment'] += ' Function {} ran on {}.'.format(name, ', '.join(changes))
    return func_ret

def wait_for_event(name, id_list, event_id='id', timeout=300, node='master'):
    if False:
        while True:
            i = 10
    "\n    Watch Salt's event bus and block until a condition is met\n\n    .. versionadded:: 2014.7.0\n\n    name\n        An event tag to watch for; supports Reactor-style globbing.\n    id_list\n        A list of event identifiers to watch for -- usually the minion ID. Each\n        time an event tag is matched the event data is inspected for\n        ``event_id``, if found it is removed from ``id_list``. When ``id_list``\n        is empty this function returns success.\n    event_id : id\n        The name of a key in the event data. Default is ``id`` for the minion\n        ID, another common value is ``name`` for use with orchestrating\n        salt-cloud events.\n    timeout : 300\n        The maximum time in seconds to wait before failing.\n\n    The following example blocks until all the listed minions complete a\n    restart and reconnect to the Salt master:\n\n    .. code-block:: yaml\n\n        reboot_all_minions:\n          salt.function:\n            - name: system.reboot\n            - tgt: '*'\n\n        wait_for_reboots:\n          salt.wait_for_event:\n            - name: salt/minion/*/start\n            - id_list:\n              - jerry\n              - stuart\n              - dave\n              - phil\n              - kevin\n              - mike\n            - require:\n              - salt: reboot_all_minions\n    "
    ret = {'name': name, 'changes': {}, 'comment': '', 'result': False}
    if __opts__.get('test'):
        ret['comment'] = f"Orchestration would wait for event '{name}'"
        ret['result'] = None
        return ret
    with salt.utils.event.get_event(node, __opts__['sock_dir'], opts=__opts__, listen=True) as sevent:
        del_counter = 0
        starttime = time.time()
        timelimit = starttime + timeout
        while True:
            event = sevent.get_event(full=True)
            is_timedout = time.time() > timelimit
            if event is None and (not is_timedout):
                log.trace('wait_for_event: No event data; waiting.')
                continue
            elif event is None and is_timedout:
                ret['comment'] = 'Timeout value reached.'
                return ret
            if fnmatch.fnmatch(event['tag'], name):
                val = event['data'].get(event_id)
                if val is None and 'data' in event['data']:
                    val = event['data']['data'].get(event_id)
                if val is not None:
                    if isinstance(val, list):
                        val_list = [id for id in id_list if id in val]
                        if not val_list:
                            log.trace("wait_for_event: Event identifier '%s' not in id_list; skipping", event_id)
                        elif val_list:
                            minions_seen = ret['changes'].setdefault('minions_seen', [])
                            for found_val in val_list:
                                id_list.remove(found_val)
                                del_counter += 1
                                minions_seen.append(found_val)
                                log.debug("wait_for_event: Event identifier '%s' removed from id_list; %s items remaining.", found_val, len(id_list))
                    else:
                        try:
                            val_idx = id_list.index(val)
                        except ValueError:
                            log.trace("wait_for_event: Event identifier '%s' not in id_list; skipping.", event_id)
                        else:
                            del id_list[val_idx]
                            del_counter += 1
                            minions_seen = ret['changes'].setdefault('minions_seen', [])
                            minions_seen.append(val)
                            log.debug("wait_for_event: Event identifier '%s' removed from id_list; %s items remaining.", val, len(id_list))
                else:
                    log.trace("wait_for_event: Event identifier '%s' not in event '%s'; skipping.", event_id, event['tag'])
            else:
                log.debug("wait_for_event: Skipping unmatched event '%s'", event['tag'])
            if len(id_list) == 0:
                ret['result'] = True
                ret['comment'] = 'All events seen in {} seconds.'.format(time.time() - starttime)
                return ret
            if is_timedout:
                ret['comment'] = 'Timeout value reached.'
                return ret

def runner(name, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Execute a runner module on the master\n\n    .. versionadded:: 2014.7.0\n\n    name\n        The name of the function to run\n    kwargs\n        Any keyword arguments to pass to the runner function\n\n    .. code-block:: yaml\n\n         run-manage-up:\n          salt.runner:\n            - name: manage.up\n    '
    try:
        jid = __orchestration_jid__
    except NameError:
        log.debug('Unable to fire args event due to missing __orchestration_jid__')
        jid = None
    try:
        kwargs['__pub_user'] = __user__
        log.debug(f"added __pub_user to kwargs using dunder user '{__user__}', kwargs '{kwargs}'")
    except NameError:
        log.warning('unable to find user for fire args event due to missing __user__')
    if __opts__.get('test', False):
        ret = {'name': name, 'result': None, 'changes': {}, 'comment': f"Runner function '{name}' would be executed."}
        return ret
    out = __salt__['saltutil.runner'](name, __orchestration_jid__=jid, __env__=__env__, full_return=True, **kwargs)
    runner_return = out.get('return')
    if isinstance(runner_return, dict) and 'Error' in runner_return:
        out['success'] = False
    success = out.get('success', True)
    ret = {'name': name, 'changes': {'return': runner_return}, 'result': success}
    ret['comment'] = "Runner function '{}' {}.".format(name, 'executed' if success else 'failed')
    if features.get('enable_deprecated_orchestration_flag', False):
        ret['__orchestration__'] = True
        salt.utils.versions.warn_until(3008, 'The __orchestration__ return flag will be removed in Salt Argon. For more information see https://github.com/saltstack/salt/pull/59917.')
    if 'jid' in out:
        ret['__jid__'] = out['jid']
    return ret

def parallel_runners(name, runners, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Executes multiple runner modules on the master in parallel.\n\n    .. versionadded:: 2018.3.0\n\n    A separate thread is spawned for each runner. This state is intended to be\n    used with the orchestrate runner in place of the ``saltmod.runner`` state\n    when different tasks should be run in parallel. In general, Salt states are\n    not safe when used concurrently, so ensure that they are used in a safe way\n    (e.g. by only targeting separate minions in parallel tasks).\n\n    name:\n        name identifying this state. The name is provided as part of the\n        output, but not used for anything else.\n\n    runners:\n        list of runners that should be run in parallel. Each element of the\n        list has to be a dictionary. This dictionary's name entry stores the\n        name of the runner function that shall be invoked. The optional kwarg\n        entry stores a dictionary of named arguments that are passed to the\n        runner function.\n\n    .. code-block:: yaml\n\n        parallel-state:\n           salt.parallel_runners:\n             - runners:\n                 my_runner_1:\n                   - name: state.orchestrate\n                   - kwarg:\n                       mods: orchestrate_state_1\n                 my_runner_2:\n                   - name: state.orchestrate\n                   - kwarg:\n                       mods: orchestrate_state_2\n    "
    if isinstance(runners, str):
        runners = {runners: [{name: runners}]}
    if not isinstance(runners, dict):
        return {'name': name, 'result': False, 'changes': {}, 'comment': 'The runners parameter must be a string or dict.'}
    for (runner_id, runner_config) in runners.items():
        if runner_config is None:
            runner_config = {}
        else:
            runner_config = salt.utils.data.repack_dictlist(runner_config)
        if 'name' not in runner_config:
            runner_config['name'] = runner_id
        runners[runner_id] = runner_config
    try:
        jid = __orchestration_jid__
    except NameError:
        log.debug('Unable to fire args event due to missing __orchestration_jid__')
        jid = None

    def call_runner(runner_config):
        if False:
            while True:
                i = 10
        return __salt__['saltutil.runner'](runner_config['name'], __orchestration_jid__=jid, __env__=__env__, full_return=True, **runner_config.get('kwarg', {}))
    try:
        outputs = _parallel_map(call_runner, list(runners.values()))
    except salt.exceptions.SaltException as exc:
        return {'name': name, 'result': False, 'success': False, 'changes': {}, 'comment': f'One of the runners raised an exception: {exc}'}
    outputs = {runner_id: out['return'] for (runner_id, out) in zip(runners.keys(), outputs)}
    highstate_output = all([out.get('outputter', '') == 'highstate' and 'data' in out for out in outputs.values()])

    def extract_changes(obj):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(obj, dict):
            return {}
        elif 'changes' in obj:
            if isinstance(obj['changes'], dict) and obj['changes'].get('out', '') == 'highstate' and ('ret' in obj['changes']):
                return obj['changes']['ret']
            else:
                return obj['changes']
        else:
            found_changes = {}
            for (key, value) in obj.items():
                change = extract_changes(value)
                if change:
                    found_changes[key] = change
            return found_changes
    if highstate_output:
        failed_runners = [runner_id for (runner_id, out) in outputs.items() if out['data'].get('retcode', 0) != 0]
        all_successful = not failed_runners
        if all_successful:
            comment = 'All runner functions executed successfully.'
        else:
            runner_comments = ['Runner {} failed with return value:\n{}'.format(runner_id, salt.output.out_format(outputs[runner_id], 'nested', __opts__, nested_indent=2)) for runner_id in failed_runners]
            comment = '\n'.join(runner_comments)
        changes = {}
        for (runner_id, out) in outputs.items():
            runner_changes = extract_changes(out['data'])
            if runner_changes:
                changes[runner_id] = runner_changes
    else:
        failed_runners = [runner_id for (runner_id, out) in outputs.items() if out.get('exit_code', 0) != 0]
        all_successful = not failed_runners
        if all_successful:
            comment = 'All runner functions executed successfully.'
        elif len(failed_runners) == 1:
            comment = f'Runner {failed_runners[0]} failed.'
        else:
            comment = 'Runners {} failed.'.format(', '.join(failed_runners))
        changes = {'ret': {runner_id: out for (runner_id, out) in outputs.items()}}
    ret = {'name': name, 'result': all_successful, 'changes': changes, 'comment': comment}
    return ret

def wheel(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Execute a wheel module on the master\n\n    .. versionadded:: 2014.7.0\n\n    name\n        The name of the function to run\n    kwargs\n        Any keyword arguments to pass to the wheel function\n\n    .. code-block:: yaml\n\n        accept_minion_key:\n          salt.wheel:\n            - name: key.accept\n            - match: frank\n    '
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    try:
        jid = __orchestration_jid__
    except NameError:
        log.debug('Unable to fire args event due to missing __orchestration_jid__')
        jid = None
    if __opts__.get('test', False):
        ret['result'] = None
        ret['changes'] = {}
        ret['comment'] = f"Wheel function '{name}' would be executed."
        return ret
    out = __salt__['saltutil.wheel'](name, __orchestration_jid__=jid, __env__=__env__, **kwargs)
    wheel_return = out.get('return')
    if isinstance(wheel_return, dict) and 'Error' in wheel_return:
        out['success'] = False
    success = out.get('success', True)
    ret = {'name': name, 'changes': {'return': wheel_return}, 'result': success}
    ret['comment'] = "Wheel function '{}' {}.".format(name, 'executed' if success else 'failed')
    if features.get('enable_deprecated_orchestration_flag', False):
        ret['__orchestration__'] = True
        salt.utils.versions.warn_until(3008, 'The __orchestration__ return flag will be removed in Salt Argon. For more information see https://github.com/saltstack/salt/pull/59917.')
    if 'jid' in out:
        ret['__jid__'] = out['jid']
    return ret