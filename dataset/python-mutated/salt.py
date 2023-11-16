"""
This runner makes Salt's
execution modules available
on the salt master.

.. versionadded:: 2016.11.0

.. _salt_salt_runner:

Salt's execution modules are normally available
on the salt minion. Use this runner to call
execution modules on the salt master.
Salt :ref:`execution modules <writing-execution-modules>`
are the functions called by the ``salt`` command.

Execution modules can be called with ``salt-run``:

.. code-block:: bash

    salt-run salt.cmd test.ping
    # call functions with arguments and keyword arguments
    salt-run salt.cmd test.arg 1 2 3 key=value a=1

Execution modules are also available to salt runners:

.. code-block:: python

    __salt__['salt.cmd'](fun=fun, args=args, kwargs=kwargs)

"""
import copy
import logging
import salt.client
import salt.loader
import salt.pillar
import salt.utils.args
from salt.exceptions import SaltClientError
log = logging.getLogger(__name__)

def cmd(fun, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    .. versionchanged:: 2018.3.0\n        Added ``with_pillar`` argument\n\n    Execute ``fun`` with the given ``args`` and ``kwargs``.  Parameter ``fun``\n    should be the string :ref:`name <all-salt.modules>` of the execution module\n    to call.\n\n    .. note::\n        Execution modules will be loaded *every time* this function is called.\n        Additionally, keep in mind that since runners execute on the master,\n        custom execution modules will need to be synced to the master using\n        :py:func:`salt-run saltutil.sync_modules\n        <salt.runners.saltutil.sync_modules>`, otherwise they will not be\n        available.\n\n    with_pillar : False\n        If ``True``, pillar data will be compiled for the master\n\n        .. note::\n            To target the master in the pillar top file, keep in mind that the\n            default ``id`` for the master is ``<hostname>_master``. This can be\n            overridden by setting an ``id`` configuration parameter in the\n            master config file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run salt.cmd test.ping\n        # call functions with arguments and keyword arguments\n        salt-run salt.cmd test.arg 1 2 3 a=1\n        salt-run salt.cmd mymod.myfunc with_pillar=True\n    '
    log.debug('Called salt.cmd runner with minion function %s', fun)
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    with_pillar = kwargs.pop('with_pillar', False)
    opts = copy.deepcopy(__opts__)
    opts['grains'] = salt.loader.grains(opts)
    if with_pillar:
        opts['pillar'] = salt.pillar.get_pillar(opts, opts['grains'], opts['id'], saltenv=opts['saltenv'], pillarenv=opts.get('pillarenv')).compile_pillar()
    else:
        opts['pillar'] = {}
    functions = salt.loader.minion_mods(opts, utils=salt.loader.utils(opts), context=__context__)
    return functions[fun](*args, **kwargs) if fun in functions else "'{}' is not available.".format(fun)

def execute(tgt, fun, arg=(), timeout=None, tgt_type='glob', ret='', jid='', kwarg=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2017.7.0\n\n    Execute ``fun`` on all minions matched by ``tgt`` and ``tgt_type``.\n    Parameter ``fun`` is the name of execution module function to call.\n\n    This function should mainly be used as a helper for runner modules,\n    in order to avoid redundant code.\n    For example, when inside a runner one needs to execute a certain function\n    on arbitrary groups of minions, only has to:\n\n    .. code-block:: python\n\n        ret1 = __salt__['salt.execute']('*', 'mod.fun')\n        ret2 = __salt__['salt.execute']('my_nodegroup', 'mod2.fun2', tgt_type='nodegroup')\n\n    It can also be used to schedule jobs directly on the master, for example:\n\n    .. code-block:: yaml\n\n        schedule:\n            collect_bgp_stats:\n                function: salt.execute\n                args:\n                    - edge-routers\n                    - bgp.neighbors\n                kwargs:\n                    tgt_type: nodegroup\n                days: 1\n                returner: redis\n    "
    with salt.client.get_local_client(__opts__['conf_file']) as client:
        try:
            return client.cmd(tgt, fun, arg=arg, timeout=timeout or __opts__['timeout'], tgt_type=tgt_type, ret=ret, jid=jid, kwarg=kwarg, **kwargs)
        except SaltClientError as client_error:
            log.error('Error while executing %s on %s (%s)', fun, tgt, tgt_type)
            log.error(client_error)
            return {}