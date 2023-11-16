"""
Execution module to work with etcd

:depends:  - python-etcd or etcd3-py

Configuration
-------------

To work with an etcd server you must configure an etcd profile. The etcd config
can be set in either the Salt Minion configuration file or in pillar:

.. code-block:: yaml

    my_etd_config:
      etcd.host: 127.0.0.1
      etcd.port: 4001

It is technically possible to configure etcd without using a profile, but this
is not considered to be a best practice, especially when multiple etcd servers
or clusters are available.

.. code-block:: yaml

    etcd.host: 127.0.0.1
    etcd.port: 4001

In order to choose whether to use etcd API v2 or v3, you can put the following
configuration option in the same place as your etcd configuration.  This option
defaults to true, meaning you will use v2 unless you specify otherwise.

.. code-block:: yaml

    etcd.require_v2: True

When using API v3, there are some specific options available to be configured
within your etcd profile.  They are defaulted to the following...

.. code-block:: yaml

    etcd.encode_keys: False
    etcd.encode_values: True
    etcd.raw_keys: False
    etcd.raw_values: False
    etcd.unicode_errors: "surrogateescape"

``etcd.encode_keys`` indicates whether you want to pre-encode keys using msgpack before
adding them to etcd.

.. note::

    If you set ``etcd.encode_keys`` to ``True``, all recursive functionality will no longer work.
    This includes ``tree`` and ``ls`` and all other methods if you set ``recurse``/``recursive`` to ``True``.
    This is due to the fact that when encoding with msgpack, keys like ``/salt`` and ``/salt/stack`` will have
    differing byte prefixes, and etcd v3 searches recursively using prefixes.

``etcd.encode_values`` indicates whether you want to pre-encode values using msgpack before
adding them to etcd.  This defaults to ``True`` to avoid data loss on non-string values wherever possible.

``etcd.raw_keys`` determines whether you want the raw key or a string returned.

``etcd.raw_values`` determines whether you want the raw value or a string returned.

``etcd.unicode_errors`` determines what you policy to follow when there are encoding/decoding errors.

.. note::

    The etcd configuration can also be set in the Salt Master config file,
    but in order to use any etcd configurations defined in the Salt Master
    config, the :conf_master:`pillar_opts` must be set to ``True``.

    Be aware that setting ``pillar_opts`` to ``True`` has security implications
    as this makes all master configuration settings available in all minion's
    pillars.

"""
import logging
import salt.utils.etcd_util
__virtualname__ = 'etcd'
log = logging.getLogger(__name__)
__func_alias__ = {'get_': 'get', 'set_': 'set', 'rm_': 'rm', 'ls_': 'ls'}

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only return if python-etcd is installed\n    '
    if salt.utils.etcd_util.HAS_ETCD_V2 or salt.utils.etcd_util.HAS_ETCD_V3:
        return __virtualname__
    return (False, 'The etcd_mod execution module cannot be loaded: python etcd library not available.')

def get_(key, recurse=False, profile=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 2014.7.0\n\n    Get a value from etcd, by direct path.  Returns None on failure.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion etcd.get /path/to/key\n        salt myminion etcd.get /path/to/key profile=my_etcd_config\n        salt myminion etcd.get /path/to/key recurse=True profile=my_etcd_config\n        salt myminion etcd.get /path/to/key host=127.0.0.1 port=2379\n    '
    client = __utils__['etcd_util.get_conn'](__opts__, profile, **kwargs)
    return client.get(key, recurse=recurse)

def set_(key, value, profile=None, ttl=None, directory=False, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2014.7.0\n\n    Set a key in etcd by direct path. Optionally, create a directory\n    or set a TTL on the key.  Returns None on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion etcd.set /path/to/key value\n        salt myminion etcd.set /path/to/key value profile=my_etcd_config\n        salt myminion etcd.set /path/to/key value host=127.0.0.1 port=2379\n        salt myminion etcd.set /path/to/dir '' directory=True\n        salt myminion etcd.set /path/to/key value ttl=5\n    "
    client = __utils__['etcd_util.get_conn'](__opts__, profile, **kwargs)
    return client.set(key, value, ttl=ttl, directory=directory)

def update(fields, path='', profile=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2016.3.0\n\n    Sets a dictionary of values in one call.  Useful for large updates\n    in syndic environments.  The dictionary can contain a mix of formats\n    such as:\n\n    .. code-block:: python\n\n        {\n          \'/some/example/key\': \'bar\',\n          \'/another/example/key\': \'baz\'\n        }\n\n    Or it may be a straight dictionary, which will be flattened to look\n    like the above format:\n\n    .. code-block:: python\n\n        {\n            \'some\': {\n                \'example\': {\n                    \'key\': \'bar\'\n                }\n            },\n            \'another\': {\n                \'example\': {\n                    \'key\': \'baz\'\n                }\n            }\n        }\n\n    You can even mix the two formats and it will be flattened to the first\n    format.  Leading and trailing \'/\' will be removed.\n\n    Empty directories can be created by setting the value of the key to an\n    empty dictionary.\n\n    The \'path\' parameter will optionally set the root of the path to use.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion etcd.update "{\'/path/to/key\': \'baz\', \'/another/key\': \'bar\'}"\n        salt myminion etcd.update "{\'/path/to/key\': \'baz\', \'/another/key\': \'bar\'}" profile=my_etcd_config\n        salt myminion etcd.update "{\'/path/to/key\': \'baz\', \'/another/key\': \'bar\'}" host=127.0.0.1 port=2379\n        salt myminion etcd.update "{\'/path/to/key\': \'baz\', \'/another/key\': \'bar\'}" path=\'/some/root\'\n    '
    client = __utils__['etcd_util.get_conn'](__opts__, profile, **kwargs)
    return client.update(fields, path)

def watch(key, recurse=False, profile=None, timeout=0, index=None, **kwargs):
    if False:
        return 10
    '\n    .. versionadded:: 2016.3.0\n\n    Makes a best effort to watch for a key or tree change in etcd.\n    Returns a dict containing the new key value ( or None if the key was\n    deleted ), the modifiedIndex of the key, whether the key changed or\n    not, the path to the key that changed and whether it is a directory or not.\n\n    If something catastrophic happens, returns {}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion etcd.watch /path/to/key\n        salt myminion etcd.watch /path/to/key timeout=10\n        salt myminion etcd.watch /patch/to/key profile=my_etcd_config index=10\n        salt myminion etcd.watch /patch/to/key host=127.0.0.1 port=2379\n    '
    client = __utils__['etcd_util.get_conn'](__opts__, profile, **kwargs)
    return client.watch(key, recurse=recurse, timeout=timeout, index=index)

def ls_(path='/', profile=None, **kwargs):
    if False:
        return 10
    '\n    .. versionadded:: 2014.7.0\n\n    Return all keys and dirs inside a specific path. Returns an empty dict on\n    failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion etcd.ls /path/to/dir/\n        salt myminion etcd.ls /path/to/dir/ profile=my_etcd_config\n        salt myminion etcd.ls /path/to/dir/ host=127.0.0.1 port=2379\n    '
    client = __utils__['etcd_util.get_conn'](__opts__, profile, **kwargs)
    return client.ls(path)

def rm_(key, recurse=False, profile=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2014.7.0\n\n    Delete a key from etcd.  Returns True if the key was deleted, False if it was\n    not and None if there was a failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion etcd.rm /path/to/key\n        salt myminion etcd.rm /path/to/key profile=my_etcd_config\n        salt myminion etcd.rm /path/to/key host=127.0.0.1 port=2379\n        salt myminion etcd.rm /path/to/dir recurse=True profile=my_etcd_config\n    '
    client = __utils__['etcd_util.get_conn'](__opts__, profile, **kwargs)
    return client.rm(key, recurse=recurse)

def tree(path='/', profile=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionadded:: 2014.7.0\n\n    Recurse through etcd and return all values.  Returns None on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion etcd.tree\n        salt myminion etcd.tree profile=my_etcd_config\n        salt myminion etcd.tree host=127.0.0.1 port=2379\n        salt myminion etcd.tree /path/to/keys profile=my_etcd_config\n    '
    client = __utils__['etcd_util.get_conn'](__opts__, profile, **kwargs)
    return client.tree(path)