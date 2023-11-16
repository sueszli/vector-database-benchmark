"""
The ``file_tree`` external pillar allows values from all files in a directory
tree to be imported as Pillar data.

.. note::

    This is an external pillar and is subject to the :ref:`rules and
    constraints <external-pillars>` governing external pillars.

.. versionadded:: 2015.5.0

In this pillar, data is organized by either Minion ID or Nodegroup name.  To
setup pillar data for a specific Minion, place it in
``<root_dir>/hosts/<minion_id>``.  To setup pillar data for an entire
Nodegroup, place it in ``<root_dir>/nodegroups/<node_group>`` where
``<node_group>`` is the Nodegroup's name.

Example ``file_tree`` Pillar
============================

Master Configuration
--------------------

.. code-block:: yaml

    ext_pillar:
      - file_tree:
          root_dir: /srv/ext_pillar
          follow_dir_links: False
          keep_newline: True

The ``root_dir`` parameter is required and points to the directory where files
for each host are stored. The ``follow_dir_links`` parameter is optional and
defaults to False. If ``follow_dir_links`` is set to True, this external pillar
will follow symbolic links to other directories.

.. warning::
    Be careful when using ``follow_dir_links``, as a recursive symlink chain
    will result in unexpected results.

.. versionchanged:: 2018.3.0
    If ``root_dir`` is a relative path, it will be treated as relative to the
    :conf_master:`pillar_roots` of the environment specified by
    :conf_minion:`pillarenv`. If an environment specifies multiple
    roots, this module will search for files relative to all of them, in order,
    merging the results.

If ``keep_newline`` is set to ``True``, then the pillar values for files ending
in newlines will keep that newline. The default behavior is to remove the
end-of-file newline. ``keep_newline`` should be turned on if the pillar data is
intended to be used to deploy a file using ``contents_pillar`` with a
:py:func:`file.managed <salt.states.file.managed>` state.

.. versionchanged:: 2015.8.4
    The ``raw_data`` parameter has been renamed to ``keep_newline``. In earlier
    releases, ``raw_data`` must be used. Also, this parameter can now be a list
    of globs, allowing for more granular control over which pillar values keep
    their end-of-file newline. The globs match paths relative to the
    directories named for minion IDs and nodegroups underneath the ``root_dir``
    (see the layout examples in the below sections).

    .. code-block:: yaml

        ext_pillar:
          - file_tree:
              root_dir: /path/to/root/directory
              keep_newline:
                - files/testdir/*

.. note::
    In earlier releases, this documentation incorrectly stated that binary
    files would not affected by the ``keep_newline`` configuration.  However,
    this module does not actually distinguish between binary and text files.

.. versionchanged:: 2017.7.0
    Templating/rendering has been added. You can now specify a default render
    pipeline and a black- and whitelist of (dis)allowed renderers.

    ``template`` must be set to ``True`` for templating to happen.

    .. code-block:: yaml

        ext_pillar:
          - file_tree:
            root_dir: /path/to/root/directory
            render_default: jinja|yaml
            renderer_blacklist:
              - gpg
            renderer_whitelist:
              - jinja
              - yaml
            template: True

Assigning Pillar Data to Individual Hosts
-----------------------------------------

To configure pillar data for each host, this external pillar will recursively
iterate over ``root_dir``/hosts/``id`` (where ``id`` is a minion ID), and
compile pillar data with each subdirectory as a dictionary key and each file
as a value.

For example, the following ``root_dir`` tree:

.. code-block:: text

    ./hosts/
    ./hosts/test-host/
    ./hosts/test-host/files/
    ./hosts/test-host/files/testdir/
    ./hosts/test-host/files/testdir/file1.txt
    ./hosts/test-host/files/testdir/file2.txt
    ./hosts/test-host/files/another-testdir/
    ./hosts/test-host/files/another-testdir/symlink-to-file1.txt

will result in the following pillar tree for minion with ID ``test-host``:

.. code-block:: text

    test-host:
        ----------
        apache:
            ----------
            config.d:
                ----------
                00_important.conf:
                    <important_config important_setting="yes" />
                20_bob_extra.conf:
                    <bob_specific_cfg has_freeze_ray="yes" />
        corporate_app:
            ----------
            settings:
                ----------
                common_settings:
                    // This is the main settings file for the corporate
                    // internal web app
                    main_setting: probably
                bob_settings:
                    role: bob

.. note::

    The leaf data in the example shown is the contents of the pillar files.
"""
import fnmatch
import logging
import os
import salt.loader
import salt.template
import salt.utils.dictupdate
import salt.utils.files
import salt.utils.minions
import salt.utils.path
import salt.utils.stringio
import salt.utils.stringutils
log = logging.getLogger(__name__)

def _on_walk_error(err):
    if False:
        while True:
            i = 10
    '\n    Log salt.utils.path.os_walk() error.\n    '
    log.error('%s: %s', err.filename, err.strerror)

def _check_newline(prefix, file_name, keep_newline):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a boolean stating whether or not a file's trailing newline should be\n    removed. To figure this out, first check if keep_newline is a boolean and\n    if so, return its opposite. Otherwise, iterate over keep_newline and check\n    if any of the patterns match the file path. If a match is found, return\n    False, otherwise return True.\n    "
    if isinstance(keep_newline, bool):
        return not keep_newline
    full_path = os.path.join(prefix, file_name)
    for pattern in keep_newline:
        try:
            if fnmatch.fnmatch(full_path, pattern):
                return False
        except TypeError:
            if fnmatch.fnmatch(full_path, str(pattern)):
                return False
    return True

def _construct_pillar(top_dir, follow_dir_links, keep_newline=False, render_default=None, renderer_blacklist=None, renderer_whitelist=None, template=False):
    if False:
        i = 10
        return i + 15
    '\n    Construct pillar from file tree.\n    '
    pillar = {}
    renderers = salt.loader.render(__opts__, __salt__)
    norm_top_dir = os.path.normpath(top_dir)
    for (dir_path, dir_names, file_names) in salt.utils.path.os_walk(top_dir, topdown=True, onerror=_on_walk_error, followlinks=follow_dir_links):
        pillar_node = pillar
        norm_dir_path = os.path.normpath(dir_path)
        prefix = os.path.relpath(norm_dir_path, norm_top_dir)
        if norm_dir_path != norm_top_dir:
            path_parts = []
            head = prefix
            while head:
                (head, tail) = os.path.split(head)
                path_parts.insert(0, tail)
            while path_parts:
                pillar_node = pillar_node[path_parts.pop(0)]
        for dir_name in dir_names:
            pillar_node[dir_name] = {}
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            if not os.path.isfile(file_path):
                log.error('file_tree: %s: not a regular file', file_path)
                continue
            contents = b''
            try:
                with salt.utils.files.fopen(file_path, 'rb') as fhr:
                    buf = fhr.read(__opts__['file_buffer_size'])
                    while buf:
                        contents += buf
                        buf = fhr.read(__opts__['file_buffer_size'])
                    if contents.endswith(b'\n') and _check_newline(prefix, file_name, keep_newline):
                        contents = contents[:-1]
            except OSError as exc:
                log.error('file_tree: Error reading %s: %s', file_path, exc.strerror)
            else:
                data = contents
                if template is True:
                    data = salt.template.compile_template_str(template=salt.utils.stringutils.to_unicode(contents), renderers=renderers, default=render_default, blacklist=renderer_blacklist, whitelist=renderer_whitelist)
                if salt.utils.stringio.is_readable(data):
                    pillar_node[file_name] = data.getvalue()
                else:
                    pillar_node[file_name] = data
    return pillar

def ext_pillar(minion_id, pillar, root_dir=None, follow_dir_links=False, debug=False, keep_newline=False, render_default=None, renderer_blacklist=None, renderer_whitelist=None, template=False):
    if False:
        return 10
    "\n    Compile pillar data from the given ``root_dir`` specific to Nodegroup names\n    and Minion IDs.\n\n    If a Minion's ID is not found at ``<root_dir>/host/<minion_id>`` or if it\n    is not included in any Nodegroups named at\n    ``<root_dir>/nodegroups/<node_group>``, no pillar data provided by this\n    pillar module will be available for that Minion.\n\n    .. versionchanged:: 2017.7.0\n        Templating/rendering has been added. You can now specify a default\n        render pipeline and a black- and whitelist of (dis)allowed renderers.\n\n        ``template`` must be set to ``True`` for templating to happen.\n\n        .. code-block:: yaml\n\n            ext_pillar:\n              - file_tree:\n                root_dir: /path/to/root/directory\n                render_default: jinja|yaml\n                renderer_blacklist:\n                  - gpg\n                renderer_whitelist:\n                  - jinja\n                  - yaml\n                template: True\n\n    :param minion_id:\n        The ID of the Minion whose pillar data is to be collected\n\n    :param pillar:\n        Unused by the ``file_tree`` pillar module\n\n    :param root_dir:\n        Filesystem directory used as the root for pillar data (e.g.\n        ``/srv/ext_pillar``)\n\n        .. versionchanged:: 2018.3.0\n            If ``root_dir`` is a relative path, it will be treated as relative to the\n            :conf_master:`pillar_roots` of the environment specified by\n            :conf_minion:`pillarenv`. If an environment specifies multiple\n            roots, this module will search for files relative to all of them, in order,\n            merging the results.\n\n    :param follow_dir_links:\n        Follow symbolic links to directories while collecting pillar files.\n        Defaults to ``False``.\n\n        .. warning::\n\n            Care should be exercised when enabling this option as it will\n            follow links that point outside of ``root_dir``.\n\n        .. warning::\n\n            Symbolic links that lead to infinite recursion are not filtered.\n\n    :param debug:\n        Enable debug information at log level ``debug``.  Defaults to\n        ``False``.  This option may be useful to help debug errors when setting\n        up the ``file_tree`` pillar module.\n\n    :param keep_newline:\n        Preserve the end-of-file newline in files.  Defaults to ``False``.\n        This option may either be a boolean or a list of file globs (as defined\n        by the `Python fnmatch package\n        <https://docs.python.org/library/fnmatch.html>`_) for which end-of-file\n        newlines are to be kept.\n\n        ``keep_newline`` should be turned on if the pillar data is intended to\n        be used to deploy a file using ``contents_pillar`` with a\n        :py:func:`file.managed <salt.states.file.managed>` state.\n\n        .. versionchanged:: 2015.8.4\n            The ``raw_data`` parameter has been renamed to ``keep_newline``. In\n            earlier releases, ``raw_data`` must be used. Also, this parameter\n            can now be a list of globs, allowing for more granular control over\n            which pillar values keep their end-of-file newline. The globs match\n            paths relative to the directories named for Minion IDs and\n            Nodegroup namess underneath the ``root_dir``.\n\n            .. code-block:: yaml\n\n                ext_pillar:\n                  - file_tree:\n                      root_dir: /srv/ext_pillar\n                      keep_newline:\n                        - apache/config.d/*\n                        - corporate_app/settings/*\n\n        .. note::\n            In earlier releases, this documentation incorrectly stated that\n            binary files would not affected by the ``keep_newline``.  However,\n            this module does not actually distinguish between binary and text\n            files.\n\n\n    :param render_default:\n        Override Salt's :conf_master:`default global renderer <renderer>` for\n        the ``file_tree`` pillar.\n\n        .. code-block:: yaml\n\n            render_default: jinja\n\n    :param renderer_blacklist:\n        Disallow renderers for pillar files.\n\n        .. code-block:: yaml\n\n            renderer_blacklist:\n              - json\n\n    :param renderer_whitelist:\n        Allow renderers for pillar files.\n\n        .. code-block:: yaml\n\n            renderer_whitelist:\n              - yaml\n              - jinja\n\n    :param template:\n        Enable templating of pillar files.  Defaults to ``False``.\n    "
    del pillar
    if not root_dir:
        log.error('file_tree: no root_dir specified')
        return {}
    if not os.path.isabs(root_dir):
        pillarenv = __opts__['pillarenv']
        if pillarenv is None:
            log.error('file_tree: root_dir is relative but pillarenv is not set')
            return {}
        log.debug('file_tree: pillarenv = %s', pillarenv)
        env_roots = __opts__['pillar_roots'].get(pillarenv, None)
        if env_roots is None:
            log.error('file_tree: root_dir is relative but no pillar_roots are specified  for pillarenv %s', pillarenv)
            return {}
        env_dirs = []
        for env_root in env_roots:
            env_dir = os.path.normpath(os.path.join(env_root, root_dir))
            if env_dir not in env_dirs or env_dir != env_dirs[-1]:
                env_dirs.append(env_dir)
        dirs = env_dirs
    else:
        dirs = [root_dir]
    result_pillar = {}
    for root in dirs:
        dir_pillar = _ext_pillar(minion_id, root, follow_dir_links, debug, keep_newline, render_default, renderer_blacklist, renderer_whitelist, template)
        result_pillar = salt.utils.dictupdate.merge(result_pillar, dir_pillar, strategy='recurse')
    return result_pillar

def _ext_pillar(minion_id, root_dir, follow_dir_links, debug, keep_newline, render_default, renderer_blacklist, renderer_whitelist, template):
    if False:
        i = 10
        return i + 15
    '\n    Compile pillar data for a single root_dir for the specified minion ID\n    '
    log.debug('file_tree: reading %s', root_dir)
    if not os.path.isdir(root_dir):
        log.error('file_tree: root_dir %s does not exist or is not a directory', root_dir)
        return {}
    if not isinstance(keep_newline, (bool, list)):
        log.error('file_tree: keep_newline must be either True/False or a list of file globs. Skipping this ext_pillar for root_dir %s', root_dir)
        return {}
    ngroup_pillar = {}
    nodegroups_dir = os.path.join(root_dir, 'nodegroups')
    if os.path.exists(nodegroups_dir) and len(__opts__.get('nodegroups', ())) > 0:
        master_ngroups = __opts__['nodegroups']
        ext_pillar_dirs = os.listdir(nodegroups_dir)
        if len(ext_pillar_dirs) > 0:
            for nodegroup in ext_pillar_dirs:
                if os.path.isdir(nodegroups_dir) and nodegroup in master_ngroups:
                    ckminions = salt.utils.minions.CkMinions(__opts__)
                    _res = ckminions.check_minions(master_ngroups[nodegroup], 'compound')
                    match = _res['minions']
                    if minion_id in match:
                        ngroup_dir = os.path.join(nodegroups_dir, str(nodegroup))
                        ngroup_pillar = salt.utils.dictupdate.merge(ngroup_pillar, _construct_pillar(ngroup_dir, follow_dir_links, keep_newline, render_default, renderer_blacklist, renderer_whitelist, template), strategy='recurse')
        elif debug is True:
            log.debug('file_tree: no nodegroups found in file tree directory %s, skipping...', ext_pillar_dirs)
    elif debug is True:
        log.debug('file_tree: no nodegroups found in master configuration')
    host_dir = os.path.join(root_dir, 'hosts', minion_id)
    if not os.path.exists(host_dir):
        if debug is True:
            log.debug('file_tree: no pillar data for minion %s found in file tree directory %s', minion_id, host_dir)
        return ngroup_pillar
    if not os.path.isdir(host_dir):
        log.error('file_tree: %s exists, but is not a directory', host_dir)
        return ngroup_pillar
    host_pillar = _construct_pillar(host_dir, follow_dir_links, keep_newline, render_default, renderer_blacklist, renderer_whitelist, template)
    return salt.utils.dictupdate.merge(ngroup_pillar, host_pillar, strategy='recurse')