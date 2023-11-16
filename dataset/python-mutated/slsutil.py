"""
Utility functions for use with or in SLS files
"""
import os
import posixpath
import textwrap
import salt.exceptions
import salt.loader
import salt.template
import salt.utils.args
import salt.utils.dictupdate
import salt.utils.path
CONTEXT_BASE = 'slsutil'

def update(dest, upd, recursive_update=True, merge_lists=False):
    if False:
        print('Hello World!')
    "\n    Merge ``upd`` recursively into ``dest``\n\n    If ``merge_lists=True``, will aggregate list object types instead of\n    replacing. This behavior is only activated when ``recursive_update=True``.\n\n    CLI Example:\n\n    .. code-block:: shell\n\n        salt '*' slsutil.update '{foo: Foo}' '{bar: Bar}'\n\n    "
    return salt.utils.dictupdate.update(dest, upd, recursive_update, merge_lists)

def merge(obj_a, obj_b, strategy='smart', renderer='yaml', merge_lists=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Merge a data structure into another by choosing a merge strategy\n\n    Strategies:\n\n    * aggregate\n    * list\n    * overwrite\n    * recurse\n    * smart\n\n    CLI Example:\n\n    .. code-block:: shell\n\n        salt '*' slsutil.merge '{foo: Foo}' '{bar: Bar}'\n    "
    return salt.utils.dictupdate.merge(obj_a, obj_b, strategy, renderer, merge_lists)

def merge_all(lst, strategy='smart', renderer='yaml', merge_lists=False):
    if False:
        return 10
    "\n    .. versionadded:: 2019.2.0\n\n    Merge a list of objects into each other in order\n\n    :type lst: Iterable\n    :param lst: List of objects to be merged.\n\n    :type strategy: String\n    :param strategy: Merge strategy. See utils.dictupdate.\n\n    :type renderer: String\n    :param renderer:\n        Renderer type. Used to determine strategy when strategy is 'smart'.\n\n    :type merge_lists: Bool\n    :param merge_lists: Defines whether to merge embedded object lists.\n\n    CLI Example:\n\n    .. code-block:: shell\n\n        $ salt-call --output=txt slsutil.merge_all '[{foo: Foo}, {foo: Bar}]'\n        local: {u'foo': u'Bar'}\n    "
    ret = {}
    for obj in lst:
        ret = salt.utils.dictupdate.merge(ret, obj, strategy, renderer, merge_lists)
    return ret

def renderer(path=None, string=None, default_renderer='jinja|yaml', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Parse a string or file through Salt\'s renderer system\n\n    .. versionchanged:: 2018.3.0\n       Add support for Salt fileserver URIs.\n\n    This is an open-ended function and can be used for a variety of tasks. It\n    makes use of Salt\'s "renderer pipes" system to run a string or file through\n    a pipe of any of the loaded renderer modules.\n\n    :param path: The path to a file on Salt\'s fileserver (any URIs supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`) or on the local file\n        system.\n    :param string: An inline string to be used as the file to send through the\n        renderer system. Note, not all renderer modules can work with strings;\n        the \'py\' renderer requires a file, for example.\n    :param default_renderer: The renderer pipe to send the file through; this\n        is overridden by a "she-bang" at the top of the file.\n    :param kwargs: Keyword args to pass to Salt\'s compile_template() function.\n\n    Keep in mind the goal of each renderer when choosing a render-pipe; for\n    example, the Jinja renderer processes a text file and produces a string,\n    however the YAML renderer processes a text file and produces a data\n    structure.\n\n    One possible use is to allow writing "map files", as are commonly seen in\n    Salt formulas, but without tying the renderer of the map file to the\n    renderer used in the other sls files. In other words, a map file could use\n    the Python renderer and still be included and used by an sls file that uses\n    the default \'jinja|yaml\' renderer.\n\n    For example, the two following map files produce identical results but one\n    is written using the normal \'jinja|yaml\' and the other is using \'py\':\n\n    .. code-block:: jinja\n\n        #!jinja|yaml\n        {% set apache = salt.grains.filter_by({\n            ...normal jinja map file here...\n        }, merge=salt.pillar.get(\'apache:lookup\')) %}\n        {{ apache | yaml() }}\n\n    .. code-block:: python\n\n        #!py\n        def run():\n            apache = __salt__.grains.filter_by({\n                ...normal map here but as a python dict...\n            }, merge=__salt__.pillar.get(\'apache:lookup\'))\n            return apache\n\n    Regardless of which of the above map files is used, it can be accessed from\n    any other sls file by calling this function. The following is a usage\n    example in Jinja:\n\n    .. code-block:: jinja\n\n        {% set apache = salt.slsutil.renderer(\'map.sls\') %}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' slsutil.renderer salt://path/to/file\n        salt \'*\' slsutil.renderer /path/to/file\n        salt \'*\' slsutil.renderer /path/to/file.jinja default_renderer=\'jinja\'\n        salt \'*\' slsutil.renderer /path/to/file.sls default_renderer=\'jinja|yaml\'\n        salt \'*\' slsutil.renderer string=\'Inline template! {{ saltenv }}\'\n        salt \'*\' slsutil.renderer string=\'Hello, {{ name }}.\' name=\'world\'\n    '
    if not path and (not string):
        raise salt.exceptions.SaltInvocationError('Must pass either path or string')
    renderers = salt.loader.render(__opts__, __salt__)
    if path:
        path_or_string = __salt__['cp.get_url'](path, saltenv=kwargs.get('saltenv', 'base'))
    elif string:
        path_or_string = ':string:'
        kwargs['input_data'] = string
    ret = salt.template.compile_template(path_or_string, renderers, default_renderer, __opts__['renderer_blacklist'], __opts__['renderer_whitelist'], **kwargs)
    return ret.read() if __utils__['stringio.is_readable'](ret) else ret

def _get_serialize_fn(serializer, fn_name):
    if False:
        return 10
    serializers = salt.loader.serializers(__opts__)
    fns = getattr(serializers, serializer, None)
    fn = getattr(fns, fn_name, None)
    if not fns:
        raise salt.exceptions.CommandExecutionError("Serializer '{}' not found.".format(serializer))
    if not fn:
        raise salt.exceptions.CommandExecutionError("Serializer '{}' does not implement {}.".format(serializer, fn_name))
    return fn

def serialize(serializer, obj, **mod_kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Serialize a Python object using one of the available\n    :ref:`all-salt.serializers`.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' --no-parse=obj slsutil.serialize \'json\' obj="{\'foo\': \'Foo!\'}\n\n    Jinja Example:\n\n    .. code-block:: jinja\n\n        {% set json_string = salt.slsutil.serialize(\'json\',\n            {\'foo\': \'Foo!\'}) %}\n    '
    kwargs = salt.utils.args.clean_kwargs(**mod_kwargs)
    return _get_serialize_fn(serializer, 'serialize')(obj, **kwargs)

def deserialize(serializer, stream_or_string, **mod_kwargs):
    if False:
        print('Hello World!')
    '\n    Deserialize a Python object using one of the available\n    :ref:`all-salt.serializers`.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' slsutil.deserialize \'json\' \'{"foo": "Foo!"}\'\n        salt \'*\' --no-parse=stream_or_string slsutil.deserialize \'json\' \\\n            stream_or_string=\'{"foo": "Foo!"}\'\n\n    Jinja Example:\n\n    .. code-block:: jinja\n\n        {% set python_object = salt.slsutil.deserialize(\'json\',\n            \'{"foo": "Foo!"}\') %}\n    '
    kwargs = salt.utils.args.clean_kwargs(**mod_kwargs)
    return _get_serialize_fn(serializer, 'deserialize')(stream_or_string, **kwargs)

def banner(width=72, commentchar='#', borderchar='#', blockstart=None, blockend=None, title=None, text=None, newline=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a standardized comment block to include in a templated file.\n\n    A common technique in configuration management is to include a comment\n    block in managed files, warning users not to modify the file. This\n    function simplifies and standardizes those comment blocks.\n\n    :param width: The width, in characters, of the banner. Default is 72.\n    :param commentchar: The character to be used in the starting position of\n        each line. This value should be set to a valid line comment character\n        for the syntax of the file in which the banner is being inserted.\n        Multiple character sequences, like '//' are supported.\n        If the file's syntax does not support line comments (such as XML),\n        use the ``blockstart`` and ``blockend`` options.\n    :param borderchar: The character to use in the top and bottom border of\n        the comment box. Must be a single character.\n    :param blockstart: The character sequence to use at the beginning of a\n        block comment. Should be used in conjunction with ``blockend``\n    :param blockend: The character sequence to use at the end of a\n        block comment. Should be used in conjunction with ``blockstart``\n    :param title: The first field of the comment block. This field appears\n        centered at the top of the box.\n    :param text: The second filed of the comment block. This field appears\n        left-justified at the bottom of the box.\n    :param newline: Boolean value to indicate whether the comment block should\n        end with a newline. Default is ``False``.\n\n    **Example 1 - the default banner:**\n\n    .. code-block:: jinja\n\n        {{ salt['slsutil.banner']() }}\n\n    .. code-block:: none\n\n        ########################################################################\n        #                                                                      #\n        #              THIS FILE IS MANAGED BY SALT - DO NOT EDIT              #\n        #                                                                      #\n        # The contents of this file are managed by Salt. Any changes to this   #\n        # file may be overwritten automatically and without warning.           #\n        ########################################################################\n\n    **Example 2 - a Javadoc-style banner:**\n\n    .. code-block:: jinja\n\n        {{ salt['slsutil.banner'](commentchar=' *', borderchar='*', blockstart='/**', blockend=' */') }}\n\n    .. code-block:: none\n\n        /**\n         ***********************************************************************\n         *                                                                     *\n         *              THIS FILE IS MANAGED BY SALT - DO NOT EDIT             *\n         *                                                                     *\n         * The contents of this file are managed by Salt. Any changes to this  *\n         * file may be overwritten automatically and without warning.          *\n         ***********************************************************************\n         */\n\n    **Example 3 - custom text:**\n\n    .. code-block:: jinja\n\n        {{ set copyright='This file may not be copied or distributed without permission of VMware, Inc.' }}\n        {{ salt['slsutil.banner'](title='Copyright 2019 VMware, Inc.', text=copyright, width=60) }}\n\n    .. code-block:: none\n\n        ############################################################\n        #                                                          #\n        #              Copyright 2019 VMware, Inc.                 #\n        #                                                          #\n        # This file may not be copied or distributed without       #\n        # permission of VMware, Inc.                               #\n        ############################################################\n\n    "
    if title is None:
        title = 'THIS FILE IS MANAGED BY SALT - DO NOT EDIT'
    if text is None:
        text = 'The contents of this file are managed by Salt. Any changes to this file may be overwritten automatically and without warning.'
    ledge = commentchar.rstrip()
    redge = commentchar.strip()
    lgutter = ledge + ' '
    rgutter = ' ' + redge
    textwidth = width - len(lgutter) - len(rgutter)
    if textwidth <= 0:
        raise salt.exceptions.ArgumentValueError('Width is too small to render banner')
    border_line = commentchar + borderchar[:1] * (width - len(ledge) - len(redge)) + redge
    spacer_line = commentchar + ' ' * (width - len(commentchar) * 2) + commentchar
    wrapper = textwrap.TextWrapper(width=textwidth)
    block = list()
    if blockstart is not None:
        block.append(blockstart)
    block.append(border_line)
    block.append(spacer_line)
    for line in wrapper.wrap(title):
        block.append(lgutter + line.center(textwidth) + rgutter)
    block.append(spacer_line)
    for line in wrapper.wrap(text):
        block.append(lgutter + line + ' ' * (textwidth - len(line)) + rgutter)
    block.append(border_line)
    if blockend is not None:
        block.append(blockend)
    result = os.linesep.join(block)
    if newline:
        return result + os.linesep
    return result

def boolstr(value, true='true', false='false'):
    if False:
        return 10
    "\n    Convert a boolean value into a string. This function is\n    intended to be used from within file templates to provide\n    an easy way to take boolean values stored in Pillars or\n    Grains, and write them out in the appropriate syntax for\n    a particular file template.\n\n    :param value: The boolean value to be converted\n    :param true: The value to return if ``value`` is ``True``\n    :param false: The value to return if ``value`` is ``False``\n\n    In this example, a pillar named ``smtp:encrypted`` stores a boolean\n    value, but the template that uses that value needs ``yes`` or ``no``\n    to be written, based on the boolean value.\n\n    *Note: this is written on two lines for clarity. The same result\n    could be achieved in one line.*\n\n    .. code-block:: jinja\n\n        {% set encrypted = salt[pillar.get]('smtp:encrypted', false) %}\n        use_tls: {{ salt['slsutil.boolstr'](encrypted, 'yes', 'no') }}\n\n    Result (assuming the value is ``True``):\n\n    .. code-block:: none\n\n        use_tls: yes\n\n    "
    if value:
        return true
    return false

def _set_context(keys, function, fun_args=None, fun_kwargs=None, force=False):
    if False:
        return 10
    '\n    Convenience function to set a value in the ``__context__`` dictionary.\n\n    .. versionadded:: 3004\n\n    :param keys: The list of keys specifying the dictionary path to set. This\n                 list can be of arbitrary length and the path will be created\n                 in the dictionary if it does not exist.\n\n    :param function: A python function to be called if the specified path does\n                     not exist, if the force parameter is ``True``.\n\n    :param fun_args: A list of positional arguments to the function.\n\n    :param fun_kwargs: A dictionary of keyword arguments to the function.\n\n    :param force: If ``True``, force the ```__context__`` path to be updated.\n                  Otherwise, only create it if it does not exist.\n    '
    target = __context__
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]
    if force or keys[-1] not in target:
        if not fun_args:
            fun_args = []
        if not fun_kwargs:
            fun_kwargs = {}
        target[keys[-1]] = function(*fun_args, *fun_kwargs)

def file_exists(path, saltenv='base'):
    if False:
        print('Hello World!')
    "\n    Return ``True`` if a file exists in the state tree, ``False`` otherwise.\n\n    .. versionadded:: 3004\n\n    :param str path: The fully qualified path to a file in the state tree.\n    :param str saltenv: The fileserver environment to search. Default: ``base``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' slsutil.file_exists nginx/defaults.yaml\n    "
    _set_context([CONTEXT_BASE, saltenv, 'file_list'], __salt__['cp.list_master'], [saltenv])
    return path in __context__[CONTEXT_BASE][saltenv]['file_list']

def dir_exists(path, saltenv='base'):
    if False:
        i = 10
        return i + 15
    "\n    Return ``True`` if a directory exists in the state tree, ``False`` otherwise.\n\n    :param str path: The fully qualified path to a directory in the state tree.\n    :param str saltenv: The fileserver environment to search. Default: ``base``\n\n    .. versionadded:: 3004\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' slsutil.dir_exists nginx/files\n    "
    _set_context([CONTEXT_BASE, saltenv, 'dir_list'], __salt__['cp.list_master_dirs'], [saltenv])
    return path in __context__[CONTEXT_BASE][saltenv]['dir_list']

def path_exists(path, saltenv='base'):
    if False:
        while True:
            i = 10
    "\n    Return ``True`` if a path exists in the state tree, ``False`` otherwise. The path\n    could refer to a file or directory.\n\n    .. versionadded:: 3004\n\n    :param str path: The fully qualified path to a file or directory in the state tree.\n    :param str saltenv: The fileserver environment to search. Default: ``base``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' slsutil.path_exists nginx/defaults.yaml\n    "
    return file_exists(path, saltenv) or dir_exists(path, saltenv)

def findup(startpath, filenames, saltenv='base'):
    if False:
        print('Hello World!')
    '\n    Find the first path matching a filename or list of filenames in a specified\n    directory or the nearest ancestor directory. Returns the full path to the\n    first file found.\n\n    .. versionadded:: 3004\n\n    :param str startpath: The fileserver path from which to begin the search.\n        An empty string refers to the state tree root.\n    :param filenames: A filename or list of filenames to search for. Searching for\n        directory names is also supported.\n    :param str saltenv: The fileserver environment to search. Default: ``base``\n\n    Example: return the path to ``defaults.yaml``, walking up the tree from the\n    state file currently being processed.\n\n    .. code-block:: jinja\n\n        {{ salt["slsutil.findup"](tplfile, "defaults.yaml") }}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' slsutil.findup formulas/shared/nginx map.jinja\n    '
    if startpath:
        startpath = posixpath.normpath(startpath)
    if startpath and (not path_exists(startpath, saltenv)):
        raise salt.exceptions.SaltInvocationError('Starting path not found in the state tree: {}'.format(startpath))
    if isinstance(filenames, str):
        filenames = [filenames]
    if not isinstance(filenames, list):
        raise salt.exceptions.SaltInvocationError('Filenames argument must be a string or list of strings')
    while True:
        for filename in filenames:
            fullname = salt.utils.path.join(startpath or '', filename, use_posixpath=True)
            if path_exists(fullname, saltenv):
                return fullname
        if not startpath:
            raise salt.exceptions.CommandExecutionError('File pattern(s) not found in path ancestry')
        startpath = os.path.dirname(startpath)