"""An Alfred 3+ version of :class:`~workflow.Workflow`.

:class:`~workflow.Workflow3` supports new features, such as
setting :ref:`workflow-variables` and
:class:`the more advanced modifiers <Modifier>` supported by Alfred 3+.

In order for the feedback mechanism to work correctly, it's important
to create :class:`Item3` and :class:`Modifier` objects via the
:meth:`Workflow3.add_item()` and :meth:`Item3.add_modifier()` methods
respectively. If you instantiate :class:`Item3` or :class:`Modifier`
objects directly, the current :class:`Workflow3` object won't be aware
of them, and they won't be sent to Alfred when you call
:meth:`Workflow3.send_feedback()`.

"""
from __future__ import print_function, unicode_literals, absolute_import
import json
import os
import sys
from .workflow import ICON_WARNING, Workflow

class Variables(dict):
    """Workflow variables for Run Script actions.

    .. versionadded: 1.26

    This class allows you to set workflow variables from
    Run Script actions.

    It is a subclass of :class:`dict`.

    >>> v = Variables(username='deanishe', password='hunter2')
    >>> v.arg = u'output value'
    >>> print(v)

    See :ref:`variables-run-script` in the User Guide for more
    information.

    Args:
        arg (unicode or list, optional): Main output/``{query}``.
        **variables: Workflow variables to set.

    In Alfred 4.1+ and Alfred-Workflow 1.40+, ``arg`` may also be a
    :class:`list` or :class:`tuple`.

    Attributes:
        arg (unicode or list): Output value (``{query}``).
            In Alfred 4.1+ and Alfred-Workflow 1.40+, ``arg`` may also be a
            :class:`list` or :class:`tuple`.
        config (dict): Configuration for downstream workflow element.

    """

    def __init__(self, arg=None, **variables):
        if False:
            i = 10
            return i + 15
        'Create a new `Variables` object.'
        self.arg = arg
        self.config = {}
        super(Variables, self).__init__(**variables)

    @property
    def obj(self):
        if False:
            print('Hello World!')
        '``alfredworkflow`` :class:`dict`.'
        o = {}
        if self:
            d2 = {}
            for (k, v) in self.items():
                d2[k] = v
            o['variables'] = d2
        if self.config:
            o['config'] = self.config
        if self.arg is not None:
            o['arg'] = self.arg
        return {'alfredworkflow': o}

    def __unicode__(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert to ``alfredworkflow`` JSON object.\n\n        Returns:\n            unicode: ``alfredworkflow`` JSON object\n\n        '
        if not self and (not self.config):
            if not self.arg:
                return u''
            if isinstance(self.arg, unicode):
                return self.arg
        return json.dumps(self.obj)

    def __str__(self):
        if False:
            print('Hello World!')
        'Convert to ``alfredworkflow`` JSON object.\n\n        Returns:\n            str: UTF-8 encoded ``alfredworkflow`` JSON object\n\n        '
        return unicode(self).encode('utf-8')

class Modifier(object):
    """Modify :class:`Item3` arg/icon/variables when modifier key is pressed.

    Don't use this class directly (as it won't be associated with any
    :class:`Item3`), but rather use :meth:`Item3.add_modifier()`
    to add modifiers to results.

    >>> it = wf.add_item('Title', 'Subtitle', valid=True)
    >>> it.setvar('name', 'default')
    >>> m = it.add_modifier('cmd')
    >>> m.setvar('name', 'alternate')

    See :ref:`workflow-variables` in the User Guide for more information
    and :ref:`example usage <example-variables>`.

    Args:
        key (unicode): Modifier key, e.g. ``"cmd"``, ``"alt"`` etc.
        subtitle (unicode, optional): Override default subtitle.
        arg (unicode, optional): Argument to pass for this modifier.
        valid (bool, optional): Override item's validity.
        icon (unicode, optional): Filepath/UTI of icon to use
        icontype (unicode, optional): Type of icon. See
            :meth:`Workflow.add_item() <workflow.Workflow.add_item>`
            for valid values.

    Attributes:
        arg (unicode): Arg to pass to following action.
        config (dict): Configuration for a downstream element, such as
            a File Filter.
        icon (unicode): Filepath/UTI of icon.
        icontype (unicode): Type of icon. See
            :meth:`Workflow.add_item() <workflow.Workflow.add_item>`
            for valid values.
        key (unicode): Modifier key (see above).
        subtitle (unicode): Override item subtitle.
        valid (bool): Override item validity.
        variables (dict): Workflow variables set by this modifier.

    """

    def __init__(self, key, subtitle=None, arg=None, valid=None, icon=None, icontype=None):
        if False:
            print('Hello World!')
        'Create a new :class:`Modifier`.\n\n        Don\'t use this class directly (as it won\'t be associated with any\n        :class:`Item3`), but rather use :meth:`Item3.add_modifier()`\n        to add modifiers to results.\n\n        Args:\n            key (unicode): Modifier key, e.g. ``"cmd"``, ``"alt"`` etc.\n            subtitle (unicode, optional): Override default subtitle.\n            arg (unicode, optional): Argument to pass for this modifier.\n            valid (bool, optional): Override item\'s validity.\n            icon (unicode, optional): Filepath/UTI of icon to use\n            icontype (unicode, optional): Type of icon. See\n                :meth:`Workflow.add_item() <workflow.Workflow.add_item>`\n                for valid values.\n\n        '
        self.key = key
        self.subtitle = subtitle
        self.arg = arg
        self.valid = valid
        self.icon = icon
        self.icontype = icontype
        self.config = {}
        self.variables = {}

    def setvar(self, name, value):
        if False:
            i = 10
            return i + 15
        'Set a workflow variable for this Item.\n\n        Args:\n            name (unicode): Name of variable.\n            value (unicode): Value of variable.\n\n        '
        self.variables[name] = value

    def getvar(self, name, default=None):
        if False:
            print('Hello World!')
        'Return value of workflow variable for ``name`` or ``default``.\n\n        Args:\n            name (unicode): Variable name.\n            default (None, optional): Value to return if variable is unset.\n\n        Returns:\n            unicode or ``default``: Value of variable if set or ``default``.\n\n        '
        return self.variables.get(name, default)

    @property
    def obj(self):
        if False:
            return 10
        'Modifier formatted for JSON serialization for Alfred 3.\n\n        Returns:\n            dict: Modifier for serializing to JSON.\n\n        '
        o = {}
        if self.subtitle is not None:
            o['subtitle'] = self.subtitle
        if self.arg is not None:
            o['arg'] = self.arg
        if self.valid is not None:
            o['valid'] = self.valid
        if self.variables:
            o['variables'] = self.variables
        if self.config:
            o['config'] = self.config
        icon = self._icon()
        if icon:
            o['icon'] = icon
        return o

    def _icon(self):
        if False:
            for i in range(10):
                print('nop')
        'Return `icon` object for item.\n\n        Returns:\n            dict: Mapping for item `icon` (may be empty).\n\n        '
        icon = {}
        if self.icon is not None:
            icon['path'] = self.icon
        if self.icontype is not None:
            icon['type'] = self.icontype
        return icon

class Item3(object):
    """Represents a feedback item for Alfred 3+.

    Generates Alfred-compliant JSON for a single item.

    Don't use this class directly (as it then won't be associated with
    any :class:`Workflow3 <workflow.Workflow3>` object), but rather use
    :meth:`Workflow3.add_item() <workflow.Workflow3.add_item>`.
    See :meth:`~workflow.Workflow3.add_item` for details of arguments.

    """

    def __init__(self, title, subtitle='', arg=None, autocomplete=None, match=None, valid=False, uid=None, icon=None, icontype=None, type=None, largetext=None, copytext=None, quicklookurl=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a new :class:`Item3` object.\n\n        Use same arguments as for\n        :class:`Workflow.Item <workflow.Workflow.Item>`.\n\n        Argument ``subtitle_modifiers`` is not supported.\n\n        '
        self.title = title
        self.subtitle = subtitle
        self.arg = arg
        self.autocomplete = autocomplete
        self.match = match
        self.valid = valid
        self.uid = uid
        self.icon = icon
        self.icontype = icontype
        self.type = type
        self.quicklookurl = quicklookurl
        self.largetext = largetext
        self.copytext = copytext
        self.modifiers = {}
        self.config = {}
        self.variables = {}

    def setvar(self, name, value):
        if False:
            while True:
                i = 10
        'Set a workflow variable for this Item.\n\n        Args:\n            name (unicode): Name of variable.\n            value (unicode): Value of variable.\n\n        '
        self.variables[name] = value

    def getvar(self, name, default=None):
        if False:
            i = 10
            return i + 15
        'Return value of workflow variable for ``name`` or ``default``.\n\n        Args:\n            name (unicode): Variable name.\n            default (None, optional): Value to return if variable is unset.\n\n        Returns:\n            unicode or ``default``: Value of variable if set or ``default``.\n\n        '
        return self.variables.get(name, default)

    def add_modifier(self, key, subtitle=None, arg=None, valid=None, icon=None, icontype=None):
        if False:
            print('Hello World!')
        'Add alternative values for a modifier key.\n\n        Args:\n            key (unicode): Modifier key, e.g. ``"cmd"`` or ``"alt"``\n            subtitle (unicode, optional): Override item subtitle.\n            arg (unicode, optional): Input for following action.\n            valid (bool, optional): Override item validity.\n            icon (unicode, optional): Filepath/UTI of icon.\n            icontype (unicode, optional): Type of icon.  See\n                :meth:`Workflow.add_item() <workflow.Workflow.add_item>`\n                for valid values.\n\n        In Alfred 4.1+ and Alfred-Workflow 1.40+, ``arg`` may also be a\n        :class:`list` or :class:`tuple`.\n\n        Returns:\n            Modifier: Configured :class:`Modifier`.\n\n        '
        mod = Modifier(key, subtitle, arg, valid, icon, icontype)
        mod.variables.update(self.variables)
        self.modifiers[key] = mod
        return mod

    @property
    def obj(self):
        if False:
            for i in range(10):
                print('nop')
        'Item formatted for JSON serialization.\n\n        Returns:\n            dict: Data suitable for Alfred 3 feedback.\n\n        '
        o = {'title': self.title, 'subtitle': self.subtitle, 'valid': self.valid}
        if self.arg is not None:
            o['arg'] = self.arg
        if self.autocomplete is not None:
            o['autocomplete'] = self.autocomplete
        if self.match is not None:
            o['match'] = self.match
        if self.uid is not None:
            o['uid'] = self.uid
        if self.type is not None:
            o['type'] = self.type
        if self.quicklookurl is not None:
            o['quicklookurl'] = self.quicklookurl
        if self.variables:
            o['variables'] = self.variables
        if self.config:
            o['config'] = self.config
        text = self._text()
        if text:
            o['text'] = text
        icon = self._icon()
        if icon:
            o['icon'] = icon
        mods = self._modifiers()
        if mods:
            o['mods'] = mods
        return o

    def _icon(self):
        if False:
            while True:
                i = 10
        'Return `icon` object for item.\n\n        Returns:\n            dict: Mapping for item `icon` (may be empty).\n\n        '
        icon = {}
        if self.icon is not None:
            icon['path'] = self.icon
        if self.icontype is not None:
            icon['type'] = self.icontype
        return icon

    def _text(self):
        if False:
            return 10
        'Return `largetext` and `copytext` object for item.\n\n        Returns:\n            dict: `text` mapping (may be empty)\n\n        '
        text = {}
        if self.largetext is not None:
            text['largetype'] = self.largetext
        if self.copytext is not None:
            text['copy'] = self.copytext
        return text

    def _modifiers(self):
        if False:
            for i in range(10):
                print('nop')
        'Build `mods` dictionary for JSON feedback.\n\n        Returns:\n            dict: Modifier mapping or `None`.\n\n        '
        if self.modifiers:
            mods = {}
            for (k, mod) in self.modifiers.items():
                mods[k] = mod.obj
            return mods
        return None

class Workflow3(Workflow):
    """Workflow class that generates Alfred 3+ feedback.

    It is a subclass of :class:`~workflow.Workflow` and most of its
    methods are documented there.

    Attributes:
        item_class (class): Class used to generate feedback items.
        variables (dict): Top level workflow variables.

    """
    item_class = Item3

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Create a new :class:`Workflow3` object.\n\n        See :class:`~workflow.Workflow` for documentation.\n\n        '
        Workflow.__init__(self, **kwargs)
        self.variables = {}
        self._rerun = 0
        self._session_id = os.getenv('_WF_SESSION_ID') or None
        if self._session_id:
            self.setvar('_WF_SESSION_ID', self._session_id)

    @property
    def _default_cachedir(self):
        if False:
            for i in range(10):
                print('nop')
        "Alfred 4's default cache directory."
        return os.path.join(os.path.expanduser('~/Library/Caches/com.runningwithcrayons.Alfred/Workflow Data/'), self.bundleid)

    @property
    def _default_datadir(self):
        if False:
            for i in range(10):
                print('nop')
        "Alfred 4's default data directory."
        return os.path.join(os.path.expanduser('~/Library/Application Support/Alfred/Workflow Data/'), self.bundleid)

    @property
    def rerun(self):
        if False:
            while True:
                i = 10
        'How often (in seconds) Alfred should re-run the Script Filter.'
        return self._rerun

    @rerun.setter
    def rerun(self, seconds):
        if False:
            for i in range(10):
                print('nop')
        'Interval at which Alfred should re-run the Script Filter.\n\n        Args:\n            seconds (int): Interval between runs.\n        '
        self._rerun = seconds

    @property
    def session_id(self):
        if False:
            print('Hello World!')
        'A unique session ID every time the user uses the workflow.\n\n        .. versionadded:: 1.25\n\n        The session ID persists while the user is using this workflow.\n        It expires when the user runs a different workflow or closes\n        Alfred.\n\n        '
        if not self._session_id:
            from uuid import uuid4
            self._session_id = uuid4().hex
            self.setvar('_WF_SESSION_ID', self._session_id)
        return self._session_id

    def setvar(self, name, value, persist=False):
        if False:
            for i in range(10):
                print('nop')
        'Set a "global" workflow variable.\n\n        .. versionchanged:: 1.33\n\n        These variables are always passed to downstream workflow objects.\n\n        If you have set :attr:`rerun`, these variables are also passed\n        back to the script when Alfred runs it again.\n\n        Args:\n            name (unicode): Name of variable.\n            value (unicode): Value of variable.\n            persist (bool, optional): Also save variable to ``info.plist``?\n\n        '
        self.variables[name] = value
        if persist:
            from .util import set_config
            set_config(name, value, self.bundleid)
            self.logger.debug('saved variable %r with value %r to info.plist', name, value)

    def getvar(self, name, default=None):
        if False:
            i = 10
            return i + 15
        'Return value of workflow variable for ``name`` or ``default``.\n\n        Args:\n            name (unicode): Variable name.\n            default (None, optional): Value to return if variable is unset.\n\n        Returns:\n            unicode or ``default``: Value of variable if set or ``default``.\n\n        '
        return self.variables.get(name, default)

    def add_item(self, title, subtitle='', arg=None, autocomplete=None, valid=False, uid=None, icon=None, icontype=None, type=None, largetext=None, copytext=None, quicklookurl=None, match=None):
        if False:
            print('Hello World!')
        'Add an item to be output to Alfred.\n\n        Args:\n            match (unicode, optional): If you have "Alfred filters results"\n                turned on for your Script Filter, Alfred (version 3.5 and\n                above) will filter against this field, not ``title``.\n\n        In Alfred 4.1+ and Alfred-Workflow 1.40+, ``arg`` may also be a\n        :class:`list` or :class:`tuple`.\n\n        See :meth:`Workflow.add_item() <workflow.Workflow.add_item>` for\n        the main documentation and other parameters.\n\n        The key difference is that this method does not support the\n        ``modifier_subtitles`` argument. Use the :meth:`~Item3.add_modifier()`\n        method instead on the returned item instead.\n\n        Returns:\n            Item3: Alfred feedback item.\n\n        '
        item = self.item_class(title, subtitle, arg, autocomplete, match, valid, uid, icon, icontype, type, largetext, copytext, quicklookurl)
        item.variables.update(self.variables)
        self._items.append(item)
        return item

    @property
    def _session_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        'Filename prefix for current session.'
        return '_wfsess-{0}-'.format(self.session_id)

    def _mk_session_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        'New cache name/key based on session ID.'
        return self._session_prefix + name

    def cache_data(self, name, data, session=False):
        if False:
            print('Hello World!')
        'Cache API with session-scoped expiry.\n\n        .. versionadded:: 1.25\n\n        Args:\n            name (str): Cache key\n            data (object): Data to cache\n            session (bool, optional): Whether to scope the cache\n                to the current session.\n\n        ``name`` and ``data`` are the same as for the\n        :meth:`~workflow.Workflow.cache_data` method on\n        :class:`~workflow.Workflow`.\n\n        If ``session`` is ``True``, then ``name`` is prefixed\n        with :attr:`session_id`.\n\n        '
        if session:
            name = self._mk_session_name(name)
        return super(Workflow3, self).cache_data(name, data)

    def cached_data(self, name, data_func=None, max_age=60, session=False):
        if False:
            while True:
                i = 10
        "Cache API with session-scoped expiry.\n\n        .. versionadded:: 1.25\n\n        Args:\n            name (str): Cache key\n            data_func (callable): Callable that returns fresh data. It\n                is called if the cache has expired or doesn't exist.\n            max_age (int): Maximum allowable age of cache in seconds.\n            session (bool, optional): Whether to scope the cache\n                to the current session.\n\n        ``name``, ``data_func`` and ``max_age`` are the same as for the\n        :meth:`~workflow.Workflow.cached_data` method on\n        :class:`~workflow.Workflow`.\n\n        If ``session`` is ``True``, then ``name`` is prefixed\n        with :attr:`session_id`.\n\n        "
        if session:
            name = self._mk_session_name(name)
        return super(Workflow3, self).cached_data(name, data_func, max_age)

    def clear_session_cache(self, current=False):
        if False:
            print('Hello World!')
        "Remove session data from the cache.\n\n        .. versionadded:: 1.25\n        .. versionchanged:: 1.27\n\n        By default, data belonging to the current session won't be\n        deleted. Set ``current=True`` to also clear current session.\n\n        Args:\n            current (bool, optional): If ``True``, also remove data for\n                current session.\n\n        "

        def _is_session_file(filename):
            if False:
                print('Hello World!')
            if current:
                return filename.startswith('_wfsess-')
            return filename.startswith('_wfsess-') and (not filename.startswith(self._session_prefix))
        self.clear_cache(_is_session_file)

    @property
    def obj(self):
        if False:
            print('Hello World!')
        'Feedback formatted for JSON serialization.\n\n        Returns:\n            dict: Data suitable for Alfred 3 feedback.\n\n        '
        items = []
        for item in self._items:
            items.append(item.obj)
        o = {'items': items}
        if self.variables:
            o['variables'] = self.variables
        if self.rerun:
            o['rerun'] = self.rerun
        return o

    def warn_empty(self, title, subtitle=u'', icon=None):
        if False:
            return 10
        'Add a warning to feedback if there are no items.\n\n        .. versionadded:: 1.31\n\n        Add a "warning" item to Alfred feedback if no other items\n        have been added. This is a handy shortcut to prevent Alfred\n        from showing its fallback searches, which is does if no\n        items are returned.\n\n        Args:\n            title (unicode): Title of feedback item.\n            subtitle (unicode, optional): Subtitle of feedback item.\n            icon (str, optional): Icon for feedback item. If not\n                specified, ``ICON_WARNING`` is used.\n\n        Returns:\n            Item3: Newly-created item.\n\n        '
        if len(self._items):
            return
        icon = icon or ICON_WARNING
        return self.add_item(title, subtitle, icon=icon)

    def send_feedback(self):
        if False:
            print('Hello World!')
        'Print stored items to console/Alfred as JSON.'
        if self.debugging:
            json.dump(self.obj, sys.stdout, indent=2, separators=(',', ': '))
        else:
            json.dump(self.obj, sys.stdout)
        sys.stdout.flush()