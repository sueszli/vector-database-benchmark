"""Top-level display functions for displaying object in different formats."""
from binascii import b2a_hex
import os
import sys
import warnings
__all__ = ['display', 'clear_output', 'publish_display_data', 'update_display', 'DisplayHandle']

def _merge(d1, d2):
    if False:
        return 10
    'Like update, but merges sub-dicts instead of clobbering at the top level.\n\n    Updates d1 in-place\n    '
    if not isinstance(d2, dict) or not isinstance(d1, dict):
        return d2
    for (key, value) in d2.items():
        d1[key] = _merge(d1.get(key), value)
    return d1

class _Sentinel:

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<deprecated>'
_sentinel = _Sentinel()

def publish_display_data(data, metadata=None, source=_sentinel, *, transient=None, **kwargs):
    if False:
        return 10
    "Publish data and metadata to all frontends.\n\n    See the ``display_data`` message in the messaging documentation for\n    more details about this message type.\n\n    Keys of data and metadata can be any mime-type.\n\n    Parameters\n    ----------\n    data : dict\n        A dictionary having keys that are valid MIME types (like\n        'text/plain' or 'image/svg+xml') and values that are the data for\n        that MIME type. The data itself must be a JSON'able data\n        structure. Minimally all data should have the 'text/plain' data,\n        which can be displayed by all frontends. If more than the plain\n        text is given, it is up to the frontend to decide which\n        representation to use.\n    metadata : dict\n        A dictionary for metadata related to the data. This can contain\n        arbitrary key, value pairs that frontends can use to interpret\n        the data. mime-type keys matching those in data can be used\n        to specify metadata about particular representations.\n    source : str, deprecated\n        Unused.\n    transient : dict, keyword-only\n        A dictionary of transient data, such as display_id.\n    "
    from IPython.core.interactiveshell import InteractiveShell
    if source is not _sentinel:
        warnings.warn('The `source` parameter emit a  deprecation warning since IPython 8.0, it had no effects for a long time and will  be removed in future versions.', DeprecationWarning, stacklevel=2)
    display_pub = InteractiveShell.instance().display_pub
    if transient:
        kwargs['transient'] = transient
    display_pub.publish(data=data, metadata=metadata, **kwargs)

def _new_id():
    if False:
        i = 10
        return i + 15
    'Generate a new random text id with urandom'
    return b2a_hex(os.urandom(16)).decode('ascii')

def display(*objs, include=None, exclude=None, metadata=None, transient=None, display_id=None, raw=False, clear=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Display a Python object in all frontends.\n\n    By default all representations will be computed and sent to the frontends.\n    Frontends can decide which representation is used and how.\n\n    In terminal IPython this will be similar to using :func:`print`, for use in richer\n    frontends see Jupyter notebook examples with rich display logic.\n\n    Parameters\n    ----------\n    *objs : object\n        The Python objects to display.\n    raw : bool, optional\n        Are the objects to be displayed already mimetype-keyed dicts of raw display data,\n        or Python objects that need to be formatted before display? [default: False]\n    include : list, tuple or set, optional\n        A list of format type strings (MIME types) to include in the\n        format data dict. If this is set *only* the format types included\n        in this list will be computed.\n    exclude : list, tuple or set, optional\n        A list of format type strings (MIME types) to exclude in the format\n        data dict. If this is set all format types will be computed,\n        except for those included in this argument.\n    metadata : dict, optional\n        A dictionary of metadata to associate with the output.\n        mime-type keys in this dictionary will be associated with the individual\n        representation formats, if they exist.\n    transient : dict, optional\n        A dictionary of transient data to associate with the output.\n        Data in this dict should not be persisted to files (e.g. notebooks).\n    display_id : str, bool optional\n        Set an id for the display.\n        This id can be used for updating this display area later via update_display.\n        If given as `True`, generate a new `display_id`\n    clear : bool, optional\n        Should the output area be cleared before displaying anything? If True,\n        this will wait for additional output before clearing. [default: False]\n    **kwargs : additional keyword-args, optional\n        Additional keyword-arguments are passed through to the display publisher.\n\n    Returns\n    -------\n    handle: DisplayHandle\n        Returns a handle on updatable displays for use with :func:`update_display`,\n        if `display_id` is given. Returns :any:`None` if no `display_id` is given\n        (default).\n\n    Examples\n    --------\n    >>> class Json(object):\n    ...     def __init__(self, json):\n    ...         self.json = json\n    ...     def _repr_pretty_(self, pp, cycle):\n    ...         import json\n    ...         pp.text(json.dumps(self.json, indent=2))\n    ...     def __repr__(self):\n    ...         return str(self.json)\n    ...\n\n    >>> d = Json({1:2, 3: {4:5}})\n\n    >>> print(d)\n    {1: 2, 3: {4: 5}}\n\n    >>> display(d)\n    {\n      "1": 2,\n      "3": {\n        "4": 5\n      }\n    }\n\n    >>> def int_formatter(integer, pp, cycle):\n    ...     pp.text(\'I\'*integer)\n\n    >>> plain = get_ipython().display_formatter.formatters[\'text/plain\']\n    >>> plain.for_type(int, int_formatter)\n    <function _repr_pprint at 0x...>\n    >>> display(7-5)\n    II\n\n    >>> del plain.type_printers[int]\n    >>> display(7-5)\n    2\n\n    See Also\n    --------\n    :func:`update_display`\n\n    Notes\n    -----\n    In Python, objects can declare their textual representation using the\n    `__repr__` method. IPython expands on this idea and allows objects to declare\n    other, rich representations including:\n\n      - HTML\n      - JSON\n      - PNG\n      - JPEG\n      - SVG\n      - LaTeX\n\n    A single object can declare some or all of these representations; all are\n    handled by IPython\'s display system.\n\n    The main idea of the first approach is that you have to implement special\n    display methods when you define your class, one for each representation you\n    want to use. Here is a list of the names of the special methods and the\n    values they must return:\n\n      - `_repr_html_`: return raw HTML as a string, or a tuple (see below).\n      - `_repr_json_`: return a JSONable dict, or a tuple (see below).\n      - `_repr_jpeg_`: return raw JPEG data, or a tuple (see below).\n      - `_repr_png_`: return raw PNG data, or a tuple (see below).\n      - `_repr_svg_`: return raw SVG data as a string, or a tuple (see below).\n      - `_repr_latex_`: return LaTeX commands in a string surrounded by "$",\n                        or a tuple (see below).\n      - `_repr_mimebundle_`: return a full mimebundle containing the mapping\n                             from all mimetypes to data.\n                             Use this for any mime-type not listed above.\n\n    The above functions may also return the object\'s metadata alonside the\n    data.  If the metadata is available, the functions will return a tuple\n    containing the data and metadata, in that order.  If there is no metadata\n    available, then the functions will return the data only.\n\n    When you are directly writing your own classes, you can adapt them for\n    display in IPython by following the above approach. But in practice, you\n    often need to work with existing classes that you can\'t easily modify.\n\n    You can refer to the documentation on integrating with the display system in\n    order to register custom formatters for already existing types\n    (:ref:`integrating_rich_display`).\n\n    .. versionadded:: 5.4 display available without import\n    .. versionadded:: 6.1 display available without import\n\n    Since IPython 5.4 and 6.1 :func:`display` is automatically made available to\n    the user without import. If you are using display in a document that might\n    be used in a pure python context or with older version of IPython, use the\n    following import at the top of your file::\n\n        from IPython.display import display\n\n    '
    from IPython.core.interactiveshell import InteractiveShell
    if not InteractiveShell.initialized():
        print(*objs)
        return
    if transient is None:
        transient = {}
    if metadata is None:
        metadata = {}
    if display_id:
        if display_id is True:
            display_id = _new_id()
        transient['display_id'] = display_id
    if kwargs.get('update') and 'display_id' not in transient:
        raise TypeError('display_id required for update_display')
    if transient:
        kwargs['transient'] = transient
    if not objs and display_id:
        objs = [{}]
        raw = True
    if not raw:
        format = InteractiveShell.instance().display_formatter.format
    if clear:
        clear_output(wait=True)
    for obj in objs:
        if raw:
            publish_display_data(data=obj, metadata=metadata, **kwargs)
        else:
            (format_dict, md_dict) = format(obj, include=include, exclude=exclude)
            if not format_dict:
                continue
            if metadata:
                _merge(md_dict, metadata)
            publish_display_data(data=format_dict, metadata=md_dict, **kwargs)
    if display_id:
        return DisplayHandle(display_id)

def update_display(obj, *, display_id, **kwargs):
    if False:
        while True:
            i = 10
    'Update an existing display by id\n\n    Parameters\n    ----------\n    obj\n        The object with which to update the display\n    display_id : keyword-only\n        The id of the display to update\n\n    See Also\n    --------\n    :func:`display`\n    '
    kwargs['update'] = True
    display(obj, display_id=display_id, **kwargs)

class DisplayHandle(object):
    """A handle on an updatable display

    Call `.update(obj)` to display a new object.

    Call `.display(obj`) to add a new instance of this display,
    and update existing instances.

    See Also
    --------

        :func:`display`, :func:`update_display`

    """

    def __init__(self, display_id=None):
        if False:
            print('Hello World!')
        if display_id is None:
            display_id = _new_id()
        self.display_id = display_id

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s display_id=%s>' % (self.__class__.__name__, self.display_id)

    def display(self, obj, **kwargs):
        if False:
            return 10
        'Make a new display with my id, updating existing instances.\n\n        Parameters\n        ----------\n        obj\n            object to display\n        **kwargs\n            additional keyword arguments passed to display\n        '
        display(obj, display_id=self.display_id, **kwargs)

    def update(self, obj, **kwargs):
        if False:
            while True:
                i = 10
        'Update existing displays with my id\n\n        Parameters\n        ----------\n        obj\n            object to display\n        **kwargs\n            additional keyword arguments passed to update_display\n        '
        update_display(obj, display_id=self.display_id, **kwargs)

def clear_output(wait=False):
    if False:
        i = 10
        return i + 15
    'Clear the output of the current cell receiving output.\n\n    Parameters\n    ----------\n    wait : bool [default: false]\n        Wait to clear the output until new output is available to replace it.'
    from IPython.core.interactiveshell import InteractiveShell
    if InteractiveShell.initialized():
        InteractiveShell.instance().display_pub.clear_output(wait)
    else:
        print('\x1b[2K\r', end='')
        sys.stdout.flush()
        print('\x1b[2K\r', end='')
        sys.stderr.flush()