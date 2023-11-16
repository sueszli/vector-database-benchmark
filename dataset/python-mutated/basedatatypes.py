import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import _natural_sort_strings, _get_int_type, split_multichar, split_string_positions, display_string_positions, chomp_empty_strings, find_closest_string
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
Undefined = object()

def _len_dict_item(item):
    if False:
        print('Hello World!')
    '\n    Because a parsed dict path is a tuple containings strings or integers, to\n    know the length of the resulting string when printing we might need to\n    convert to a string before calling len on it.\n    '
    try:
        l = len(item)
    except TypeError:
        try:
            l = len('%d' % (item,))
        except TypeError:
            raise ValueError('Cannot find string length of an item that is not string-like nor an integer.')
    return l

def _str_to_dict_path_full(key_path_str):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert a key path string into a tuple of key path elements and also\n    return a tuple of indices marking the beginning of each element in the\n    string.\n\n    Parameters\n    ----------\n    key_path_str : str\n        Key path string, where nested keys are joined on '.' characters\n        and array indexes are specified using brackets\n        (e.g. 'foo.bar[1]')\n    Returns\n    -------\n    tuple[str | int]\n    tuple [int]\n    "
    if len(key_path_str):
        key_path2 = split_multichar([key_path_str], list('.[]'))
        key_path3 = []
        underscore_props = BaseFigure._valid_underscore_properties

        def _make_hyphen_key(key):
            if False:
                i = 10
                return i + 15
            if '_' in key[1:]:
                for (under_prop, hyphen_prop) in underscore_props.items():
                    key = key.replace(under_prop, hyphen_prop)
            return key

        def _make_underscore_key(key):
            if False:
                return 10
            return key.replace('-', '_')
        key_path2b = list(map(_make_hyphen_key, key_path2))

        def _split_and_chomp(s):
            if False:
                for i in range(10):
                    print('nop')
            if not len(s):
                return s
            s_split = split_multichar([s], list('_'))
            s_chomped = chomp_empty_strings(s_split, '_', reverse=True)
            return s_chomped
        key_path2c = list(reduce(lambda x, y: x + y if type(y) == type(list()) else x + [y], map(_split_and_chomp, key_path2b), []))
        key_path2d = list(map(_make_underscore_key, key_path2c))
        all_elem_idcs = tuple(split_string_positions(list(key_path2d)))
        key_elem_pairs = list(filter(lambda t: len(t[1]), enumerate(key_path2d)))
        key_path3 = [x for (_, x) in key_elem_pairs]
        elem_idcs = [all_elem_idcs[i] for (i, _) in key_elem_pairs]
        for i in range(len(key_path3)):
            try:
                key_path3[i] = int(key_path3[i])
            except ValueError as _:
                pass
    else:
        key_path3 = []
        elem_idcs = []
    return (tuple(key_path3), elem_idcs)

def _remake_path_from_tuple(props):
    if False:
        while True:
            i = 10
    '\n    try to remake a path using the properties in props\n    '
    if len(props) == 0:
        return ''

    def _add_square_brackets_to_number(n):
        if False:
            i = 10
            return i + 15
        if type(n) == type(int()):
            return '[%d]' % (n,)
        return n

    def _prepend_dot_if_not_number(s):
        if False:
            return 10
        if not s.startswith('['):
            return '.' + s
        return s
    props_all_str = list(map(_add_square_brackets_to_number, props))
    props_w_underscore = props_all_str[:1] + list(map(_prepend_dot_if_not_number, props_all_str[1:]))
    return ''.join(props_w_underscore)

def _check_path_in_prop_tree(obj, path, error_cast=None):
    if False:
        while True:
            i = 10
    '\n    obj:        the object in which the first property is looked up\n    path:       the path that will be split into properties to be looked up\n                path can also be a tuple. In this case, it is combined using .\n                and [] because it is impossible to reconstruct the string fully\n                in order to give a decent error message.\n    error_cast: this function walks down the property tree by looking up values\n                in objects. So this will throw exceptions that are thrown by\n                __getitem__, but in some cases we are checking the path for a\n                different reason and would prefer throwing a more relevant\n                exception (e.g., __getitem__ throws KeyError but __setitem__\n                throws ValueError for subclasses of BasePlotlyType and\n                BaseFigure). So the resulting error can be "casted" to the\n                passed in type, if not None.\n    returns\n          an Exception object or None. The caller can raise this\n          exception to see where the lookup error occurred.\n    '
    if isinstance(path, tuple):
        path = _remake_path_from_tuple(path)
    (prop, prop_idcs) = _str_to_dict_path_full(path)
    prev_objs = []
    for (i, p) in enumerate(prop):
        arg = ''
        prev_objs.append(obj)
        try:
            obj = obj[p]
        except (ValueError, KeyError, IndexError, TypeError) as e:
            arg = e.args[0]
            if issubclass(e.__class__, TypeError):
                if i > 0:
                    validator = prev_objs[i - 1]._get_validator(prop[i - 1])
                    arg += "\n\nInvalid value received for the '{plotly_name}' property of {parent_name}\n\n{description}".format(parent_name=validator.parent_name, plotly_name=validator.plotly_name, description=validator.description())
                disp_i = max(i - 1, 0)
                dict_item_len = _len_dict_item(prop[disp_i])
                trailing_underscores = ''
                if prop[i][0] == '_':
                    trailing_underscores = ' and path has trailing underscores'
                if trailing_underscores != '' and disp_i != i:
                    dict_item_len += _len_dict_item(prop[i])
                arg += '\n\nProperty does not support subscripting%s:\n%s\n%s' % (trailing_underscores, path, display_string_positions(prop_idcs, disp_i, length=dict_item_len, char='^'))
            else:
                arg += '\nBad property path:\n%s\n%s' % (path, display_string_positions(prop_idcs, i, length=_len_dict_item(prop[i]), char='^'))
            if isinstance(e, KeyError):
                e = PlotlyKeyError()
            if error_cast is not None:
                e = error_cast()
            e.args = (arg,)
            return e
    return None

def _combine_dicts(dicts):
    if False:
        return 10
    all_args = dict()
    for d in dicts:
        for k in d:
            all_args[k] = d[k]
    return all_args

def _indexing_combinations(dims, alls, product=False):
    if False:
        print('Hello World!')
    "\n    Gives indexing tuples specified by the coordinates in dims.\n    If a member of dims is 'all' then it is replaced by the corresponding member\n    in alls.\n    If product is True, then the cartesian product of all the indices is\n    returned, otherwise the zip (that means index lists of mis-matched length\n    will yield a list of tuples whose length is the length of the shortest\n    list).\n    "
    if len(dims) == 0:
        return []
    if len(dims) != len(alls):
        raise ValueError('Must have corresponding values in alls for each value of dims. Got dims=%s and alls=%s.' % (str(dims), str(alls)))
    r = []
    for (d, a) in zip(dims, alls):
        if d == 'all':
            d = a
        elif not isinstance(d, list):
            d = [d]
        r.append(d)
    if product:
        return itertools.product(*r)
    else:
        return zip(*r)

def _is_select_subplot_coordinates_arg(*args):
    if False:
        while True:
            i = 10
    "Returns true if any args are lists or the string 'all'"
    return any((a == 'all' or isinstance(a, list) for a in args))

def _axis_spanning_shapes_docstr(shape_type):
    if False:
        while True:
            i = 10
    docstr = ''
    if shape_type == 'hline':
        docstr = '\nAdd a horizontal line to a plot or subplot that extends infinitely in the\nx-dimension.\n\nParameters\n----------\ny: float or int\n    A number representing the y coordinate of the horizontal line.'
    elif shape_type == 'vline':
        docstr = '\nAdd a vertical line to a plot or subplot that extends infinitely in the\ny-dimension.\n\nParameters\n----------\nx: float or int\n    A number representing the x coordinate of the vertical line.'
    elif shape_type == 'hrect':
        docstr = '\nAdd a rectangle to a plot or subplot that extends infinitely in the\nx-dimension.\n\nParameters\n----------\ny0: float or int\n    A number representing the y coordinate of one side of the rectangle.\ny1: float or int\n    A number representing the y coordinate of the other side of the rectangle.'
    elif shape_type == 'vrect':
        docstr = '\nAdd a rectangle to a plot or subplot that extends infinitely in the\ny-dimension.\n\nParameters\n----------\nx0: float or int\n    A number representing the x coordinate of one side of the rectangle.\nx1: float or int\n    A number representing the x coordinate of the other side of the rectangle.'
    docstr += '\nexclude_empty_subplots: Boolean\n    If True (default) do not place the shape on subplots that have no data\n    plotted on them.\nrow: None, int or \'all\'\n    Subplot row for shape indexed starting at 1. If \'all\', addresses all rows in\n    the specified column(s). If both row and col are None, addresses the\n    first subplot if subplots exist, or the only plot. By default is "all".\ncol: None, int or \'all\'\n    Subplot column for shape indexed starting at 1. If \'all\', addresses all rows in\n    the specified column(s). If both row and col are None, addresses the\n    first subplot if subplots exist, or the only plot. By default is "all".\nannotation: dict or plotly.graph_objects.layout.Annotation. If dict(),\n    it is interpreted as describing an annotation. The annotation is\n    placed relative to the shape based on annotation_position (see\n    below) unless its x or y value has been specified for the annotation\n    passed here. xref and yref are always the same as for the added\n    shape and cannot be overridden.'
    if shape_type in ['hline', 'vline']:
        docstr += '\nannotation_position: a string containing optionally ["top", "bottom"]\n    and ["left", "right"] specifying where the text should be anchored\n    to on the line. Example positions are "bottom left", "right top",\n    "right", "bottom". If an annotation is added but annotation_position is\n    not specified, this defaults to "top right".'
    elif shape_type in ['hrect', 'vrect']:
        docstr += '\nannotation_position: a string containing optionally ["inside", "outside"], ["top", "bottom"]\n    and ["left", "right"] specifying where the text should be anchored\n    to on the rectangle. Example positions are "outside top left", "inside\n    bottom", "right", "inside left", "inside" ("outside" is not supported). If\n    an annotation is added but annotation_position is not specified this\n    defaults to "inside top right".'
    docstr += '\nannotation_*: any parameters to go.layout.Annotation can be passed as\n    keywords by prefixing them with "annotation_". For example, to specify the\n    annotation text "example" you can pass annotation_text="example" as a\n    keyword argument.\n**kwargs:\n    Any named function parameters that can be passed to \'add_shape\',\n    except for x0, x1, y0, y1 or type.'
    return docstr

def _generator(i):
    if False:
        return 10
    ' "cast" an iterator to a generator'
    for x in i:
        yield x

class BaseFigure(object):
    """
    Base class for all figure types (both widget and non-widget)
    """
    _bracket_re = re.compile('^(.*)\\[(\\d+)\\]$')
    _valid_underscore_properties = {'error_x': 'error-x', 'error_y': 'error-y', 'error_z': 'error-z', 'copy_xstyle': 'copy-xstyle', 'copy_ystyle': 'copy-ystyle', 'copy_zstyle': 'copy-zstyle', 'paper_bgcolor': 'paper-bgcolor', 'plot_bgcolor': 'plot-bgcolor'}
    _set_trace_uid = False
    _allow_disable_validation = True

    def __init__(self, data=None, layout_plotly=None, frames=None, skip_invalid=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a BaseFigure object\n\n        Parameters\n        ----------\n        data\n            One of:\n            - A list or tuple of trace objects (or dicts that can be coerced\n            into trace objects)\n\n            - If `data` is a dict that contains a 'data',\n            'layout', or 'frames' key then these values are used to\n            construct the figure.\n\n            - If `data` is a `BaseFigure` instance then the `data`, `layout`,\n            and `frames` properties are extracted from the input figure\n        layout_plotly\n            The plotly layout dict.\n\n            Note: this property is named `layout_plotly` rather than `layout`\n            to deconflict it with the `layout` constructor parameter of the\n            `widgets.DOMWidget` ipywidgets class, as the `BaseFigureWidget`\n            class is a subclass of both BaseFigure and widgets.DOMWidget.\n\n            If the `data` property is a BaseFigure instance, or a dict that\n            contains a 'layout' key, then this property is ignored.\n        frames\n            A list or tuple of `plotly.graph_objs.Frame` objects (or dicts\n            that can be coerced into Frame objects)\n\n            If the `data` property is a BaseFigure instance, or a dict that\n            contains a 'frames' key, then this property is ignored.\n\n        skip_invalid: bool\n            If True, invalid properties in the figure specification will be\n            skipped silently. If False (default) invalid properties in the\n            figure specification will result in a ValueError\n\n        Raises\n        ------\n        ValueError\n            if a property in the specification of data, layout, or frames\n            is invalid AND skip_invalid is False\n        "
        from .validators import DataValidator, LayoutValidator, FramesValidator
        super(BaseFigure, self).__init__()
        self._validate = kwargs.pop('_validate', True)
        layout = layout_plotly
        self._grid_str = None
        self._grid_ref = None
        if isinstance(data, BaseFigure):
            self._grid_str = data._grid_str
            self._grid_ref = data._grid_ref
            (data, layout, frames) = (data.data, data.layout, data.frames)
        elif isinstance(data, dict) and ('data' in data or 'layout' in data or 'frames' in data):
            self._grid_str = data.get('_grid_str', None)
            self._grid_ref = data.get('_grid_ref', None)
            (data, layout, frames) = (data.get('data', None), data.get('layout', None), data.get('frames', None))
        self._data_validator = DataValidator(set_uid=self._set_trace_uid)
        data = self._data_validator.validate_coerce(data, skip_invalid=skip_invalid, _validate=self._validate)
        self._data_objs = data
        self._data = [deepcopy(trace._props) for trace in data]
        self._data_defaults = [{} for _ in data]
        for (trace_ind, trace) in enumerate(data):
            trace._parent = self
            trace._orphan_props.clear()
            trace._trace_ind = trace_ind
        self._layout_validator = LayoutValidator()
        self._layout_obj = self._layout_validator.validate_coerce(layout, skip_invalid=skip_invalid, _validate=self._validate)
        self._layout = deepcopy(self._layout_obj._props)
        self._layout_defaults = {}
        self._layout_obj._orphan_props.clear()
        self._layout_obj._parent = self
        from plotly.offline.offline import _get_jconfig
        self._config = _get_jconfig(None)
        self._frames_validator = FramesValidator()
        self._frame_objs = self._frames_validator.validate_coerce(frames, skip_invalid=skip_invalid)
        self._in_batch_mode = False
        self._batch_trace_edits = OrderedDict()
        self._batch_layout_edits = OrderedDict()
        from . import animation
        self._animation_duration_validator = animation.DurationValidator()
        self._animation_easing_validator = animation.EasingValidator()
        self._initialize_layout_template()
        for (k, v) in kwargs.items():
            err = _check_path_in_prop_tree(self, k)
            if err is None:
                self[k] = v
            elif not skip_invalid:
                type_err = TypeError('invalid Figure property: {}'.format(k))
                type_err.args = (type_err.args[0] + '\n%s' % (err.args[0],),)
                raise type_err

    def __reduce__(self):
        if False:
            print('Hello World!')
        '\n        Custom implementation of reduce is used to support deep copying\n        and pickling\n        '
        props = self.to_dict()
        props['_grid_str'] = self._grid_str
        props['_grid_ref'] = self._grid_ref
        return (self.__class__, (props,))

    def __setitem__(self, prop, value):
        if False:
            for i in range(10):
                print('nop')
        orig_prop = prop
        prop = BaseFigure._str_to_dict_path(prop)
        if len(prop) == 0:
            raise KeyError(orig_prop)
        elif len(prop) == 1:
            prop = prop[0]
            if prop == 'data':
                self.data = value
            elif prop == 'layout':
                self.layout = value
            elif prop == 'frames':
                self.frames = value
            else:
                raise KeyError(prop)
        else:
            err = _check_path_in_prop_tree(self, orig_prop, error_cast=ValueError)
            if err is not None:
                raise err
            res = self
            for p in prop[:-1]:
                res = res[p]
            res._validate = self._validate
            res[prop[-1]] = value

    def __setattr__(self, prop, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        prop : str\n            The name of a direct child of this object\n        value\n            New property value\n        Returns\n        -------\n        None\n        '
        if prop.startswith('_') or hasattr(self, prop):
            super(BaseFigure, self).__setattr__(prop, value)
        else:
            raise AttributeError(prop)

    def __getitem__(self, prop):
        if False:
            i = 10
            return i + 15
        orig_prop = prop
        prop = BaseFigure._str_to_dict_path(prop)
        if len(prop) == 1:
            prop = prop[0]
            if prop == 'data':
                return self._data_validator.present(self._data_objs)
            elif prop == 'layout':
                return self._layout_validator.present(self._layout_obj)
            elif prop == 'frames':
                return self._frames_validator.present(self._frame_objs)
            else:
                raise KeyError(orig_prop)
        else:
            err = _check_path_in_prop_tree(self, orig_prop, error_cast=PlotlyKeyError)
            if err is not None:
                raise err
            res = self
            for p in prop:
                res = res[p]
            return res

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(('data', 'layout', 'frames'))

    def __contains__(self, prop):
        if False:
            print('Hello World!')
        prop = BaseFigure._str_to_dict_path(prop)
        if prop[0] not in ('data', 'layout', 'frames'):
            return False
        elif len(prop) == 1:
            return True
        else:
            return prop[1:] in self[prop[0]]

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, BaseFigure):
            return False
        else:
            return BasePlotlyType._vals_equal(self.to_plotly_json(), other.to_plotly_json())

    def __repr__(self):
        if False:
            return 10
        '\n        Customize Figure representation when displayed in the\n        terminal/notebook\n        '
        props = self.to_plotly_json()
        template_props = props.get('layout', {}).get('template', {})
        if template_props:
            props['layout']['template'] = '...'
        repr_str = BasePlotlyType._build_repr_for_class(props=props, class_name=self.__class__.__name__)
        return repr_str

    def _repr_html_(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Customize html representation\n        '
        bundle = self._repr_mimebundle_()
        if 'text/html' in bundle:
            return bundle['text/html']
        else:
            return self.to_html(full_html=False, include_plotlyjs='cdn')

    def _repr_mimebundle_(self, include=None, exclude=None, validate=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return mimebundle corresponding to default renderer.\n        '
        import plotly.io as pio
        renderer_str = pio.renderers.default
        renderers = pio._renderers.renderers
        from plotly.io._utils import validate_coerce_fig_to_dict
        fig_dict = validate_coerce_fig_to_dict(self, validate)
        return renderers._build_mime_bundle(fig_dict, renderer_str, **kwargs)

    def _ipython_display_(self):
        if False:
            i = 10
            return i + 15
        '\n        Handle rich display of figures in ipython contexts\n        '
        import plotly.io as pio
        if pio.renderers.render_on_display and pio.renderers.default:
            pio.show(self)
        else:
            print(repr(self))

    def update(self, dict1=None, overwrite=False, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Update the properties of the figure with a dict and/or with\n        keyword arguments.\n\n        This recursively updates the structure of the figure\n        object with the values in the input dict / keyword arguments.\n\n        Parameters\n        ----------\n        dict1 : dict\n            Dictionary of properties to be updated\n        overwrite: bool\n            If True, overwrite existing properties. If False, apply updates\n            to existing properties recursively, preserving existing\n            properties that are not specified in the update operation.\n        kwargs :\n            Keyword/value pair of properties to be updated\n\n        Examples\n        --------\n        >>> import plotly.graph_objs as go\n        >>> fig = go.Figure(data=[{'y': [1, 2, 3]}])\n        >>> fig.update(data=[{'y': [4, 5, 6]}]) # doctest: +ELLIPSIS\n        Figure(...)\n        >>> fig.to_plotly_json() # doctest: +SKIP\n            {'data': [{'type': 'scatter',\n               'uid': 'e86a7c7a-346a-11e8-8aa8-a0999b0c017b',\n               'y': array([4, 5, 6], dtype=int32)}],\n             'layout': {}}\n\n        >>> fig = go.Figure(layout={'xaxis':\n        ...                         {'color': 'green',\n        ...                          'range': [0, 1]}})\n        >>> fig.update({'layout': {'xaxis': {'color': 'pink'}}}) # doctest: +ELLIPSIS\n        Figure(...)\n        >>> fig.to_plotly_json() # doctest: +SKIP\n            {'data': [],\n             'layout': {'xaxis':\n                        {'color': 'pink',\n                         'range': [0, 1]}}}\n\n        Returns\n        -------\n        BaseFigure\n            Updated figure\n        "
        with self.batch_update():
            for d in [dict1, kwargs]:
                if d:
                    for (k, v) in d.items():
                        update_target = self[k]
                        if update_target == () or overwrite:
                            if k == 'data':
                                self.data = ()
                                self.add_traces(v)
                            else:
                                self[k] = v
                        elif isinstance(update_target, BasePlotlyType) and isinstance(v, (dict, BasePlotlyType)) or (isinstance(update_target, tuple) and isinstance(update_target[0], BasePlotlyType)):
                            BaseFigure._perform_update(self[k], v)
                        else:
                            self[k] = v
        return self

    def pop(self, key, *args):
        if False:
            while True:
                i = 10
        '\n        Remove the value associated with the specified key and return it\n\n        Parameters\n        ----------\n        key: str\n            Property name\n        dflt\n            The default value to return if key was not found in figure\n\n        Returns\n        -------\n        value\n            The removed value that was previously associated with key\n\n        Raises\n        ------\n        KeyError\n            If key is not in object and no dflt argument specified\n        '
        if key not in self and args:
            return args[0]
        elif key in self:
            val = self[key]
            self[key] = None
            return val
        else:
            raise KeyError(key)

    @property
    def data(self):
        if False:
            while True:
                i = 10
        "\n        The `data` property is a tuple of the figure's trace objects\n\n        Returns\n        -------\n        tuple[BaseTraceType]\n        "
        return self['data']

    @data.setter
    def data(self, new_data):
        if False:
            i = 10
            return i + 15
        err_header = 'The data property of a figure may only be assigned \na list or tuple that contains a permutation of a subset of itself.\n'
        if new_data is None:
            new_data = ()
        if not isinstance(new_data, (list, tuple)):
            err_msg = err_header + '    Received value with type {typ}'.format(typ=type(new_data))
            raise ValueError(err_msg)
        for trace in new_data:
            if not isinstance(trace, BaseTraceType):
                err_msg = err_header + '    Received element value of type {typ}'.format(typ=type(trace))
                raise ValueError(err_msg)
        orig_uids = [id(trace) for trace in self.data]
        new_uids = [id(trace) for trace in new_data]
        invalid_uids = set(new_uids).difference(set(orig_uids))
        if invalid_uids:
            err_msg = err_header
            raise ValueError(err_msg)
        uid_counter = collections.Counter(new_uids)
        duplicate_uids = [uid for (uid, count) in uid_counter.items() if count > 1]
        if duplicate_uids:
            err_msg = err_header + '    Received duplicated traces'
            raise ValueError(err_msg)
        remove_uids = set(orig_uids).difference(set(new_uids))
        delete_inds = []
        for (i, trace) in enumerate(self.data):
            if id(trace) in remove_uids:
                delete_inds.append(i)
                old_trace = self.data[i]
                old_trace._orphan_props.update(deepcopy(old_trace._props))
                old_trace._parent = None
                old_trace._trace_ind = None
        traces_props_post_removal = [t for t in self._data]
        traces_prop_defaults_post_removal = [t for t in self._data_defaults]
        uids_post_removal = [id(trace_data) for trace_data in self.data]
        for i in reversed(delete_inds):
            del traces_props_post_removal[i]
            del traces_prop_defaults_post_removal[i]
            del uids_post_removal[i]
            del self._data[i]
        if delete_inds:
            self._send_deleteTraces_msg(delete_inds)
        new_inds = []
        for uid in uids_post_removal:
            new_inds.append(new_uids.index(uid))
        current_inds = list(range(len(traces_props_post_removal)))
        if not all([i1 == i2 for (i1, i2) in zip(new_inds, current_inds)]):
            msg_current_inds = current_inds
            msg_new_inds = new_inds
            moving_traces_data = []
            for ci in reversed(current_inds):
                moving_traces_data.insert(0, self._data[ci])
                del self._data[ci]
            (new_inds, moving_traces_data) = zip(*sorted(zip(new_inds, moving_traces_data)))
            for (ni, trace_data) in zip(new_inds, moving_traces_data):
                self._data.insert(ni, trace_data)
            self._send_moveTraces_msg(msg_current_inds, msg_new_inds)
        self._data_defaults = [_trace for (i, _trace) in sorted(zip(new_inds, traces_prop_defaults_post_removal))]
        self._data_objs = list(new_data)
        for (trace_ind, trace) in enumerate(self._data_objs):
            trace._trace_ind = trace_ind

    def select_traces(self, selector=None, row=None, col=None, secondary_y=None):
        if False:
            return 10
        "\n        Select traces from a particular subplot cell and/or traces\n        that satisfy custom selection criteria.\n\n        Parameters\n        ----------\n        selector: dict, function, int, str or None (default None)\n            Dict to use as selection criteria.\n            Traces will be selected if they contain properties corresponding\n            to all of the dictionary's keys, with values that exactly match\n            the supplied values. If None (the default), all traces are\n            selected. If a function, it must be a function accepting a single\n            argument and returning a boolean. The function will be called on\n            each trace and those for which the function returned True\n            will be in the selection. If an int N, the Nth trace matching row\n            and col will be selected (N can be negative). If a string S, the selector\n            is equivalent to dict(type=S).\n        row, col: int or None (default None)\n            Subplot row and column index of traces to select.\n            To select traces by row and column, the Figure must have been\n            created using plotly.subplots.make_subplots.  If None\n            (the default), all traces are selected.\n        secondary_y: boolean or None (default None)\n            * If True, only select traces associated with the secondary\n              y-axis of the subplot.\n            * If False, only select traces associated with the primary\n              y-axis of the subplot.\n            * If None (the default), do not filter traces based on secondary\n              y-axis.\n\n            To select traces by secondary y-axis, the Figure must have been\n            created using plotly.subplots.make_subplots. See the docstring\n            for the specs argument to make_subplots for more info on\n            creating subplots with secondary y-axes.\n        Returns\n        -------\n        generator\n            Generator that iterates through all of the traces that satisfy\n            all of the specified selection criteria\n        "
        if not selector and (not isinstance(selector, int)):
            selector = {}
        if row is not None or col is not None or secondary_y is not None:
            grid_ref = self._validate_get_grid_ref()
            filter_by_subplot = True
            if row is None and col is not None:
                grid_subplot_ref_tuples = [ref_row[col - 1] for ref_row in grid_ref]
            elif col is None and row is not None:
                grid_subplot_ref_tuples = grid_ref[row - 1]
            elif col is not None and row is not None:
                grid_subplot_ref_tuples = [grid_ref[row - 1][col - 1]]
            else:
                grid_subplot_ref_tuples = [refs for refs_row in grid_ref for refs in refs_row]
            grid_subplot_refs = []
            for refs in grid_subplot_ref_tuples:
                if not refs:
                    continue
                if secondary_y is not True:
                    grid_subplot_refs.append(refs[0])
                if secondary_y is not False and len(refs) > 1:
                    grid_subplot_refs.append(refs[1])
        else:
            filter_by_subplot = False
            grid_subplot_refs = None
        return self._perform_select_traces(filter_by_subplot, grid_subplot_refs, selector)

    def _perform_select_traces(self, filter_by_subplot, grid_subplot_refs, selector):
        if False:
            i = 10
            return i + 15
        from plotly._subplots import _get_subplot_ref_for_trace

        def _filter_by_subplot_ref(trace):
            if False:
                for i in range(10):
                    print('nop')
            trace_subplot_ref = _get_subplot_ref_for_trace(trace)
            return trace_subplot_ref in grid_subplot_refs
        funcs = []
        if filter_by_subplot:
            funcs.append(_filter_by_subplot_ref)
        return _generator(self._filter_by_selector(self.data, funcs, selector))

    @staticmethod
    def _selector_matches(obj, selector):
        if False:
            i = 10
            return i + 15
        if selector is None:
            return True
        if isinstance(selector, str):
            selector = dict(type=selector)
        if isinstance(selector, dict) or isinstance(selector, BasePlotlyType):
            for k in selector:
                if k not in obj:
                    return False
                obj_val = obj[k]
                selector_val = selector[k]
                if isinstance(obj_val, BasePlotlyType):
                    obj_val = obj_val.to_plotly_json()
                if isinstance(selector_val, BasePlotlyType):
                    selector_val = selector_val.to_plotly_json()
                if obj_val != selector_val:
                    return False
            return True
        elif callable(selector):
            return selector(obj)
        else:
            raise TypeError('selector must be dict or a function accepting a graph object returning a boolean.')

    def _filter_by_selector(self, objects, funcs, selector):
        if False:
            print('Hello World!')
        '\n        objects is a sequence of objects, funcs a list of functions that\n        return True if the object should be included in the selection and False\n        otherwise and selector is an argument to the self._selector_matches\n        function.\n        If selector is an integer, the resulting sequence obtained after\n        sucessively filtering by each function in funcs is indexed by this\n        integer.\n        Otherwise selector is used as the selector argument to\n        self._selector_matches which is used to filter down the sequence.\n        The function returns the sequence (an iterator).\n        '
        if not isinstance(selector, int):
            funcs.append(lambda obj: self._selector_matches(obj, selector))

        def _filt(last, f):
            if False:
                i = 10
                return i + 15
            return filter(f, last)
        filtered_objects = reduce(_filt, funcs, objects)
        if isinstance(selector, int):
            return iter([list(filtered_objects)[selector]])
        return filtered_objects

    def for_each_trace(self, fn, selector=None, row=None, col=None, secondary_y=None):
        if False:
            print('Hello World!')
        "\n        Apply a function to all traces that satisfy the specified selection\n        criteria\n\n        Parameters\n        ----------\n        fn:\n            Function that inputs a single trace object.\n        selector: dict, function, int, str or None (default None)\n            Dict to use as selection criteria.\n            Traces will be selected if they contain properties corresponding\n            to all of the dictionary's keys, with values that exactly match\n            the supplied values. If None (the default), all traces are\n            selected. If a function, it must be a function accepting a single\n            argument and returning a boolean. The function will be called on\n            each trace and those for which the function returned True\n            will be in the selection. If an int N, the Nth trace matching row\n            and col will be selected (N can be negative). If a string S, the selector\n            is equivalent to dict(type=S).\n        row, col: int or None (default None)\n            Subplot row and column index of traces to select.\n            To select traces by row and column, the Figure must have been\n            created using plotly.subplots.make_subplots.  If None\n            (the default), all traces are selected.\n        secondary_y: boolean or None (default None)\n            * If True, only select traces associated with the secondary\n              y-axis of the subplot.\n            * If False, only select traces associated with the primary\n              y-axis of the subplot.\n            * If None (the default), do not filter traces based on secondary\n              y-axis.\n\n            To select traces by secondary y-axis, the Figure must have been\n            created using plotly.subplots.make_subplots. See the docstring\n            for the specs argument to make_subplots for more info on\n            creating subplots with secondary y-axes.\n        Returns\n        -------\n        self\n            Returns the Figure object that the method was called on\n        "
        for trace in self.select_traces(selector=selector, row=row, col=col, secondary_y=secondary_y):
            fn(trace)
        return self

    def update_traces(self, patch=None, selector=None, row=None, col=None, secondary_y=None, overwrite=False, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Perform a property update operation on all traces that satisfy the\n        specified selection criteria\n\n        Parameters\n        ----------\n        patch: dict or None (default None)\n            Dictionary of property updates to be applied to all traces that\n            satisfy the selection criteria.\n        selector: dict, function, int, str or None (default None)\n            Dict to use as selection criteria.\n            Traces will be selected if they contain properties corresponding\n            to all of the dictionary's keys, with values that exactly match\n            the supplied values. If None (the default), all traces are\n            selected. If a function, it must be a function accepting a single\n            argument and returning a boolean. The function will be called on\n            each trace and those for which the function returned True\n            will be in the selection. If an int N, the Nth trace matching row\n            and col will be selected (N can be negative). If a string S, the selector\n            is equivalent to dict(type=S).\n        row, col: int or None (default None)\n            Subplot row and column index of traces to select.\n            To select traces by row and column, the Figure must have been\n            created using plotly.subplots.make_subplots.  If None\n            (the default), all traces are selected.\n        secondary_y: boolean or None (default None)\n            * If True, only select traces associated with the secondary\n              y-axis of the subplot.\n            * If False, only select traces associated with the primary\n              y-axis of the subplot.\n            * If None (the default), do not filter traces based on secondary\n              y-axis.\n\n            To select traces by secondary y-axis, the Figure must have been\n            created using plotly.subplots.make_subplots. See the docstring\n            for the specs argument to make_subplots for more info on\n            creating subplots with secondary y-axes.\n        overwrite: bool\n            If True, overwrite existing properties. If False, apply updates\n            to existing properties recursively, preserving existing\n            properties that are not specified in the update operation.\n        **kwargs\n            Additional property updates to apply to each selected trace. If\n            a property is specified in both patch and in **kwargs then the\n            one in **kwargs takes precedence.\n\n        Returns\n        -------\n        self\n            Returns the Figure object that the method was called on\n        "
        for trace in self.select_traces(selector=selector, row=row, col=col, secondary_y=secondary_y):
            trace.update(patch, overwrite=overwrite, **kwargs)
        return self

    def update_layout(self, dict1=None, overwrite=False, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Update the properties of the figure's layout with a dict and/or with\n        keyword arguments.\n\n        This recursively updates the structure of the original\n        layout with the values in the input dict / keyword arguments.\n\n        Parameters\n        ----------\n        dict1 : dict\n            Dictionary of properties to be updated\n        overwrite: bool\n            If True, overwrite existing properties. If False, apply updates\n            to existing properties recursively, preserving existing\n            properties that are not specified in the update operation.\n        kwargs :\n            Keyword/value pair of properties to be updated\n\n        Returns\n        -------\n        BaseFigure\n            The Figure object that the update_layout method was called on\n        "
        self.layout.update(dict1, overwrite=overwrite, **kwargs)
        return self

    def _select_layout_subplots_by_prefix(self, prefix, selector=None, row=None, col=None, secondary_y=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper called by code generated select_* methods\n        '
        if row is not None or col is not None or secondary_y is not None:
            grid_ref = self._validate_get_grid_ref()
            container_to_row_col = {}
            for (r, subplot_row) in enumerate(grid_ref):
                for (c, subplot_refs) in enumerate(subplot_row):
                    if not subplot_refs:
                        continue
                    for (i, subplot_ref) in enumerate(subplot_refs):
                        for layout_key in subplot_ref.layout_keys:
                            if layout_key.startswith(prefix):
                                is_secondary_y = i == 1
                                container_to_row_col[layout_key] = (r + 1, c + 1, is_secondary_y)
        else:
            container_to_row_col = None
        layout_keys_filters = [lambda k: k.startswith(prefix) and self.layout[k] is not None, lambda k: row is None or container_to_row_col.get(k, (None, None, None))[0] == row, lambda k: col is None or container_to_row_col.get(k, (None, None, None))[1] == col, lambda k: secondary_y is None or container_to_row_col.get(k, (None, None, None))[2] == secondary_y]
        layout_keys = reduce(lambda last, f: filter(f, last), layout_keys_filters, _natural_sort_strings(list(self.layout)))
        layout_objs = [self.layout[k] for k in layout_keys]
        return _generator(self._filter_by_selector(layout_objs, [], selector))

    def _select_annotations_like(self, prop, selector=None, row=None, col=None, secondary_y=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper to select annotation-like elements from a layout object array.\n        Compatible with layout.annotations, layout.shapes, and layout.images\n        '
        xref_to_col = {}
        yref_to_row = {}
        yref_to_secondary_y = {}
        if isinstance(row, int) or isinstance(col, int) or secondary_y is not None:
            grid_ref = self._validate_get_grid_ref()
            for (r, subplot_row) in enumerate(grid_ref):
                for (c, subplot_refs) in enumerate(subplot_row):
                    if not subplot_refs:
                        continue
                    for (i, subplot_ref) in enumerate(subplot_refs):
                        if subplot_ref.subplot_type == 'xy':
                            is_secondary_y = i == 1
                            (xaxis, yaxis) = subplot_ref.layout_keys
                            xref = xaxis.replace('axis', '')
                            yref = yaxis.replace('axis', '')
                            xref_to_col[xref] = c + 1
                            yref_to_row[yref] = r + 1
                            yref_to_secondary_y[yref] = is_secondary_y

        def _filter_row(obj):
            if False:
                print('Hello World!')
            'Filter objects in rows by column'
            return col is None or xref_to_col.get(obj.xref, None) == col

        def _filter_col(obj):
            if False:
                print('Hello World!')
            'Filter objects in columns by row'
            return row is None or yref_to_row.get(obj.yref, None) == row

        def _filter_sec_y(obj):
            if False:
                for i in range(10):
                    print('nop')
            'Filter objects on secondary y axes'
            return secondary_y is None or yref_to_secondary_y.get(obj.yref, None) == secondary_y
        funcs = [_filter_row, _filter_col, _filter_sec_y]
        return _generator(self._filter_by_selector(self.layout[prop], funcs, selector))

    def _add_annotation_like(self, prop_singular, prop_plural, new_obj, row=None, col=None, secondary_y=None, exclude_empty_subplots=False):
        if False:
            print('Hello World!')
        if row is not None and col is None:
            raise ValueError('Received row parameter but not col.\nrow and col must be specified together')
        elif col is not None and row is None:
            raise ValueError('Received col parameter but not row.\nrow and col must be specified together')
        if row is not None and _is_select_subplot_coordinates_arg(row, col):
            rows_cols = self._select_subplot_coordinates(row, col)
            for (r, c) in rows_cols:
                self._add_annotation_like(prop_singular, prop_plural, new_obj, row=r, col=c, secondary_y=secondary_y, exclude_empty_subplots=exclude_empty_subplots)
            return self
        if row is not None:
            grid_ref = self._validate_get_grid_ref()
            if row > len(grid_ref):
                raise IndexError('row index %d out-of-bounds, row index must be between 1 and %d, inclusive.' % (row, len(grid_ref)))
            if col > len(grid_ref[row - 1]):
                raise IndexError('column index %d out-of-bounds, column index must be between 1 and %d, inclusive.' % (row, len(grid_ref[row - 1])))
            refs = grid_ref[row - 1][col - 1]
            if not refs:
                raise ValueError('No subplot found at position ({r}, {c})'.format(r=row, c=col))
            if refs[0].subplot_type != 'xy':
                raise ValueError('\nCannot add {prop_singular} to subplot at position ({r}, {c}) because subplot\nis of type {subplot_type}.'.format(prop_singular=prop_singular, r=row, c=col, subplot_type=refs[0].subplot_type))
            if new_obj.yref is None or new_obj.yref == 'y' or 'paper' in new_obj.yref or ('domain' in new_obj.yref):
                if len(refs) == 1 and secondary_y:
                    raise ValueError('\n    Cannot add {prop_singular} to secondary y-axis of subplot at position ({r}, {c})\n    because subplot does not have a secondary y-axis'.format(prop_singular=prop_singular, r=row, c=col))
                if secondary_y:
                    (xaxis, yaxis) = refs[1].layout_keys
                else:
                    (xaxis, yaxis) = refs[0].layout_keys
                (xref, yref) = (xaxis.replace('axis', ''), yaxis.replace('axis', ''))
            else:
                yref = new_obj.yref
                xaxis = refs[0].layout_keys[0]
                xref = xaxis.replace('axis', '')
            if exclude_empty_subplots and (not self._subplot_not_empty(xref, yref, selector=bool(exclude_empty_subplots))):
                return self

            def _add_domain(ax_letter, new_axref):
                if False:
                    i = 10
                    return i + 15
                axref = ax_letter + 'ref'
                if axref in new_obj._props.keys() and 'domain' in new_obj[axref]:
                    new_axref += ' domain'
                return new_axref
            (xref, yref) = map(lambda t: _add_domain(*t), zip(['x', 'y'], [xref, yref]))
            new_obj.update(xref=xref, yref=yref)
        self.layout[prop_plural] += (new_obj,)
        new_obj.update(xref=None, yref=None)
        return self

    def plotly_restyle(self, restyle_data, trace_indexes=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Perform a Plotly restyle operation on the figure's traces\n\n        Parameters\n        ----------\n        restyle_data : dict\n            Dict of trace style updates.\n\n            Keys are strings that specify the properties to be updated.\n            Nested properties are expressed by joining successive keys on\n            '.' characters (e.g. 'marker.color').\n\n            Values may be scalars or lists. When values are scalars,\n            that scalar value is applied to all traces specified by the\n            `trace_indexes` parameter.  When values are lists,\n            the restyle operation will cycle through the elements\n            of the list as it cycles through the traces specified by the\n            `trace_indexes` parameter.\n\n            Caution: To use plotly_restyle to update a list property (e.g.\n            the `x` property of the scatter trace), the property value\n            should be a scalar list containing the list to update with. For\n            example, the following command would be used to update the 'x'\n            property of the first trace to the list [1, 2, 3]\n\n            >>> import plotly.graph_objects as go\n            >>> fig = go.Figure(go.Scatter(x=[2, 4, 6]))\n            >>> fig.plotly_restyle({'x': [[1, 2, 3]]}, 0)\n\n        trace_indexes : int or list of int\n            Trace index, or list of trace indexes, that the restyle operation\n            applies to. Defaults to all trace indexes.\n\n        Returns\n        -------\n        None\n        "
        trace_indexes = self._normalize_trace_indexes(trace_indexes)
        source_view_id = kwargs.get('source_view_id', None)
        restyle_changes = self._perform_plotly_restyle(restyle_data, trace_indexes)
        if restyle_changes:
            msg_kwargs = {'source_view_id': source_view_id} if source_view_id is not None else {}
            self._send_restyle_msg(restyle_changes, trace_indexes=trace_indexes, **msg_kwargs)
            self._dispatch_trace_change_callbacks(restyle_changes, trace_indexes)

    def _perform_plotly_restyle(self, restyle_data, trace_indexes):
        if False:
            return 10
        "\n        Perform a restyle operation on the figure's traces data and return\n        the changes that were applied\n\n        Parameters\n        ----------\n        restyle_data : dict[str, any]\n            See docstring for plotly_restyle\n        trace_indexes : list[int]\n            List of trace indexes that restyle operation applies to\n        Returns\n        -------\n        restyle_changes: dict[str, any]\n            Subset of restyle_data including only the keys / values that\n            resulted in a change to the figure's traces data\n        "
        restyle_changes = {}
        for (key_path_str, v) in restyle_data.items():
            any_vals_changed = False
            for (i, trace_ind) in enumerate(trace_indexes):
                if trace_ind >= len(self._data):
                    raise ValueError('Trace index {trace_ind} out of range'.format(trace_ind=trace_ind))
                trace_v = v[i % len(v)] if isinstance(v, list) else v
                if trace_v is not Undefined:
                    trace_obj = self.data[trace_ind]
                    if not BaseFigure._is_key_path_compatible(key_path_str, trace_obj):
                        trace_class = trace_obj.__class__.__name__
                        raise ValueError("\nInvalid property path '{key_path_str}' for trace class {trace_class}\n".format(key_path_str=key_path_str, trace_class=trace_class))
                    val_changed = BaseFigure._set_in(self._data[trace_ind], key_path_str, trace_v)
                    any_vals_changed = any_vals_changed or val_changed
            if any_vals_changed:
                restyle_changes[key_path_str] = v
        return restyle_changes

    def _restyle_child(self, child, key_path_str, val):
        if False:
            i = 10
            return i + 15
        "\n        Process restyle operation on a child trace object\n\n        Note: This method name/signature must match the one in\n        BasePlotlyType. BasePlotlyType objects call their parent's\n        _restyle_child method without knowing whether their parent is a\n        BasePlotlyType or a BaseFigure.\n\n        Parameters\n        ----------\n        child : BaseTraceType\n            Child being restyled\n        key_path_str : str\n            A key path string (e.g. 'foo.bar[0]')\n        val\n            Restyle value\n\n        Returns\n        -------\n        None\n        "
        trace_index = child._trace_ind
        if not self._in_batch_mode:
            send_val = [val]
            restyle = {key_path_str: send_val}
            self._send_restyle_msg(restyle, trace_indexes=trace_index)
            self._dispatch_trace_change_callbacks(restyle, [trace_index])
        else:
            if trace_index not in self._batch_trace_edits:
                self._batch_trace_edits[trace_index] = OrderedDict()
            self._batch_trace_edits[trace_index][key_path_str] = val

    def _normalize_trace_indexes(self, trace_indexes):
        if False:
            return 10
        '\n        Input trace index specification and return list of the specified trace\n        indexes\n\n        Parameters\n        ----------\n        trace_indexes : None or int or list[int]\n\n        Returns\n        -------\n        list[int]\n        '
        if trace_indexes is None:
            trace_indexes = list(range(len(self.data)))
        if not isinstance(trace_indexes, (list, tuple)):
            trace_indexes = [trace_indexes]
        return list(trace_indexes)

    @staticmethod
    def _str_to_dict_path(key_path_str):
        if False:
            i = 10
            return i + 15
        "\n        Convert a key path string into a tuple of key path elements.\n\n        Parameters\n        ----------\n        key_path_str : str\n            Key path string, where nested keys are joined on '.' characters\n            and array indexes are specified using brackets\n            (e.g. 'foo.bar[1]')\n        Returns\n        -------\n        tuple[str | int]\n        "
        if isinstance(key_path_str, str) and '.' not in key_path_str and ('[' not in key_path_str) and ('_' not in key_path_str):
            return (key_path_str,)
        elif isinstance(key_path_str, tuple):
            return key_path_str
        else:
            ret = _str_to_dict_path_full(key_path_str)[0]
            return ret

    @staticmethod
    def _set_in(d, key_path_str, v):
        if False:
            while True:
                i = 10
        "\n        Set a value in a nested dict using a key path string\n        (e.g. 'foo.bar[0]')\n\n        Parameters\n        ----------\n        d : dict\n            Input dict to set property in\n        key_path_str : str\n            Key path string, where nested keys are joined on '.' characters\n            and array indexes are specified using brackets\n            (e.g. 'foo.bar[1]')\n        v\n            New value\n        Returns\n        -------\n        bool\n            True if set resulted in modification of dict (i.e. v was not\n            already present at the specified location), False otherwise.\n        "
        assert isinstance(d, dict)
        key_path = BaseFigure._str_to_dict_path(key_path_str)
        val_parent = d
        for (kp, key_path_el) in enumerate(key_path[:-1]):
            if isinstance(val_parent, list) and isinstance(key_path_el, int):
                while len(val_parent) <= key_path_el:
                    val_parent.append(None)
            elif isinstance(val_parent, dict) and key_path_el not in val_parent:
                if isinstance(key_path[kp + 1], int):
                    val_parent[key_path_el] = []
                else:
                    val_parent[key_path_el] = {}
            val_parent = val_parent[key_path_el]
        last_key = key_path[-1]
        val_changed = False
        if v is Undefined:
            pass
        elif v is None:
            if isinstance(val_parent, dict):
                if last_key in val_parent:
                    val_parent.pop(last_key)
                    val_changed = True
            elif isinstance(val_parent, list):
                if isinstance(last_key, int) and 0 <= last_key < len(val_parent):
                    val_parent[last_key] = None
                    val_changed = True
            else:
                raise ValueError('\n    Cannot remove element of type {typ} at location {raw_key}'.format(typ=type(val_parent), raw_key=key_path_str))
        elif isinstance(val_parent, dict):
            if last_key not in val_parent or not BasePlotlyType._vals_equal(val_parent[last_key], v):
                val_parent[last_key] = v
                val_changed = True
        elif isinstance(val_parent, list):
            if isinstance(last_key, int):
                while len(val_parent) <= last_key:
                    val_parent.append(None)
                if not BasePlotlyType._vals_equal(val_parent[last_key], v):
                    val_parent[last_key] = v
                    val_changed = True
        else:
            raise ValueError('\n    Cannot set element of type {typ} at location {raw_key}'.format(typ=type(val_parent), raw_key=key_path_str))
        return val_changed

    @staticmethod
    def _raise_invalid_rows_cols(name, n, invalid):
        if False:
            for i in range(10):
                print('nop')
        rows_err_msg = '\n        If specified, the {name} parameter must be a list or tuple of integers\n        of length {n} (The number of traces being added)\n\n        Received: {invalid}\n        '.format(name=name, n=n, invalid=invalid)
        raise ValueError(rows_err_msg)

    @staticmethod
    def _validate_rows_cols(name, n, vals):
        if False:
            print('Hello World!')
        if vals is None:
            pass
        elif isinstance(vals, (list, tuple)):
            if len(vals) != n:
                BaseFigure._raise_invalid_rows_cols(name=name, n=n, invalid=vals)
            int_type = _get_int_type()
            if [r for r in vals if not isinstance(r, int_type)]:
                BaseFigure._raise_invalid_rows_cols(name=name, n=n, invalid=vals)
        else:
            BaseFigure._raise_invalid_rows_cols(name=name, n=n, invalid=vals)

    def add_trace(self, trace, row=None, col=None, secondary_y=None, exclude_empty_subplots=False):
        if False:
            while True:
                i = 10
        "\n        Add a trace to the figure\n\n        Parameters\n        ----------\n        trace : BaseTraceType or dict\n            Either:\n              - An instances of a trace classe from the plotly.graph_objs\n                package (e.g plotly.graph_objs.Scatter, plotly.graph_objs.Bar)\n              - or a dicts where:\n\n                  - The 'type' property specifies the trace type (e.g.\n                    'scatter', 'bar', 'area', etc.). If the dict has no 'type'\n                    property then 'scatter' is assumed.\n                  - All remaining properties are passed to the constructor\n                    of the specified trace type.\n\n        row : 'all', int or None (default)\n            Subplot row index (starting from 1) for the trace to be\n            added. Only valid if figure was created using\n            `plotly.tools.make_subplots`.\n            If 'all', addresses all rows in the specified column(s).\n        col : 'all', int or None (default)\n            Subplot col index (starting from 1) for the trace to be\n            added. Only valid if figure was created using\n            `plotly.tools.make_subplots`.\n            If 'all', addresses all columns in the specified row(s).\n        secondary_y: boolean or None (default None)\n            If True, associate this trace with the secondary y-axis of the\n            subplot at the specified row and col. Only valid if all of the\n            following conditions are satisfied:\n              * The figure was created using `plotly.subplots.make_subplots`.\n              * The row and col arguments are not None\n              * The subplot at the specified row and col has type xy\n                (which is the default) and secondary_y True.  These\n                properties are specified in the specs argument to\n                make_subplots. See the make_subplots docstring for more info.\n              * The trace argument is a 2D cartesian trace\n                (scatter, bar, etc.)\n        exclude_empty_subplots: boolean\n            If True, the trace will not be added to subplots that don't already\n            have traces.\n        Returns\n        -------\n        BaseFigure\n            The Figure that add_trace was called on\n\n        Examples\n        --------\n\n        >>> from plotly import subplots\n        >>> import plotly.graph_objs as go\n\n        Add two Scatter traces to a figure\n\n        >>> fig = go.Figure()\n        >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2])) # doctest: +ELLIPSIS\n        Figure(...)\n        >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2])) # doctest: +ELLIPSIS\n        Figure(...)\n\n\n        Add two Scatter traces to vertically stacked subplots\n\n        >>> fig = subplots.make_subplots(rows=2)\n        >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=1, col=1) # doctest: +ELLIPSIS\n        Figure(...)\n        >>> fig.add_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=2, col=1) # doctest: +ELLIPSIS\n        Figure(...)\n        "
        if row is not None and col is None:
            raise ValueError('Received row parameter but not col.\nrow and col must be specified together')
        elif col is not None and row is None:
            raise ValueError('Received col parameter but not row.\nrow and col must be specified together')
        if row is not None and _is_select_subplot_coordinates_arg(row, col):
            rows_cols = self._select_subplot_coordinates(row, col)
            for (r, c) in rows_cols:
                self.add_trace(trace, row=r, col=c, secondary_y=secondary_y, exclude_empty_subplots=exclude_empty_subplots)
            return self
        return self.add_traces(data=[trace], rows=[row] if row is not None else None, cols=[col] if col is not None else None, secondary_ys=[secondary_y] if secondary_y is not None else None, exclude_empty_subplots=exclude_empty_subplots)

    def add_traces(self, data, rows=None, cols=None, secondary_ys=None, exclude_empty_subplots=False):
        if False:
            while True:
                i = 10
        "\n        Add traces to the figure\n\n        Parameters\n        ----------\n        data : list[BaseTraceType or dict]\n            A list of trace specifications to be added.\n            Trace specifications may be either:\n\n              - Instances of trace classes from the plotly.graph_objs\n                package (e.g plotly.graph_objs.Scatter, plotly.graph_objs.Bar)\n              - Dicts where:\n\n                  - The 'type' property specifies the trace type (e.g.\n                    'scatter', 'bar', 'area', etc.). If the dict has no 'type'\n                    property then 'scatter' is assumed.\n                  - All remaining properties are passed to the constructor\n                    of the specified trace type.\n\n        rows : None, list[int], or int (default None)\n            List of subplot row indexes (starting from 1) for the traces to be\n            added. Only valid if figure was created using\n            `plotly.tools.make_subplots`\n            If a single integer is passed, all traces will be added to row number\n\n        cols : None or list[int] (default None)\n            List of subplot column indexes (starting from 1) for the traces\n            to be added. Only valid if figure was created using\n            `plotly.tools.make_subplots`\n            If a single integer is passed, all traces will be added to column number\n\n\n        secondary_ys: None or list[boolean] (default None)\n            List of secondary_y booleans for traces to be added. See the\n            docstring for `add_trace` for more info.\n\n        exclude_empty_subplots: boolean\n            If True, the trace will not be added to subplots that don't already\n            have traces.\n\n        Returns\n        -------\n        BaseFigure\n            The Figure that add_traces was called on\n\n        Examples\n        --------\n\n        >>> from plotly import subplots\n        >>> import plotly.graph_objs as go\n\n        Add two Scatter traces to a figure\n\n        >>> fig = go.Figure()\n        >>> fig.add_traces([go.Scatter(x=[1,2,3], y=[2,1,2]),\n        ...                 go.Scatter(x=[1,2,3], y=[2,1,2])]) # doctest: +ELLIPSIS\n        Figure(...)\n\n        Add two Scatter traces to vertically stacked subplots\n\n        >>> fig = subplots.make_subplots(rows=2)\n        >>> fig.add_traces([go.Scatter(x=[1,2,3], y=[2,1,2]),\n        ...                 go.Scatter(x=[1,2,3], y=[2,1,2])],\n        ...                 rows=[1, 2], cols=[1, 1]) # doctest: +ELLIPSIS\n        Figure(...)\n        "
        data = self._data_validator.validate_coerce(data)
        for (ind, new_trace) in enumerate(data):
            new_trace._trace_ind = ind + len(self.data)
        int_type = _get_int_type()
        if isinstance(rows, int_type):
            rows = [rows] * len(data)
        if isinstance(cols, int_type):
            cols = [cols] * len(data)
        n = len(data)
        BaseFigure._validate_rows_cols('rows', n, rows)
        BaseFigure._validate_rows_cols('cols', n, cols)
        if rows is not None and cols is None:
            raise ValueError('Received rows parameter but not cols.\nrows and cols must be specified together')
        elif cols is not None and rows is None:
            raise ValueError('Received cols parameter but not rows.\nrows and cols must be specified together')
        if secondary_ys is not None and rows is None:
            rows = [1] * len(secondary_ys)
            cols = rows
        elif secondary_ys is None and rows is not None:
            secondary_ys = [None] * len(rows)
        if rows is not None:
            for (trace, row, col, secondary_y) in zip(data, rows, cols, secondary_ys):
                self._set_trace_grid_position(trace, row, col, secondary_y)
        if exclude_empty_subplots:
            data = list(filter(lambda trace: self._subplot_not_empty(trace['xaxis'], trace['yaxis'], bool(exclude_empty_subplots)), data))
        new_traces_data = [deepcopy(trace._props) for trace in data]
        for trace in data:
            trace._parent = self
            trace._orphan_props.clear()
        self._data.extend(new_traces_data)
        self._data_defaults = self._data_defaults + [{} for _ in data]
        self._data_objs = self._data_objs + data
        self._send_addTraces_msg(new_traces_data)
        return self

    def print_grid(self):
        if False:
            return 10
        "\n        Print a visual layout of the figure's axes arrangement.\n        This is only valid for figures that are created\n        with plotly.tools.make_subplots.\n        "
        if self._grid_str is None:
            raise Exception('Use plotly.tools.make_subplots to create a subplot grid.')
        print(self._grid_str)

    def append_trace(self, trace, row, col):
        if False:
            while True:
                i = 10
        '\n        Add a trace to the figure bound to axes at the specified row,\n        col index.\n\n        A row, col index grid is generated for figures created with\n        plotly.tools.make_subplots, and can be viewed with the `print_grid`\n        method\n\n        Parameters\n        ----------\n        trace\n            The data trace to be bound\n        row: int\n            Subplot row index (see Figure.print_grid)\n        col: int\n            Subplot column index (see Figure.print_grid)\n\n        Examples\n        --------\n\n        >>> from plotly import tools\n        >>> import plotly.graph_objs as go\n        >>> # stack two subplots vertically\n        >>> fig = tools.make_subplots(rows=2)\n\n        This is the format of your plot grid:\n        [ (1,1) x1,y1 ]\n        [ (2,1) x2,y2 ]\n\n        >>> fig.append_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=1, col=1)\n        >>> fig.append_trace(go.Scatter(x=[1,2,3], y=[2,1,2]), row=2, col=1)\n        '
        warnings.warn('The append_trace method is deprecated and will be removed in a future version.\nPlease use the add_trace method with the row and col parameters.\n', DeprecationWarning)
        self.add_trace(trace=trace, row=row, col=col)

    def _set_trace_grid_position(self, trace, row, col, secondary_y=False):
        if False:
            i = 10
            return i + 15
        from plotly._subplots import _set_trace_grid_reference
        grid_ref = self._validate_get_grid_ref()
        return _set_trace_grid_reference(trace, self.layout, grid_ref, row, col, secondary_y)

    def _validate_get_grid_ref(self):
        if False:
            i = 10
            return i + 15
        try:
            grid_ref = self._grid_ref
            if grid_ref is None:
                raise AttributeError('_grid_ref')
        except AttributeError:
            raise Exception('In order to reference traces by row and column, you must first use plotly.tools.make_subplots to create the figure with a subplot grid.')
        return grid_ref

    def _get_subplot_rows_columns(self):
        if False:
            print('Hello World!')
        '\n        Returns a pair of lists, the first containing all the row indices and\n        the second all the column indices.\n        '
        grid_ref = self._validate_get_grid_ref()
        nrows = len(grid_ref)
        ncols = len(grid_ref[0])
        return (range(1, nrows + 1), range(1, ncols + 1))

    def _get_subplot_coordinates(self):
        if False:
            print('Hello World!')
        '\n        Returns an iterator over (row,col) pairs representing all the possible\n        subplot coordinates.\n        '
        return itertools.product(*self._get_subplot_rows_columns())

    def _select_subplot_coordinates(self, rows, cols, product=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Allows selecting all or a subset of the subplots.\n        If any of rows or columns is 'all', product is set to True. This is\n        probably the expected behaviour, so that rows=1,cols='all' selects all\n        the columns in row 1 (otherwise it would just select the subplot in the\n        first row and first column).\n        "
        product |= any([s == 'all' for s in [rows, cols]])
        t = _indexing_combinations([rows, cols], list(self._get_subplot_rows_columns()), product=product)
        t = list(t)
        grid_ref = self._validate_get_grid_ref()
        t = list(filter(lambda u: grid_ref[u[0] - 1][u[1] - 1] is not None, t))
        return t

    def get_subplot(self, row, col, secondary_y=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return an object representing the subplot at the specified row\n        and column.  May only be used on Figures created using\n        plotly.tools.make_subplots\n\n        Parameters\n        ----------\n        row: int\n            1-based index of subplot row\n        col: int\n            1-based index of subplot column\n        secondary_y: bool\n            If True, select the subplot that consists of the x-axis and the\n            secondary y-axis at the specified row/col. Only valid if the\n            subplot at row/col is an 2D cartesian subplot that was created\n            with a secondary y-axis.  See the docstring for the specs argument\n            to make_subplots for more info on creating a subplot with a\n            secondary y-axis.\n        Returns\n        -------\n        subplot\n            * None: if subplot is empty\n            * plotly.graph_objs.layout.Scene: if subplot type is 'scene'\n            * plotly.graph_objs.layout.Polar: if subplot type is 'polar'\n            * plotly.graph_objs.layout.Ternary: if subplot type is 'ternary'\n            * plotly.graph_objs.layout.Mapbox: if subplot type is 'ternary'\n            * SubplotDomain namedtuple with `x` and `y` fields:\n              if subplot type is 'domain'.\n                - x: length 2 list of the subplot start and stop width\n                - y: length 2 list of the subplot start and stop height\n            * SubplotXY namedtuple with `xaxis` and `yaxis` fields:\n              if subplot type is 'xy'.\n                - xaxis: plotly.graph_objs.layout.XAxis instance for subplot\n                - yaxis: plotly.graph_objs.layout.YAxis instance for subplot\n        "
        from plotly._subplots import _get_grid_subplot
        return _get_grid_subplot(self, row, col, secondary_y)

    def _get_child_props(self, child):
        if False:
            i = 10
            return i + 15
        '\n        Return the properties dict for a child trace or child layout\n\n        Note: this method must match the name/signature of one on\n        BasePlotlyType\n\n        Parameters\n        ----------\n        child : BaseTraceType | BaseLayoutType\n\n        Returns\n        -------\n        dict\n        '
        if isinstance(child, BaseTraceType):
            trace_index = child._trace_ind
            return self._data[trace_index]
        elif child is self.layout:
            return self._layout
        else:
            raise ValueError('Unrecognized child: %s' % child)

    def _get_child_prop_defaults(self, child):
        if False:
            return 10
        '\n        Return the default properties dict for a child trace or child layout\n\n        Note: this method must match the name/signature of one on\n        BasePlotlyType\n\n        Parameters\n        ----------\n        child : BaseTraceType | BaseLayoutType\n\n        Returns\n        -------\n        dict\n        '
        if isinstance(child, BaseTraceType):
            trace_index = child._trace_ind
            return self._data_defaults[trace_index]
        elif child is self.layout:
            return self._layout_defaults
        else:
            raise ValueError('Unrecognized child: %s' % child)

    def _init_child_props(self, child):
        if False:
            print('Hello World!')
        '\n        Initialize the properites dict for a child trace or layout\n\n        Note: this method must match the name/signature of one on\n        BasePlotlyType\n\n        Parameters\n        ----------\n        child : BaseTraceType | BaseLayoutType\n\n        Returns\n        -------\n        None\n        '
        pass

    def _initialize_layout_template(self):
        if False:
            return 10
        import plotly.io as pio
        if self._layout_obj._props.get('template', None) is None:
            if pio.templates.default is not None:
                if self._allow_disable_validation:
                    self._layout_obj._validate = False
                try:
                    if isinstance(pio.templates.default, BasePlotlyType):
                        template_object = pio.templates.default
                    else:
                        template_object = pio.templates[pio.templates.default]
                    self._layout_obj.template = template_object
                finally:
                    self._layout_obj._validate = self._validate

    @property
    def layout(self):
        if False:
            print('Hello World!')
        '\n        The `layout` property of the figure\n\n        Returns\n        -------\n        plotly.graph_objs.Layout\n        '
        return self['layout']

    @layout.setter
    def layout(self, new_layout):
        if False:
            i = 10
            return i + 15
        new_layout = self._layout_validator.validate_coerce(new_layout)
        new_layout_data = deepcopy(new_layout._props)
        if self._layout_obj:
            old_layout_data = deepcopy(self._layout_obj._props)
            self._layout_obj._orphan_props.update(old_layout_data)
            self._layout_obj._parent = None
        self._layout = new_layout_data
        new_layout._parent = self
        new_layout._orphan_props.clear()
        self._layout_obj = new_layout
        self._initialize_layout_template()
        self._send_relayout_msg(new_layout_data)

    def plotly_relayout(self, relayout_data, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Perform a Plotly relayout operation on the figure's layout\n\n        Parameters\n        ----------\n        relayout_data : dict\n            Dict of layout updates\n\n            dict keys are strings that specify the properties to be updated.\n            Nested properties are expressed by joining successive keys on\n            '.' characters (e.g. 'xaxis.range')\n\n            dict values are the values to use to update the layout.\n\n        Returns\n        -------\n        None\n        "
        if 'source_view_id' in kwargs:
            msg_kwargs = {'source_view_id': kwargs['source_view_id']}
        else:
            msg_kwargs = {}
        relayout_changes = self._perform_plotly_relayout(relayout_data)
        if relayout_changes:
            self._send_relayout_msg(relayout_changes, **msg_kwargs)
            self._dispatch_layout_change_callbacks(relayout_changes)

    def _perform_plotly_relayout(self, relayout_data):
        if False:
            print('Hello World!')
        "\n        Perform a relayout operation on the figure's layout data and return\n        the changes that were applied\n\n        Parameters\n        ----------\n        relayout_data : dict[str, any]\n            See the docstring for plotly_relayout\n        Returns\n        -------\n        relayout_changes: dict[str, any]\n            Subset of relayout_data including only the keys / values that\n            resulted in a change to the figure's layout data\n        "
        relayout_changes = {}
        for (key_path_str, v) in relayout_data.items():
            if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):
                raise ValueError("\nInvalid property path '{key_path_str}' for layout\n".format(key_path_str=key_path_str))
            val_changed = BaseFigure._set_in(self._layout, key_path_str, v)
            if val_changed:
                relayout_changes[key_path_str] = v
        return relayout_changes

    @staticmethod
    def _is_key_path_compatible(key_path_str, plotly_obj):
        if False:
            while True:
                i = 10
        '\n        Return whether the specifieid key path string is compatible with\n        the specified plotly object for the purpose of relayout/restyle\n        operation\n        '
        key_path_tuple = BaseFigure._str_to_dict_path(key_path_str)
        if isinstance(key_path_tuple[-1], int):
            key_path_tuple = key_path_tuple[:-1]
        return key_path_tuple in plotly_obj

    def _relayout_child(self, child, key_path_str, val):
        if False:
            return 10
        "\n        Process relayout operation on child layout object\n\n        Parameters\n        ----------\n        child : BaseLayoutType\n            The figure's layout\n        key_path_str :\n            A key path string (e.g. 'foo.bar[0]')\n        val\n            Relayout value\n\n        Returns\n        -------\n        None\n        "
        assert child is self.layout
        if not self._in_batch_mode:
            relayout_msg = {key_path_str: val}
            self._send_relayout_msg(relayout_msg)
            self._dispatch_layout_change_callbacks(relayout_msg)
        else:
            self._batch_layout_edits[key_path_str] = val

    @staticmethod
    def _build_dispatch_plan(key_path_strs):
        if False:
            i = 10
            return i + 15
        "\n        Build a dispatch plan for a list of key path strings\n\n        A dispatch plan is a dict:\n           - *from* path tuples that reference an object that has descendants\n             that are referenced in `key_path_strs`.\n           - *to* sets of tuples that correspond to descendants of the object\n             above.\n\n        Parameters\n        ----------\n        key_path_strs : list[str]\n            List of key path strings. For example:\n\n            ['xaxis.rangeselector.font.color', 'xaxis.rangeselector.bgcolor']\n\n        Returns\n        -------\n        dispatch_plan: dict[tuple[str|int], set[tuple[str|int]]]\n\n        Examples\n        --------\n\n        >>> key_path_strs = ['xaxis.rangeselector.font.color',\n        ...                  'xaxis.rangeselector.bgcolor']\n\n        >>> BaseFigure._build_dispatch_plan(key_path_strs) # doctest: +SKIP\n            {(): {'xaxis',\n                  ('xaxis', 'rangeselector'),\n                  ('xaxis', 'rangeselector', 'bgcolor'),\n                  ('xaxis', 'rangeselector', 'font'),\n                  ('xaxis', 'rangeselector', 'font', 'color')},\n             ('xaxis',): {('rangeselector',),\n                          ('rangeselector', 'bgcolor'),\n                          ('rangeselector', 'font'),\n                          ('rangeselector', 'font', 'color')},\n             ('xaxis', 'rangeselector'): {('bgcolor',),\n                                          ('font',),\n                                          ('font', 'color')},\n             ('xaxis', 'rangeselector', 'font'): {('color',)}}\n        "
        dispatch_plan = {}
        for key_path_str in key_path_strs:
            key_path = BaseFigure._str_to_dict_path(key_path_str)
            key_path_so_far = ()
            keys_left = key_path
            for next_key in key_path:
                if key_path_so_far not in dispatch_plan:
                    dispatch_plan[key_path_so_far] = set()
                to_add = [keys_left[:i + 1] for i in range(len(keys_left))]
                dispatch_plan[key_path_so_far].update(to_add)
                key_path_so_far = key_path_so_far + (next_key,)
                keys_left = keys_left[1:]
        return dispatch_plan

    def _dispatch_layout_change_callbacks(self, relayout_data):
        if False:
            while True:
                i = 10
        '\n        Dispatch property change callbacks given relayout_data\n\n        Parameters\n        ----------\n        relayout_data : dict[str, any]\n            See docstring for plotly_relayout.\n\n        Returns\n        -------\n        None\n        '
        key_path_strs = list(relayout_data.keys())
        dispatch_plan = BaseFigure._build_dispatch_plan(key_path_strs)
        for (path_tuple, changed_paths) in dispatch_plan.items():
            if path_tuple in self.layout:
                dispatch_obj = self.layout[path_tuple]
                if isinstance(dispatch_obj, BasePlotlyType):
                    dispatch_obj._dispatch_change_callbacks(changed_paths)

    def _dispatch_trace_change_callbacks(self, restyle_data, trace_indexes):
        if False:
            print('Hello World!')
        '\n        Dispatch property change callbacks given restyle_data\n\n        Parameters\n        ----------\n        restyle_data : dict[str, any]\n            See docstring for plotly_restyle.\n\n        trace_indexes : list[int]\n            List of trace indexes that restyle operation applied to\n\n        Returns\n        -------\n        None\n        '
        key_path_strs = list(restyle_data.keys())
        dispatch_plan = BaseFigure._build_dispatch_plan(key_path_strs)
        for (path_tuple, changed_paths) in dispatch_plan.items():
            for trace_ind in trace_indexes:
                trace = self.data[trace_ind]
                if path_tuple in trace:
                    dispatch_obj = trace[path_tuple]
                    if isinstance(dispatch_obj, BasePlotlyType):
                        dispatch_obj._dispatch_change_callbacks(changed_paths)

    @property
    def frames(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The `frames` property is a tuple of the figure's frame objects\n\n        Returns\n        -------\n        tuple[plotly.graph_objs.Frame]\n        "
        return self['frames']

    @frames.setter
    def frames(self, new_frames):
        if False:
            for i in range(10):
                print('nop')
        self._frame_objs = self._frames_validator.validate_coerce(new_frames)

    def plotly_update(self, restyle_data=None, relayout_data=None, trace_indexes=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform a Plotly update operation on the figure.\n\n        Note: This operation both mutates and returns the figure\n\n        Parameters\n        ----------\n        restyle_data : dict\n            Traces update specification. See the docstring for the\n            `plotly_restyle` method for details\n        relayout_data : dict\n            Layout update specification. See the docstring for the\n            `plotly_relayout` method for details\n        trace_indexes :\n            Trace index, or list of trace indexes, that the update operation\n            applies to. Defaults to all trace indexes.\n\n        Returns\n        -------\n        BaseFigure\n            None\n        '
        if 'source_view_id' in kwargs:
            msg_kwargs = {'source_view_id': kwargs['source_view_id']}
        else:
            msg_kwargs = {}
        (restyle_changes, relayout_changes, trace_indexes) = self._perform_plotly_update(restyle_data=restyle_data, relayout_data=relayout_data, trace_indexes=trace_indexes)
        if restyle_changes or relayout_changes:
            self._send_update_msg(restyle_data=restyle_changes, relayout_data=relayout_changes, trace_indexes=trace_indexes, **msg_kwargs)
        if restyle_changes:
            self._dispatch_trace_change_callbacks(restyle_changes, trace_indexes)
        if relayout_changes:
            self._dispatch_layout_change_callbacks(relayout_changes)

    def _perform_plotly_update(self, restyle_data=None, relayout_data=None, trace_indexes=None):
        if False:
            i = 10
            return i + 15
        if not restyle_data and (not relayout_data):
            return (None, None, None)
        if restyle_data is None:
            restyle_data = {}
        if relayout_data is None:
            relayout_data = {}
        trace_indexes = self._normalize_trace_indexes(trace_indexes)
        relayout_changes = self._perform_plotly_relayout(relayout_data)
        restyle_changes = self._perform_plotly_restyle(restyle_data, trace_indexes)
        return (restyle_changes, relayout_changes, trace_indexes)

    def _send_addTraces_msg(self, new_traces_data):
        if False:
            return 10
        pass

    def _send_moveTraces_msg(self, current_inds, new_inds):
        if False:
            for i in range(10):
                print('nop')
        pass

    def _send_deleteTraces_msg(self, delete_inds):
        if False:
            while True:
                i = 10
        pass

    def _send_restyle_msg(self, style, trace_indexes=None, source_view_id=None):
        if False:
            return 10
        pass

    def _send_relayout_msg(self, layout, source_view_id=None):
        if False:
            i = 10
            return i + 15
        pass

    def _send_update_msg(self, restyle_data, relayout_data, trace_indexes=None, source_view_id=None):
        if False:
            i = 10
            return i + 15
        pass

    def _send_animate_msg(self, styles_data, relayout_data, trace_indexes, animation_opts):
        if False:
            i = 10
            return i + 15
        pass

    @contextmanager
    def batch_update(self):
        if False:
            return 10
        "\n        A context manager that batches up trace and layout assignment\n        operations into a singe plotly_update message that is executed when\n        the context exits.\n\n        Examples\n        --------\n        For example, suppose we have a figure widget, `fig`, with a single\n        trace.\n\n        >>> import plotly.graph_objs as go\n        >>> fig = go.FigureWidget(data=[{'y': [3, 4, 2]}])\n\n        If we want to update the xaxis range, the yaxis range, and the\n        marker color, we could do so using a series of three property\n        assignments as follows:\n\n        >>> fig.layout.xaxis.range = [0, 5]\n        >>> fig.layout.yaxis.range = [0, 10]\n        >>> fig.data[0].marker.color = 'green'\n\n        This will work, however it will result in three messages being\n        sent to the front end (two relayout messages for the axis range\n        updates followed by one restyle message for the marker color\n        update). This can cause the plot to appear to stutter as the\n        three updates are applied incrementally.\n\n        We can avoid this problem by performing these three assignments in a\n        `batch_update` context as follows:\n\n        >>> with fig.batch_update():\n        ...     fig.layout.xaxis.range = [0, 5]\n        ...     fig.layout.yaxis.range = [0, 10]\n        ...     fig.data[0].marker.color = 'green'\n\n        Now, these three property updates will be sent to the frontend in a\n        single update message, and they will be applied by the front end\n        simultaneously.\n        "
        if self._in_batch_mode is True:
            yield
        else:
            try:
                self._in_batch_mode = True
                yield
            finally:
                self._in_batch_mode = False
                (restyle_data, relayout_data, trace_indexes) = self._build_update_params_from_batch()
                self.plotly_update(restyle_data=restyle_data, relayout_data=relayout_data, trace_indexes=trace_indexes)
                self._batch_layout_edits.clear()
                self._batch_trace_edits.clear()

    def _build_update_params_from_batch(self):
        if False:
            print('Hello World!')
        '\n        Convert `_batch_trace_edits` and `_batch_layout_edits` into the\n        `restyle_data`, `relayout_data`, and `trace_indexes` params accepted\n        by the `plotly_update` method.\n\n        Returns\n        -------\n        (dict, dict, list[int])\n        '
        batch_style_commands = self._batch_trace_edits
        trace_indexes = sorted(set([trace_ind for trace_ind in batch_style_commands]))
        all_props = sorted(set([prop for trace_style in self._batch_trace_edits.values() for prop in trace_style]))
        restyle_data = {prop: [Undefined for _ in range(len(trace_indexes))] for prop in all_props}
        for (trace_ind, trace_style) in batch_style_commands.items():
            for (trace_prop, trace_val) in trace_style.items():
                restyle_trace_index = trace_indexes.index(trace_ind)
                restyle_data[trace_prop][restyle_trace_index] = trace_val
        relayout_data = self._batch_layout_edits
        return (restyle_data, relayout_data, trace_indexes)

    @contextmanager
    def batch_animate(self, duration=500, easing='cubic-in-out'):
        if False:
            i = 10
            return i + 15
        "\n        Context manager to animate trace / layout updates\n\n        Parameters\n        ----------\n        duration : number\n            The duration of the transition, in milliseconds.\n            If equal to zero, updates are synchronous.\n        easing : string\n            The easing function used for the transition.\n            One of:\n                - linear\n                - quad\n                - cubic\n                - sin\n                - exp\n                - circle\n                - elastic\n                - back\n                - bounce\n                - linear-in\n                - quad-in\n                - cubic-in\n                - sin-in\n                - exp-in\n                - circle-in\n                - elastic-in\n                - back-in\n                - bounce-in\n                - linear-out\n                - quad-out\n                - cubic-out\n                - sin-out\n                - exp-out\n                - circle-out\n                - elastic-out\n                - back-out\n                - bounce-out\n                - linear-in-out\n                - quad-in-out\n                - cubic-in-out\n                - sin-in-out\n                - exp-in-out\n                - circle-in-out\n                - elastic-in-out\n                - back-in-out\n                - bounce-in-out\n\n        Examples\n        --------\n        Suppose we have a figure widget, `fig`, with a single trace.\n\n        >>> import plotly.graph_objs as go\n        >>> fig = go.FigureWidget(data=[{'y': [3, 4, 2]}])\n\n        1) Animate a change in the xaxis and yaxis ranges using default\n        duration and easing parameters.\n\n        >>> with fig.batch_animate():\n        ...     fig.layout.xaxis.range = [0, 5]\n        ...     fig.layout.yaxis.range = [0, 10]\n\n        2) Animate a change in the size and color of the trace's markers\n        over 2 seconds using the elastic-in-out easing method\n\n        >>> with fig.batch_animate(duration=2000, easing='elastic-in-out'):\n        ...     fig.data[0].marker.color = 'green'\n        ...     fig.data[0].marker.size = 20\n        "
        duration = self._animation_duration_validator.validate_coerce(duration)
        easing = self._animation_easing_validator.validate_coerce(easing)
        if self._in_batch_mode is True:
            yield
        else:
            try:
                self._in_batch_mode = True
                yield
            finally:
                self._in_batch_mode = False
                self._perform_batch_animate({'transition': {'duration': duration, 'easing': easing}, 'frame': {'duration': duration}})

    def _perform_batch_animate(self, animation_opts):
        if False:
            return 10
        '\n        Perform the batch animate operation\n\n        This method should be called with the batch_animate() context\n        manager exits.\n\n        Parameters\n        ----------\n        animation_opts : dict\n            Animation options as accepted by frontend Plotly.animation command\n\n        Returns\n        -------\n        None\n        '
        (restyle_data, relayout_data, trace_indexes) = self._build_update_params_from_batch()
        (restyle_changes, relayout_changes, trace_indexes) = self._perform_plotly_update(restyle_data, relayout_data, trace_indexes)
        if self._batch_trace_edits:
            (animate_styles, animate_trace_indexes) = zip(*[(trace_style, trace_index) for (trace_index, trace_style) in self._batch_trace_edits.items()])
        else:
            (animate_styles, animate_trace_indexes) = ({}, [])
        animate_layout = copy(self._batch_layout_edits)
        self._send_animate_msg(styles_data=list(animate_styles), relayout_data=animate_layout, trace_indexes=list(animate_trace_indexes), animation_opts=animation_opts)
        self._batch_layout_edits.clear()
        self._batch_trace_edits.clear()
        if restyle_changes:
            self._dispatch_trace_change_callbacks(restyle_changes, trace_indexes)
        if relayout_changes:
            self._dispatch_layout_change_callbacks(relayout_changes)

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        '\n        Convert figure to a dictionary\n\n        Note: the dictionary includes the properties explicitly set by the\n        user, it does not include default values of unspecified properties\n\n        Returns\n        -------\n        dict\n        '
        data = deepcopy(self._data)
        layout = deepcopy(self._layout)
        res = {'data': data, 'layout': layout}
        frames = deepcopy([frame._props for frame in self._frame_objs])
        if frames:
            res['frames'] = frames
        return res

    def to_plotly_json(self):
        if False:
            print('Hello World!')
        '\n        Convert figure to a JSON representation as a Python dict\n\n        Note: May include some JSON-invalid data types, use the `PlotlyJSONEncoder` util\n        or the `to_json` method to encode to a string.\n\n        Returns\n        -------\n        dict\n        '
        return self.to_dict()

    @staticmethod
    def _to_ordered_dict(d, skip_uid=False):
        if False:
            i = 10
            return i + 15
        '\n        Static helper for converting dict or list to structure of ordered\n        dictionaries\n        '
        if isinstance(d, dict):
            result = collections.OrderedDict()
            for key in sorted(d.keys()):
                if skip_uid and key == 'uid':
                    continue
                else:
                    result[key] = BaseFigure._to_ordered_dict(d[key], skip_uid=skip_uid)
        elif isinstance(d, list) and d and isinstance(d[0], dict):
            result = [BaseFigure._to_ordered_dict(el, skip_uid=skip_uid) for el in d]
        else:
            result = d
        return result

    def to_ordered_dict(self, skip_uid=True):
        if False:
            return 10
        result = collections.OrderedDict()
        result['data'] = BaseFigure._to_ordered_dict(self._data, skip_uid=skip_uid)
        result['layout'] = BaseFigure._to_ordered_dict(self._layout)
        if self._frame_objs:
            frames_props = [frame._props for frame in self._frame_objs]
            result['frames'] = BaseFigure._to_ordered_dict(frames_props)
        return result

    def show(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Show a figure using either the default renderer(s) or the renderer(s)\n        specified by the renderer argument\n\n        Parameters\n        ----------\n        renderer: str or None (default None)\n            A string containing the names of one or more registered renderers\n            (separated by '+' characters) or None.  If None, then the default\n            renderers specified in plotly.io.renderers.default are used.\n\n        validate: bool (default True)\n            True if the figure should be validated before being shown,\n            False otherwise.\n\n        width: int or float\n            An integer or float that determines the number of pixels wide the\n            plot is. The default is set in plotly.js.\n\n        height: int or float\n            An integer or float that determines the number of pixels wide the\n            plot is. The default is set in plotly.js.\n\n        config: dict\n            A dict of parameters to configure the figure. The defaults are set\n            in plotly.js.\n\n        Returns\n        -------\n        None\n        "
        import plotly.io as pio
        return pio.show(self, *args, **kwargs)

    def to_json(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Convert a figure to a JSON string representation\n\n        Parameters\n        ----------\n        validate: bool (default True)\n            True if the figure should be validated before being converted to\n            JSON, False otherwise.\n\n        pretty: bool (default False)\n            True if JSON representation should be pretty-printed, False if\n            representation should be as compact as possible.\n\n        remove_uids: bool (default True)\n            True if trace UIDs should be omitted from the JSON representation\n\n        engine: str (default None)\n            The JSON encoding engine to use. One of:\n              - "json" for an encoder based on the built-in Python json module\n              - "orjson" for a fast encoder the requires the orjson package\n            If not specified, the default encoder is set to the current value of\n            plotly.io.json.config.default_encoder.\n\n        Returns\n        -------\n        str\n            Representation of figure as a JSON string\n        '
        import plotly.io as pio
        return pio.to_json(self, *args, **kwargs)

    def full_figure_for_development(self, warn=True, as_dict=False):
        if False:
            print('Hello World!')
        '\n        Compute default values for all attributes not specified in the input figure and\n        returns the output as a "full" figure. This function calls Plotly.js via Kaleido\n        to populate unspecified attributes. This function is intended for interactive use\n        during development to learn more about how Plotly.js computes default values and is\n        not generally necessary or recommended for production use.\n\n        Parameters\n        ----------\n        fig:\n            Figure object or dict representing a figure\n\n        warn: bool\n            If False, suppress warnings about not using this in production.\n\n        as_dict: bool\n            If True, output is a dict with some keys that go.Figure can\'t parse.\n            If False, output is a go.Figure with unparseable keys skipped.\n\n        Returns\n        -------\n        plotly.graph_objects.Figure or dict\n            The full figure\n        '
        import plotly.io as pio
        return pio.full_figure_for_development(self, warn, as_dict)

    def write_json(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Convert a figure to JSON and write it to a file or writeable\n        object\n\n        Parameters\n        ----------\n        file: str or writeable\n            A string representing a local file path or a writeable object\n            (e.g. an open file descriptor)\n\n        pretty: bool (default False)\n            True if JSON representation should be pretty-printed, False if\n            representation should be as compact as possible.\n\n        remove_uids: bool (default True)\n            True if trace UIDs should be omitted from the JSON representation\n\n        engine: str (default None)\n            The JSON encoding engine to use. One of:\n              - "json" for an encoder based on the built-in Python json module\n              - "orjson" for a fast encoder the requires the orjson package\n            If not specified, the default encoder is set to the current value of\n            plotly.io.json.config.default_encoder.\n\n        Returns\n        -------\n        None\n        '
        import plotly.io as pio
        return pio.write_json(self, *args, **kwargs)

    def to_html(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Convert a figure to an HTML string representation.\n\n        Parameters\n        ----------\n        config: dict or None (default None)\n            Plotly.js figure config options\n        auto_play: bool (default=True)\n            Whether to automatically start the animation sequence on page load\n            if the figure contains frames. Has no effect if the figure does not\n            contain frames.\n        include_plotlyjs: bool or string (default True)\n            Specifies how the plotly.js library is included/loaded in the output\n            div string.\n\n            If True, a script tag containing the plotly.js source code (~3MB)\n            is included in the output.  HTML files generated with this option are\n            fully self-contained and can be used offline.\n\n            If 'cdn', a script tag that references the plotly.js CDN is included\n            in the output. HTML files generated with this option are about 3MB\n            smaller than those generated with include_plotlyjs=True, but they\n            require an active internet connection in order to load the plotly.js\n            library.\n\n            If 'directory', a script tag is included that references an external\n            plotly.min.js bundle that is assumed to reside in the same\n            directory as the HTML file.\n\n            If 'require', Plotly.js is loaded using require.js.  This option\n            assumes that require.js is globally available and that it has been\n            globally configured to know how to find Plotly.js as 'plotly'.\n            This option is not advised when full_html=True as it will result\n            in a non-functional html file.\n\n            If a string that ends in '.js', a script tag is included that\n            references the specified path. This approach can be used to point\n            the resulting HTML file to an alternative CDN or local bundle.\n\n            If False, no script tag referencing plotly.js is included. This is\n            useful when the resulting div string will be placed inside an HTML\n            document that already loads plotly.js. This option is not advised\n            when full_html=True as it will result in a non-functional html file.\n        include_mathjax: bool or string (default False)\n            Specifies how the MathJax.js library is included in the output html\n            div string.  MathJax is required in order to display labels\n            with LaTeX typesetting.\n\n            If False, no script tag referencing MathJax.js will be included in the\n            output.\n\n            If 'cdn', a script tag that references a MathJax CDN location will be\n            included in the output.  HTML div strings generated with this option\n            will be able to display LaTeX typesetting as long as internet access\n            is available.\n\n            If a string that ends in '.js', a script tag is included that\n            references the specified path. This approach can be used to point the\n            resulting HTML div string to an alternative CDN.\n        post_script: str or list or None (default None)\n            JavaScript snippet(s) to be included in the resulting div just after\n            plot creation.  The string(s) may include '{plot_id}' placeholders\n            that will then be replaced by the `id` of the div element that the\n            plotly.js figure is associated with.  One application for this script\n            is to install custom plotly.js event handlers.\n        full_html: bool (default True)\n            If True, produce a string containing a complete HTML document\n            starting with an <html> tag.  If False, produce a string containing\n            a single <div> element.\n        animation_opts: dict or None (default None)\n            dict of custom animation parameters to be passed to the function\n            Plotly.animate in Plotly.js. See\n            https://github.com/plotly/plotly.js/blob/master/src/plots/animation_attributes.js\n            for available options. Has no effect if the figure does not contain\n            frames, or auto_play is False.\n        default_width, default_height: number or str (default '100%')\n            The default figure width/height to use if the provided figure does not\n            specify its own layout.width/layout.height property.  May be\n            specified in pixels as an integer (e.g. 500), or as a css width style\n            string (e.g. '500px', '100%').\n        validate: bool (default True)\n            True if the figure should be validated before being converted to\n            JSON, False otherwise.\n        div_id: str (default None)\n            If provided, this is the value of the id attribute of the div tag. If None, the\n            id attribute is a UUID.\n\n        Returns\n        -------\n        str\n            Representation of figure as an HTML div string\n        "
        import plotly.io as pio
        return pio.to_html(self, *args, **kwargs)

    def write_html(self, *args, **kwargs):
        if False:
            return 10
        "\n        Write a figure to an HTML file representation\n\n        Parameters\n        ----------\n        file: str or writeable\n            A string representing a local file path or a writeable object\n            (e.g. a pathlib.Path object or an open file descriptor)\n        config: dict or None (default None)\n            Plotly.js figure config options\n        auto_play: bool (default=True)\n            Whether to automatically start the animation sequence on page load\n            if the figure contains frames. Has no effect if the figure does not\n            contain frames.\n        include_plotlyjs: bool or string (default True)\n            Specifies how the plotly.js library is included/loaded in the output\n            div string.\n\n            If True, a script tag containing the plotly.js source code (~3MB)\n            is included in the output.  HTML files generated with this option are\n            fully self-contained and can be used offline.\n\n            If 'cdn', a script tag that references the plotly.js CDN is included\n            in the output. HTML files generated with this option are about 3MB\n            smaller than those generated with include_plotlyjs=True, but they\n            require an active internet connection in order to load the plotly.js\n            library.\n\n            If 'directory', a script tag is included that references an external\n            plotly.min.js bundle that is assumed to reside in the same\n            directory as the HTML file. If `file` is a string to a local file path\n            and `full_html` is True then\n\n            If 'directory', a script tag is included that references an external\n            plotly.min.js bundle that is assumed to reside in the same\n            directory as the HTML file.  If `file` is a string to a local file\n            path and `full_html` is True, then the plotly.min.js bundle is copied\n            into the directory of the resulting HTML file. If a file named\n            plotly.min.js already exists in the output directory then this file\n            is left unmodified and no copy is performed. HTML files generated\n            with this option can be used offline, but they require a copy of\n            the plotly.min.js bundle in the same directory. This option is\n            useful when many figures will be saved as HTML files in the same\n            directory because the plotly.js source code will be included only\n            once per output directory, rather than once per output file.\n\n            If 'require', Plotly.js is loaded using require.js.  This option\n            assumes that require.js is globally available and that it has been\n            globally configured to know how to find Plotly.js as 'plotly'.\n            This option is not advised when full_html=True as it will result\n            in a non-functional html file.\n\n            If a string that ends in '.js', a script tag is included that\n            references the specified path. This approach can be used to point\n            the resulting HTML file to an alternative CDN or local bundle.\n\n            If False, no script tag referencing plotly.js is included. This is\n            useful when the resulting div string will be placed inside an HTML\n            document that already loads plotly.js.  This option is not advised\n            when full_html=True as it will result in a non-functional html file.\n\n        include_mathjax: bool or string (default False)\n            Specifies how the MathJax.js library is included in the output html\n            div string.  MathJax is required in order to display labels\n            with LaTeX typesetting.\n\n            If False, no script tag referencing MathJax.js will be included in the\n            output.\n\n            If 'cdn', a script tag that references a MathJax CDN location will be\n            included in the output.  HTML div strings generated with this option\n            will be able to display LaTeX typesetting as long as internet access\n            is available.\n\n            If a string that ends in '.js', a script tag is included that\n            references the specified path. This approach can be used to point the\n            resulting HTML div string to an alternative CDN.\n        post_script: str or list or None (default None)\n            JavaScript snippet(s) to be included in the resulting div just after\n            plot creation.  The string(s) may include '{plot_id}' placeholders\n            that will then be replaced by the `id` of the div element that the\n            plotly.js figure is associated with.  One application for this script\n            is to install custom plotly.js event handlers.\n        full_html: bool (default True)\n            If True, produce a string containing a complete HTML document\n            starting with an <html> tag.  If False, produce a string containing\n            a single <div> element.\n        animation_opts: dict or None (default None)\n            dict of custom animation parameters to be passed to the function\n            Plotly.animate in Plotly.js. See\n            https://github.com/plotly/plotly.js/blob/master/src/plots/animation_attributes.js\n            for available options. Has no effect if the figure does not contain\n            frames, or auto_play is False.\n        default_width, default_height: number or str (default '100%')\n            The default figure width/height to use if the provided figure does not\n            specify its own layout.width/layout.height property.  May be\n            specified in pixels as an integer (e.g. 500), or as a css width style\n            string (e.g. '500px', '100%').\n        validate: bool (default True)\n            True if the figure should be validated before being converted to\n            JSON, False otherwise.\n        auto_open: bool (default True)\n            If True, open the saved file in a web browser after saving.\n            This argument only applies if `full_html` is True.\n        div_id: str (default None)\n            If provided, this is the value of the id attribute of the div tag. If None, the\n            id attribute is a UUID.\n\n        Returns\n        -------\n        str\n            Representation of figure as an HTML div string\n        "
        import plotly.io as pio
        return pio.write_html(self, *args, **kwargs)

    def to_image(self, *args, **kwargs):
        if False:
            return 10
        '\n        Convert a figure to a static image bytes string\n\n        Parameters\n        ----------\n        format: str or None\n            The desired image format. One of\n              - \'png\'\n              - \'jpg\' or \'jpeg\'\n              - \'webp\'\n              - \'svg\'\n              - \'pdf\'\n              - \'eps\' (Requires the poppler library to be installed)\n\n            If not specified, will default to `plotly.io.config.default_format`\n\n        width: int or None\n            The width of the exported image in layout pixels. If the `scale`\n            property is 1.0, this will also be the width of the exported image\n            in physical pixels.\n\n            If not specified, will default to `plotly.io.config.default_width`\n\n        height: int or None\n            The height of the exported image in layout pixels. If the `scale`\n            property is 1.0, this will also be the height of the exported image\n            in physical pixels.\n\n            If not specified, will default to `plotly.io.config.default_height`\n\n        scale: int or float or None\n            The scale factor to use when exporting the figure. A scale factor\n            larger than 1.0 will increase the image resolution with respect\n            to the figure\'s layout pixel dimensions. Whereas as scale factor of\n            less than 1.0 will decrease the image resolution.\n\n            If not specified, will default to `plotly.io.config.default_scale`\n\n        validate: bool\n            True if the figure should be validated before being converted to\n            an image, False otherwise.\n\n        engine: str\n            Image export engine to use:\n             - "kaleido": Use Kaleido for image export\n             - "orca": Use Orca for image export\n             - "auto" (default): Use Kaleido if installed, otherwise use orca\n\n        Returns\n        -------\n        bytes\n            The image data\n        '
        import plotly.io as pio
        return pio.to_image(self, *args, **kwargs)

    def write_image(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Convert a figure to a static image and write it to a file or writeable\n        object\n\n        Parameters\n        ----------\n        file: str or writeable\n            A string representing a local file path or a writeable object\n            (e.g. a pathlib.Path object or an open file descriptor)\n\n        format: str or None\n            The desired image format. One of\n              - \'png\'\n              - \'jpg\' or \'jpeg\'\n              - \'webp\'\n              - \'svg\'\n              - \'pdf\'\n              - \'eps\' (Requires the poppler library to be installed)\n\n            If not specified and `file` is a string then this will default to the\n            file extension. If not specified and `file` is not a string then this\n            will default to `plotly.io.config.default_format`\n\n        width: int or None\n            The width of the exported image in layout pixels. If the `scale`\n            property is 1.0, this will also be the width of the exported image\n            in physical pixels.\n\n            If not specified, will default to `plotly.io.config.default_width`\n\n        height: int or None\n            The height of the exported image in layout pixels. If the `scale`\n            property is 1.0, this will also be the height of the exported image\n            in physical pixels.\n\n            If not specified, will default to `plotly.io.config.default_height`\n\n        scale: int or float or None\n            The scale factor to use when exporting the figure. A scale factor\n            larger than 1.0 will increase the image resolution with respect\n            to the figure\'s layout pixel dimensions. Whereas as scale factor of\n            less than 1.0 will decrease the image resolution.\n\n            If not specified, will default to `plotly.io.config.default_scale`\n\n        validate: bool\n            True if the figure should be validated before being converted to\n            an image, False otherwise.\n\n        engine: str\n            Image export engine to use:\n             - "kaleido": Use Kaleido for image export\n             - "orca": Use Orca for image export\n             - "auto" (default): Use Kaleido if installed, otherwise use orca\n        Returns\n        -------\n        None\n        '
        import plotly.io as pio
        return pio.write_image(self, *args, **kwargs)

    @staticmethod
    def _is_dict_list(v):
        if False:
            while True:
                i = 10
        '\n        Return true of the input object is a list of dicts\n        '
        return isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)

    @staticmethod
    def _perform_update(plotly_obj, update_obj, overwrite=False):
        if False:
            while True:
                i = 10
        '\n        Helper to support the update() methods on :class:`BaseFigure` and\n        :class:`BasePlotlyType`\n\n        Parameters\n        ----------\n        plotly_obj : BasePlotlyType|tuple[BasePlotlyType]\n            Object to up updated\n        update_obj : dict|list[dict]|tuple[dict]\n            When ``plotly_obj`` is an instance of :class:`BaseFigure`,\n            ``update_obj`` should be a dict\n\n            When ``plotly_obj`` is a tuple of instances of\n            :class:`BasePlotlyType`, ``update_obj`` should be a tuple or list\n            of dicts\n        '
        from _plotly_utils.basevalidators import CompoundValidator, CompoundArrayValidator
        if update_obj is None:
            return
        elif isinstance(plotly_obj, BasePlotlyType):
            for key in update_obj:
                if key not in plotly_obj and isinstance(plotly_obj, BaseLayoutType):
                    match = plotly_obj._subplot_re_match(key)
                    if match:
                        plotly_obj[key] = {}
                        continue
                err = _check_path_in_prop_tree(plotly_obj, key, error_cast=ValueError)
                if err is not None:
                    raise err
            if isinstance(update_obj, BasePlotlyType):
                update_obj = update_obj.to_plotly_json()
            for key in update_obj:
                val = update_obj[key]
                if overwrite:
                    plotly_obj[key] = val
                    continue
                validator = plotly_obj._get_prop_validator(key)
                if isinstance(validator, CompoundValidator) and isinstance(val, dict):
                    BaseFigure._perform_update(plotly_obj[key], val)
                elif isinstance(validator, CompoundArrayValidator):
                    if plotly_obj[key]:
                        BaseFigure._perform_update(plotly_obj[key], val)
                        if isinstance(val, (list, tuple)) and len(val) > len(plotly_obj[key]):
                            plotly_obj[key] = plotly_obj[key] + tuple(val[len(plotly_obj[key]):])
                    else:
                        plotly_obj[key] = val
                else:
                    plotly_obj[key] = val
        elif isinstance(plotly_obj, tuple):
            if len(update_obj) == 0:
                return
            else:
                for (i, plotly_element) in enumerate(plotly_obj):
                    if isinstance(update_obj, dict):
                        if i in update_obj:
                            update_element = update_obj[i]
                        else:
                            continue
                    else:
                        update_element = update_obj[i % len(update_obj)]
                    BaseFigure._perform_update(plotly_element, update_element)
        else:
            raise ValueError('Unexpected plotly object with type {typ}'.format(typ=type(plotly_obj)))

    @staticmethod
    def _index_is(iterable, val):
        if False:
            print('Hello World!')
        '\n        Return the index of a value in an iterable using object identity\n        (not object equality as is the case for list.index)\n\n        '
        index_list = [i for (i, curr_val) in enumerate(iterable) if curr_val is val]
        if not index_list:
            raise ValueError('Invalid value')
        return index_list[0]

    def _make_axis_spanning_layout_object(self, direction, shape):
        if False:
            return 10
        '\n        Convert a shape drawn on a plot or a subplot into one whose yref or xref\n        ends with " domain" and has coordinates so that the shape will seem to\n        extend infinitely in that dimension. This is useful for drawing lines or\n        boxes on a plot where one dimension of the shape will not move out of\n        bounds when moving the plot\'s view.\n        Note that the shape already added to the (sub)plot must have the\n        corresponding axis reference referring to an actual axis (e.g., \'x\',\n        \'y2\' etc. are accepted, but not \'paper\'). This will be the case if the\n        shape was added with "add_shape".\n        Shape must have the x0, x1, y0, y1 fields already initialized.\n        '
        if direction == 'vertical':
            ref = 'yref'
        elif direction == 'horizontal':
            ref = 'xref'
        else:
            raise ValueError("Bad direction: %s. Permissible values are 'vertical' and 'horizontal'." % (direction,))
        shape[ref] += ' domain'
        return shape

    def _process_multiple_axis_spanning_shapes(self, shape_args, row, col, shape_type, exclude_empty_subplots=True, annotation=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Add a shape or multiple shapes and call _make_axis_spanning_layout_object on\n        all the new shapes.\n        '
        if shape_type in ['vline', 'vrect']:
            direction = 'vertical'
        elif shape_type in ['hline', 'hrect']:
            direction = 'horizontal'
        else:
            raise ValueError("Bad shape_type %s, needs to be one of 'vline', 'hline', 'vrect', 'hrect'" % (shape_type,))
        if (row is not None or col is not None) and (not self._has_subplots()):
            row = None
            col = None
        n_shapes_before = len(self.layout['shapes'])
        n_annotations_before = len(self.layout['annotations'])
        (shape_kwargs, annotation_kwargs) = shapeannotation.split_dict_by_key_prefix(kwargs, 'annotation_')
        augmented_annotation = shapeannotation.axis_spanning_shape_annotation(annotation, shape_type, shape_args, annotation_kwargs)
        self.add_shape(row=row, col=col, exclude_empty_subplots=exclude_empty_subplots, **_combine_dicts([shape_args, shape_kwargs]))
        if augmented_annotation is not None:
            self.add_annotation(augmented_annotation, row=row, col=col, exclude_empty_subplots=exclude_empty_subplots, yref=shape_kwargs.get('yref', 'y'))
        for (layout_obj, n_layout_objs_before) in zip(['shapes', 'annotations'], [n_shapes_before, n_annotations_before]):
            n_layout_objs_after = len(self.layout[layout_obj])
            if n_layout_objs_after > n_layout_objs_before and (row is None and col is None):
                if self.layout[layout_obj][-1].xref is None:
                    self.layout[layout_obj][-1].update(xref='x')
                if self.layout[layout_obj][-1].yref is None:
                    self.layout[layout_obj][-1].update(yref='y')
            new_layout_objs = tuple(filter(lambda x: x is not None, [self._make_axis_spanning_layout_object(direction, self.layout[layout_obj][n]) for n in range(n_layout_objs_before, n_layout_objs_after)]))
            self.layout[layout_obj] = self.layout[layout_obj][:n_layout_objs_before] + new_layout_objs

    def add_vline(self, x, row='all', col='all', exclude_empty_subplots=True, annotation=None, **kwargs):
        if False:
            print('Hello World!')
        self._process_multiple_axis_spanning_shapes(dict(type='line', x0=x, x1=x, y0=0, y1=1), row, col, 'vline', exclude_empty_subplots=exclude_empty_subplots, annotation=annotation, **kwargs)
        return self
    add_vline.__doc__ = _axis_spanning_shapes_docstr('vline')

    def add_hline(self, y, row='all', col='all', exclude_empty_subplots=True, annotation=None, **kwargs):
        if False:
            while True:
                i = 10
        self._process_multiple_axis_spanning_shapes(dict(type='line', x0=0, x1=1, y0=y, y1=y), row, col, 'hline', exclude_empty_subplots=exclude_empty_subplots, annotation=annotation, **kwargs)
        return self
    add_hline.__doc__ = _axis_spanning_shapes_docstr('hline')

    def add_vrect(self, x0, x1, row='all', col='all', exclude_empty_subplots=True, annotation=None, **kwargs):
        if False:
            while True:
                i = 10
        self._process_multiple_axis_spanning_shapes(dict(type='rect', x0=x0, x1=x1, y0=0, y1=1), row, col, 'vrect', exclude_empty_subplots=exclude_empty_subplots, annotation=annotation, **kwargs)
        return self
    add_vrect.__doc__ = _axis_spanning_shapes_docstr('vrect')

    def add_hrect(self, y0, y1, row='all', col='all', exclude_empty_subplots=True, annotation=None, **kwargs):
        if False:
            return 10
        self._process_multiple_axis_spanning_shapes(dict(type='rect', x0=0, x1=1, y0=y0, y1=y1), row, col, 'hrect', exclude_empty_subplots=exclude_empty_subplots, annotation=annotation, **kwargs)
        return self
    add_hrect.__doc__ = _axis_spanning_shapes_docstr('hrect')

    def _has_subplots(self):
        if False:
            return 10
        'Returns True if figure contains subplots, otherwise it contains a\n        single plot and so this returns False.'
        return self._grid_ref is not None

    def _subplot_not_empty(self, xref, yref, selector='all'):
        if False:
            return 10
        '\n        xref: string representing the axis. Objects in the plot will be checked\n              for this xref (for layout objects) or xaxis (for traces) to\n              determine if they lie in a certain subplot.\n        yref: string representing the axis. Objects in the plot will be checked\n              for this yref (for layout objects) or yaxis (for traces) to\n              determine if they lie in a certain subplot.\n        selector: can be "all" or an iterable containing some combination of\n                  "traces", "shapes", "annotations", "images". Only the presence\n                  of objects specified in selector will be checked. So if\n                  ["traces","shapes"] is passed then a plot we be considered\n                  non-empty if it contains traces or shapes. If\n                  bool(selector) returns False, no checking is performed and\n                  this function returns True. If selector is True, it is\n                  converted to "all".\n        '
        if not selector:
            return True
        if selector is True:
            selector = 'all'
        if selector == 'all':
            selector = ['traces', 'shapes', 'annotations', 'images']
        ret = False
        for s in selector:
            if s == 'traces':
                obj = self.data
                xaxiskw = 'xaxis'
                yaxiskw = 'yaxis'
            elif s in ['shapes', 'annotations', 'images']:
                obj = self.layout[s]
                xaxiskw = 'xref'
                yaxiskw = 'yref'
            else:
                obj = None
            if obj:
                ret |= any((t == (xref, yref) for t in [('x' if d[xaxiskw] is None else d[xaxiskw], 'y' if d[yaxiskw] is None else d[yaxiskw]) for d in obj]))
        return ret

    def set_subplots(self, rows=None, cols=None, **make_subplots_args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add subplots to this figure. If the figure already contains subplots,\n        then this throws an error. Accepts any keyword arguments that\n        plotly.subplots.make_subplots accepts.\n        '
        if rows is not None:
            make_subplots_args['rows'] = rows
        if cols is not None:
            make_subplots_args['cols'] = cols
        if self._has_subplots():
            raise ValueError('This figure already has subplots.')
        return _subplots.make_subplots(figure=self, **make_subplots_args)

class BasePlotlyType(object):
    """
    BasePlotlyType is the base class for all objects in the trace, layout,
    and frame object hierarchies
    """
    _mapped_properties = {}
    _parent_path_str = ''
    _path_str = ''
    _valid_props = set()

    def __init__(self, plotly_name, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Construct a new BasePlotlyType\n\n        Parameters\n        ----------\n        plotly_name : str\n            The lowercase name of the plotly object\n        kwargs : dict\n            Invalid props/values to raise on\n        '
        self._skip_invalid = False
        self._validate = True
        self._process_kwargs(**kwargs)
        self._plotly_name = plotly_name
        self._compound_props = {}
        self._compound_array_props = {}
        self._orphan_props = {}
        self._parent = None
        self._change_callbacks = {}
        self.__validators = None

    def _get_validator(self, prop):
        if False:
            for i in range(10):
                print('nop')
        from .validator_cache import ValidatorCache
        return ValidatorCache.get_validator(self._path_str, prop)

    @property
    def _validators(self):
        if False:
            print('Hello World!')
        "\n        Validators used to be stored in a private _validators property. This was\n        eliminated when we switched to building validators on demand using the\n        _get_validator method.\n\n        This property returns a simple object that\n\n        Returns\n        -------\n        dict-like interface for accessing the object's validators\n        "
        obj = self
        if self.__validators is None:

            class ValidatorCompat(object):

                def __getitem__(self, item):
                    if False:
                        print('Hello World!')
                    return obj._get_validator(item)

                def __contains__(self, item):
                    if False:
                        for i in range(10):
                            print('nop')
                    return obj.__contains__(item)

                def __iter__(self):
                    if False:
                        while True:
                            i = 10
                    return iter(obj)

                def items(self):
                    if False:
                        i = 10
                        return i + 15
                    return [(k, self[k]) for k in self]
            self.__validators = ValidatorCompat()
        return self.__validators

    def _process_kwargs(self, **kwargs):
        if False:
            print('Hello World!')
        '\n        Process any extra kwargs that are not predefined as constructor params\n        '
        for (k, v) in kwargs.items():
            err = _check_path_in_prop_tree(self, k, error_cast=ValueError)
            if err is None:
                self[k] = v
            elif not self._validate:
                self[k] = v
            elif not self._skip_invalid:
                raise err

    @property
    def plotly_name(self):
        if False:
            print('Hello World!')
        '\n        The plotly name of the object\n\n        Returns\n        -------\n        str\n        '
        return self._plotly_name

    @property
    def _prop_descriptions(self):
        if False:
            while True:
                i = 10
        '\n        Formatted string containing all of this obejcts child properties\n        and their descriptions\n\n        Returns\n        -------\n        str\n        '
        raise NotImplementedError

    @property
    def _props(self):
        if False:
            i = 10
            return i + 15
        "\n        Dictionary used to store this object properties.  When the object\n        has a parent, this dict is retreived from the parent. When the\n        object does not have a parent, this dict is the object's\n        `_orphan_props` property\n\n        Note: Property will return None if the object has a parent and the\n        object's properties have not been initialized using the\n        `_init_props` method.\n\n        Returns\n        -------\n        dict|None\n        "
        if self.parent is None:
            return self._orphan_props
        else:
            return self.parent._get_child_props(self)

    def _get_child_props(self, child):
        if False:
            return 10
        '\n        Return properties dict for child\n\n        Parameters\n        ----------\n        child : BasePlotlyType\n\n        Returns\n        -------\n        dict\n        '
        if self._props is None:
            return None
        elif child.plotly_name in self:
            from _plotly_utils.basevalidators import CompoundValidator, CompoundArrayValidator
            validator = self._get_validator(child.plotly_name)
            if isinstance(validator, CompoundValidator):
                return self._props.get(child.plotly_name, None)
            elif isinstance(validator, CompoundArrayValidator):
                children = self[child.plotly_name]
                child_ind = BaseFigure._index_is(children, child)
                assert child_ind is not None
                children_props = self._props.get(child.plotly_name, None)
                return children_props[child_ind] if children_props is not None and len(children_props) > child_ind else None
        else:
            raise ValueError('Invalid child with name: %s' % child.plotly_name)

    def _init_props(self):
        if False:
            while True:
                i = 10
        "\n        Ensure that this object's properties dict has been initialized. When\n        the object has a parent, this ensures that the parent has an\n        initialized properties dict with this object's plotly_name as a key.\n\n        Returns\n        -------\n        None\n        "
        if self._props is not None:
            pass
        else:
            self._parent._init_child_props(self)

    def _init_child_props(self, child):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure that a properties dict has been initialized for a child object\n\n        Parameters\n        ----------\n        child : BasePlotlyType\n\n        Returns\n        -------\n        None\n        '
        self._init_props()
        if child.plotly_name in self._compound_props:
            if child.plotly_name not in self._props:
                self._props[child.plotly_name] = {}
        elif child.plotly_name in self._compound_array_props:
            children = self._compound_array_props[child.plotly_name]
            child_ind = BaseFigure._index_is(children, child)
            assert child_ind is not None
            if child.plotly_name not in self._props:
                self._props[child.plotly_name] = []
            children_list = self._props[child.plotly_name]
            while len(children_list) <= child_ind:
                children_list.append({})
        else:
            raise ValueError('Invalid child with name: %s' % child.plotly_name)

    def _get_child_prop_defaults(self, child):
        if False:
            return 10
        '\n        Return default properties dict for child\n\n        Parameters\n        ----------\n        child : BasePlotlyType\n\n        Returns\n        -------\n        dict\n        '
        if self._prop_defaults is None:
            return None
        elif child.plotly_name in self._compound_props:
            return self._prop_defaults.get(child.plotly_name, None)
        elif child.plotly_name in self._compound_array_props:
            children = self._compound_array_props[child.plotly_name]
            child_ind = BaseFigure._index_is(children, child)
            assert child_ind is not None
            children_props = self._prop_defaults.get(child.plotly_name, None)
            return children_props[child_ind] if children_props is not None and len(children_props) > child_ind else None
        else:
            raise ValueError('Invalid child with name: %s' % child.plotly_name)

    @property
    def _prop_defaults(self):
        if False:
            print('Hello World!')
        '\n        Return default properties dict\n\n        Returns\n        -------\n        dict\n        '
        if self.parent is None:
            return None
        else:
            return self.parent._get_child_prop_defaults(self)

    def _get_prop_validator(self, prop):
        if False:
            i = 10
            return i + 15
        '\n        Return the validator associated with the specified property\n\n        Parameters\n        ----------\n        prop: str\n            A property that exists in this object\n\n        Returns\n        -------\n        BaseValidator\n        '
        if prop in self._mapped_properties:
            prop_path = self._mapped_properties[prop]
            plotly_obj = self[prop_path[:-1]]
            prop = prop_path[-1]
        else:
            prop_path = BaseFigure._str_to_dict_path(prop)
            plotly_obj = self[prop_path[:-1]]
            prop = prop_path[-1]
        return plotly_obj._get_validator(prop)

    @property
    def parent(self):
        if False:
            print('Hello World!')
        "\n        Return the object's parent, or None if the object has no parent\n        Returns\n        -------\n        BasePlotlyType|BaseFigure\n        "
        return self._parent

    @property
    def figure(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reference to the top-level Figure or FigureWidget that this object\n        belongs to. None if the object does not belong to a Figure\n\n        Returns\n        -------\n        Union[BaseFigure, None]\n        '
        top_parent = self
        while top_parent is not None:
            if isinstance(top_parent, BaseFigure):
                break
            else:
                top_parent = top_parent.parent
        return top_parent

    def __reduce__(self):
        if False:
            return 10
        '\n        Custom implementation of reduce is used to support deep copying\n        and pickling\n        '
        props = self.to_plotly_json()
        return (self.__class__, (props,))

    def __getitem__(self, prop):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get item or nested item from object\n\n        Parameters\n        ----------\n        prop : str|tuple\n\n            If prop is the name of a property of this object, then the\n            property is returned.\n\n            If prop is a nested property path string (e.g. 'foo[1].bar'),\n            then a nested property is returned (e.g. obj['foo'][1]['bar'])\n\n            If prop is a path tuple (e.g. ('foo', 1, 'bar')), then a nested\n            property is returned (e.g. obj['foo'][1]['bar']).\n\n        Returns\n        -------\n        Any\n        "
        from _plotly_utils.basevalidators import CompoundValidator, CompoundArrayValidator, BaseDataValidator
        orig_prop = prop
        prop = BaseFigure._str_to_dict_path(prop)
        if prop and prop[0] in self._mapped_properties:
            prop = self._mapped_properties[prop[0]] + prop[1:]
            orig_prop = _remake_path_from_tuple(prop)
        if len(prop) == 1:
            prop = prop[0]
            if prop not in self._valid_props:
                self._raise_on_invalid_property_error(_error_to_raise=PlotlyKeyError)(prop)
            validator = self._get_validator(prop)
            if isinstance(validator, CompoundValidator):
                if self._compound_props.get(prop, None) is None:
                    self._compound_props[prop] = validator.data_class(_parent=self, plotly_name=prop)
                    self._compound_props[prop]._plotly_name = prop
                return validator.present(self._compound_props[prop])
            elif isinstance(validator, (CompoundArrayValidator, BaseDataValidator)):
                if self._compound_array_props.get(prop, None) is None:
                    if self._props is not None:
                        self._compound_array_props[prop] = [validator.data_class(_parent=self) for _ in self._props.get(prop, [])]
                    else:
                        self._compound_array_props[prop] = []
                return validator.present(self._compound_array_props[prop])
            elif self._props is not None and prop in self._props:
                return validator.present(self._props[prop])
            elif self._prop_defaults is not None:
                return validator.present(self._prop_defaults.get(prop, None))
            else:
                return None
        else:
            err = _check_path_in_prop_tree(self, orig_prop, error_cast=PlotlyKeyError)
            if err is not None:
                raise err
            res = self
            for p in prop:
                res = res[p]
            return res

    def __contains__(self, prop):
        if False:
            while True:
                i = 10
        "\n        Determine whether object contains a property or nested property\n\n        Parameters\n        ----------\n        prop : str|tuple\n            If prop is a simple string (e.g. 'foo'), then return true of the\n            object contains an element named 'foo'\n\n            If prop is a property path string (e.g. 'foo[0].bar'),\n            then return true if the obejct contains the nested elements for\n            each entry in the path string (e.g. 'bar' in obj['foo'][0])\n\n            If prop is a property path tuple (e.g. ('foo', 0, 'bar')),\n            then return true if the object contains the nested elements for\n            each entry in the path string (e.g. 'bar' in obj['foo'][0])\n\n        Returns\n        -------\n        bool\n        "
        prop = BaseFigure._str_to_dict_path(prop)
        if prop and prop[0] in self._mapped_properties:
            prop = self._mapped_properties[prop[0]] + prop[1:]
        obj = self
        for p in prop:
            if isinstance(p, int):
                if isinstance(obj, tuple) and 0 <= p < len(obj):
                    obj = obj[p]
                else:
                    return False
            elif hasattr(obj, '_valid_props') and p in obj._valid_props:
                obj = obj[p]
            else:
                return False
        return True

    def __setitem__(self, prop, value):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        prop : str\n            The name of a direct child of this object\n\n            Note: Setting nested properties using property path string or\n            property path tuples is not supported.\n        value\n            New property value\n\n        Returns\n        -------\n        None\n        '
        from _plotly_utils.basevalidators import CompoundValidator, CompoundArrayValidator, BaseDataValidator
        orig_prop = prop
        prop = BaseFigure._str_to_dict_path(prop)
        if len(prop) == 0:
            raise KeyError(orig_prop)
        if prop[0] in self._mapped_properties:
            prop = self._mapped_properties[prop[0]] + prop[1:]
        if len(prop) == 1:
            prop = prop[0]
            if self._validate:
                if prop not in self._valid_props:
                    self._raise_on_invalid_property_error()(prop)
                validator = self._get_validator(prop)
                if isinstance(validator, CompoundValidator):
                    self._set_compound_prop(prop, value)
                elif isinstance(validator, (CompoundArrayValidator, BaseDataValidator)):
                    self._set_array_prop(prop, value)
                else:
                    self._set_prop(prop, value)
            else:
                self._init_props()
                if isinstance(value, BasePlotlyType):
                    value = value.to_plotly_json()
                if isinstance(value, (list, tuple)) and value and isinstance(value[0], BasePlotlyType):
                    value = [v.to_plotly_json() if isinstance(v, BasePlotlyType) else v for v in value]
                self._props[prop] = value
                self._compound_props.pop(prop, None)
                self._compound_array_props.pop(prop, None)
        else:
            err = _check_path_in_prop_tree(self, orig_prop, error_cast=ValueError)
            if err is not None:
                raise err
            res = self
            for p in prop[:-1]:
                res = res[p]
            res._validate = self._validate
            res[prop[-1]] = value

    def __setattr__(self, prop, value):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        prop : str\n            The name of a direct child of this object\n        value\n            New property value\n        Returns\n        -------\n        None\n        '
        if prop.startswith('_') or hasattr(self, prop) or prop in self._valid_props:
            super(BasePlotlyType, self).__setattr__(prop, value)
        else:
            self._raise_on_invalid_property_error()(prop)

    def __iter__(self):
        if False:
            return 10
        "\n        Return an iterator over the object's properties\n        "
        res = list(self._valid_props)
        for prop in self._mapped_properties:
            res.append(prop)
        return iter(res)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        '\n        Test for equality\n\n        To be considered equal, `other` must have the same type as this object\n        and their `to_plotly_json` representaitons must be identical.\n\n        Parameters\n        ----------\n        other\n            The object to compare against\n\n        Returns\n        -------\n        bool\n        '
        if not isinstance(other, self.__class__):
            return False
        else:
            return BasePlotlyType._vals_equal(self._props if self._props is not None else {}, other._props if other._props is not None else {})

    @staticmethod
    def _build_repr_for_class(props, class_name, parent_path_str=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Helper to build representation string for a class\n\n        Parameters\n        ----------\n        class_name : str\n            Name of the class being represented\n        parent_path_str : str of None (default)\n            Name of the class's parent package to display\n        props : dict\n            Properties to unpack into the constructor\n\n        Returns\n        -------\n        str\n            The representation string\n        "
        from plotly.utils import ElidedPrettyPrinter
        if parent_path_str:
            class_name = parent_path_str + '.' + class_name
        if len(props) == 0:
            repr_str = class_name + '()'
        else:
            pprinter = ElidedPrettyPrinter(threshold=200, width=120)
            pprint_res = pprinter.pformat(props)
            body = '   ' + pprint_res[1:-1].replace('\n', '\n   ')
            repr_str = class_name + '({\n ' + body + '\n})'
        return repr_str

    def __repr__(self):
        if False:
            return 10
        '\n        Customize object representation when displayed in the\n        terminal/notebook\n        '
        from _plotly_utils.basevalidators import LiteralValidator
        props = self._props if self._props is not None else {}
        props = {p: v for (p, v) in props.items() if p in self._valid_props and (not isinstance(self._get_validator(p), LiteralValidator))}
        if 'template' in props:
            props['template'] = '...'
        repr_str = BasePlotlyType._build_repr_for_class(props=props, class_name=self.__class__.__name__, parent_path_str=self._parent_path_str)
        return repr_str

    def _raise_on_invalid_property_error(self, _error_to_raise=None):
        if False:
            while True:
                i = 10
        '\n        Returns a function that raises informative exception when invalid\n        property names are encountered. The _error_to_raise argument allows\n        specifying the exception to raise, which is ValueError if None.\n\n        Parameters\n        ----------\n        args : list[str]\n            List of property names that have already been determined to be\n            invalid\n\n        Raises\n        ------\n        ValueError by default, or _error_to_raise if not None\n        '
        if _error_to_raise is None:
            _error_to_raise = ValueError

        def _ret(*args):
            if False:
                i = 10
                return i + 15
            invalid_props = args
            if invalid_props:
                if len(invalid_props) == 1:
                    prop_str = 'property'
                    invalid_str = repr(invalid_props[0])
                else:
                    prop_str = 'properties'
                    invalid_str = repr(invalid_props)
                module_root = 'plotly.graph_objs.'
                if self._parent_path_str:
                    full_obj_name = module_root + self._parent_path_str + '.' + self.__class__.__name__
                else:
                    full_obj_name = module_root + self.__class__.__name__
                guessed_prop = None
                if len(invalid_props) == 1:
                    try:
                        guessed_prop = find_closest_string(invalid_props[0], self._valid_props)
                    except Exception:
                        pass
                guessed_prop_suggestion = ''
                if guessed_prop is not None:
                    guessed_prop_suggestion = 'Did you mean "%s"?' % (guessed_prop,)
                raise _error_to_raise('Invalid {prop_str} specified for object of type {full_obj_name}: {invalid_str}\n\n{guessed_prop_suggestion}\n\n    Valid properties:\n{prop_descriptions}\n{guessed_prop_suggestion}\n'.format(prop_str=prop_str, full_obj_name=full_obj_name, invalid_str=invalid_str, prop_descriptions=self._prop_descriptions, guessed_prop_suggestion=guessed_prop_suggestion))
        return _ret

    def update(self, dict1=None, overwrite=False, **kwargs):
        if False:
            print('Hello World!')
        '\n        Update the properties of an object with a dict and/or with\n        keyword arguments.\n\n        This recursively updates the structure of the original\n        object with the values in the input dict / keyword arguments.\n\n        Parameters\n        ----------\n        dict1 : dict\n            Dictionary of properties to be updated\n        overwrite: bool\n            If True, overwrite existing properties. If False, apply updates\n            to existing properties recursively, preserving existing\n            properties that are not specified in the update operation.\n        kwargs :\n            Keyword/value pair of properties to be updated\n\n        Returns\n        -------\n        BasePlotlyType\n            Updated plotly object\n        '
        if self.figure:
            with self.figure.batch_update():
                BaseFigure._perform_update(self, dict1, overwrite=overwrite)
                BaseFigure._perform_update(self, kwargs, overwrite=overwrite)
        else:
            BaseFigure._perform_update(self, dict1, overwrite=overwrite)
            BaseFigure._perform_update(self, kwargs, overwrite=overwrite)
        return self

    def pop(self, key, *args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove the value associated with the specified key and return it\n\n        Parameters\n        ----------\n        key: str\n            Property name\n        dflt\n            The default value to return if key was not found in object\n\n        Returns\n        -------\n        value\n            The removed value that was previously associated with key\n\n        Raises\n        ------\n        KeyError\n            If key is not in object and no dflt argument specified\n        '
        if key not in self and args:
            return args[0]
        elif key in self:
            val = self[key]
            self[key] = None
            return val
        else:
            raise KeyError(key)

    @property
    def _in_batch_mode(self):
        if False:
            i = 10
            return i + 15
        '\n        True if the object belongs to a figure that is currently in batch mode\n        Returns\n        -------\n        bool\n        '
        return self.parent and self.parent._in_batch_mode

    def _set_prop(self, prop, val):
        if False:
            i = 10
            return i + 15
        '\n        Set the value of a simple property\n\n        Parameters\n        ----------\n        prop : str\n            Name of a simple (non-compound, non-array) property\n        val\n            The new property value\n\n        Returns\n        -------\n        Any\n            The coerced assigned value\n        '
        if val is Undefined:
            return
        validator = self._get_validator(prop)
        try:
            val = validator.validate_coerce(val)
        except ValueError as err:
            if self._skip_invalid:
                return
            else:
                raise err
        if val is None:
            if self._props and prop in self._props:
                if not self._in_batch_mode:
                    self._props.pop(prop)
                self._send_prop_set(prop, val)
        else:
            self._init_props()
            if prop not in self._props or not BasePlotlyType._vals_equal(self._props[prop], val):
                if not self._in_batch_mode:
                    self._props[prop] = val
                self._send_prop_set(prop, val)
        return val

    def _set_compound_prop(self, prop, val):
        if False:
            print('Hello World!')
        '\n        Set the value of a compound property\n\n        Parameters\n        ----------\n        prop : str\n            Name of a compound property\n        val\n            The new property value\n\n        Returns\n        -------\n        BasePlotlyType\n            The coerced assigned object\n        '
        if val is Undefined:
            return
        validator = self._get_validator(prop)
        val = validator.validate_coerce(val, skip_invalid=self._skip_invalid)
        curr_val = self._compound_props.get(prop, None)
        if curr_val is not None:
            curr_dict_val = deepcopy(curr_val._props)
        else:
            curr_dict_val = None
        if val is not None:
            new_dict_val = deepcopy(val._props)
        else:
            new_dict_val = None
        if not self._in_batch_mode:
            if not new_dict_val:
                if self._props and prop in self._props:
                    self._props.pop(prop)
            else:
                self._init_props()
                self._props[prop] = new_dict_val
        if not BasePlotlyType._vals_equal(curr_dict_val, new_dict_val):
            self._send_prop_set(prop, new_dict_val)
        if isinstance(val, BasePlotlyType):
            val._parent = self
            val._orphan_props.clear()
        if curr_val is not None:
            if curr_dict_val is not None:
                curr_val._orphan_props.update(curr_dict_val)
            curr_val._parent = None
        self._compound_props[prop] = val
        return val

    def _set_array_prop(self, prop, val):
        if False:
            print('Hello World!')
        '\n        Set the value of a compound property\n\n        Parameters\n        ----------\n        prop : str\n            Name of a compound property\n        val\n            The new property value\n\n        Returns\n        -------\n        tuple[BasePlotlyType]\n            The coerced assigned object\n        '
        if val is Undefined:
            return
        validator = self._get_validator(prop)
        val = validator.validate_coerce(val, skip_invalid=self._skip_invalid)
        curr_val = self._compound_array_props.get(prop, None)
        if curr_val is not None:
            curr_dict_vals = [deepcopy(cv._props) for cv in curr_val]
        else:
            curr_dict_vals = None
        if val is not None:
            new_dict_vals = [deepcopy(nv._props) for nv in val]
        else:
            new_dict_vals = None
        if not self._in_batch_mode:
            if not new_dict_vals:
                if self._props and prop in self._props:
                    self._props.pop(prop)
            else:
                self._init_props()
                self._props[prop] = new_dict_vals
        if not BasePlotlyType._vals_equal(curr_dict_vals, new_dict_vals):
            self._send_prop_set(prop, new_dict_vals)
        if val is not None:
            for v in val:
                v._orphan_props.clear()
                v._parent = self
        if curr_val is not None:
            for (cv, cv_dict) in zip(curr_val, curr_dict_vals):
                if cv_dict is not None:
                    cv._orphan_props.update(cv_dict)
                cv._parent = None
        self._compound_array_props[prop] = val
        return val

    def _send_prop_set(self, prop_path_str, val):
        if False:
            print('Hello World!')
        "\n        Notify parent that a property has been set to a new value\n\n        Parameters\n        ----------\n        prop_path_str : str\n            Property path string (e.g. 'foo[0].bar') of property that\n            was set, relative to this object\n        val\n            New value for property. Either a simple value, a dict,\n            or a tuple of dicts. This should *not* be a BasePlotlyType object.\n\n        Returns\n        -------\n        None\n        "
        raise NotImplementedError()

    def _prop_set_child(self, child, prop_path_str, val):
        if False:
            print('Hello World!')
        "\n        Propagate property setting notification from child to parent\n\n        Parameters\n        ----------\n        child : BasePlotlyType\n            Child object\n        prop_path_str : str\n            Property path string (e.g. 'foo[0].bar') of property that\n            was set, relative to `child`\n        val\n            New value for property. Either a simple value, a dict,\n            or a tuple of dicts. This should *not* be a BasePlotlyType object.\n\n        Returns\n        -------\n        None\n        "
        child_prop_val = getattr(self, child.plotly_name)
        if isinstance(child_prop_val, (list, tuple)):
            child_ind = BaseFigure._index_is(child_prop_val, child)
            obj_path = '{child_name}.{child_ind}.{prop}'.format(child_name=child.plotly_name, child_ind=child_ind, prop=prop_path_str)
        else:
            obj_path = '{child_name}.{prop}'.format(child_name=child.plotly_name, prop=prop_path_str)
        self._send_prop_set(obj_path, val)

    def _restyle_child(self, child, prop, val):
        if False:
            return 10
        '\n        Propagate _restyle_child to parent\n\n        Note: This method must match the name and signature of the\n        corresponding method on BaseFigure\n        '
        self._prop_set_child(child, prop, val)

    def _relayout_child(self, child, prop, val):
        if False:
            while True:
                i = 10
        '\n        Propagate _relayout_child to parent\n\n        Note: This method must match the name and signature of the\n        corresponding method on BaseFigure\n        '
        self._prop_set_child(child, prop, val)

    def _dispatch_change_callbacks(self, changed_paths):
        if False:
            while True:
                i = 10
        '\n        Execute the appropriate change callback functions given a set of\n        changed property path tuples\n\n        Parameters\n        ----------\n        changed_paths : set[tuple[int|str]]\n\n        Returns\n        -------\n        None\n        '
        for (prop_path_tuples, callbacks) in self._change_callbacks.items():
            common_paths = changed_paths.intersection(set(prop_path_tuples))
            if common_paths:
                callback_args = [self[cb_path] for cb_path in prop_path_tuples]
                for callback in callbacks:
                    callback(self, *callback_args)

    def on_change(self, callback, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Register callback function to be called when certain properties or\n        subproperties of this object are modified.\n\n        Callback will be invoked whenever ANY of these properties is\n        modified. Furthermore, the callback will only be invoked once even\n        if multiple properties are modified during the same restyle /\n        relayout / update operation.\n\n        Parameters\n        ----------\n        callback : function\n            Function that accepts 1 + len(`args`) parameters. First parameter\n            is this object. Second through last parameters are the\n            property / subpropery values referenced by args.\n        args : list[str|tuple[int|str]]\n            List of property references where each reference may be one of:\n\n              1) A property name string (e.g. \'foo\') for direct properties\n              2) A property path string (e.g. \'foo[0].bar\') for\n                 subproperties\n              3) A property path tuple (e.g. (\'foo\', 0, \'bar\')) for\n                 subproperties\n\n        append : bool\n            True if callback should be appended to previously registered\n            callback on the same properties, False if callback should replace\n            previously registered callbacks on the same properties. Defaults\n            to False.\n\n        Examples\n        --------\n\n        Register callback that prints out the range extents of the xaxis and\n        yaxis whenever either either of them changes.\n\n        >>> import plotly.graph_objects as go\n        >>> fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 0]))\n        >>> fig.layout.on_change(\n        ...   lambda obj, xrange, yrange: print("%s-%s" % (xrange, yrange)),\n        ...   (\'xaxis\', \'range\'), (\'yaxis\', \'range\'))\n\n\n        Returns\n        -------\n        None\n        '
        if not self.figure:
            class_name = self.__class__.__name__
            msg = '\n{class_name} object is not a descendant of a Figure.\non_change callbacks are not supported in this case.\n'.format(class_name=class_name)
            raise ValueError(msg)
        if len(args) == 0:
            raise ValueError('At least one change property must be specified')
        invalid_args = [arg for arg in args if arg not in self]
        if invalid_args:
            raise ValueError('Invalid property specification(s): %s' % invalid_args)
        append = kwargs.get('append', False)
        arg_tuples = tuple([BaseFigure._str_to_dict_path(a) for a in args])
        if arg_tuples not in self._change_callbacks or not append:
            self._change_callbacks[arg_tuples] = []
        self._change_callbacks[arg_tuples].append(callback)

    def to_plotly_json(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return plotly JSON representation of object as a Python dict\n\n        Note: May include some JSON-invalid data types, use the `PlotlyJSONEncoder` util\n        or the `to_json` method to encode to a string.\n\n        Returns\n        -------\n        dict\n        '
        return deepcopy(self._props if self._props is not None else {})

    def to_json(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Convert object to a JSON string representation\n\n        Parameters\n        ----------\n        validate: bool (default True)\n            True if the object should be validated before being converted to\n            JSON, False otherwise.\n\n        pretty: bool (default False)\n            True if JSON representation should be pretty-printed, False if\n            representation should be as compact as possible.\n\n        remove_uids: bool (default True)\n            True if trace UIDs should be omitted from the JSON representation\n\n        engine: str (default None)\n            The JSON encoding engine to use. One of:\n              - "json" for an encoder based on the built-in Python json module\n              - "orjson" for a fast encoder the requires the orjson package\n            If not specified, the default encoder is set to the current value of\n            plotly.io.json.config.default_encoder.\n\n        Returns\n        -------\n        str\n            Representation of object as a JSON string\n        '
        import plotly.io as pio
        return pio.to_json(self, *args, **kwargs)

    @staticmethod
    def _vals_equal(v1, v2):
        if False:
            i = 10
            return i + 15
        '\n        Recursive equality function that handles nested dicts / tuples / lists\n        that contain numpy arrays.\n\n        v1\n            First value to compare\n        v2\n            Second value to compare\n\n        Returns\n        -------\n        bool\n            True if v1 and v2 are equal, False otherwise\n        '
        np = get_module('numpy', should_load=False)
        if np is not None and (isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray)):
            return np.array_equal(v1, v2)
        elif isinstance(v1, (list, tuple)):
            return isinstance(v2, (list, tuple)) and len(v1) == len(v2) and all((BasePlotlyType._vals_equal(e1, e2) for (e1, e2) in zip(v1, v2)))
        elif isinstance(v1, dict):
            return isinstance(v2, dict) and set(v1.keys()) == set(v2.keys()) and all((BasePlotlyType._vals_equal(v1[k], v2[k]) for k in v1))
        else:
            return v1 == v2

class BaseLayoutHierarchyType(BasePlotlyType):
    """
    Base class for all types in the layout hierarchy
    """

    @property
    def _parent_path_str(self):
        if False:
            i = 10
            return i + 15
        pass

    def __init__(self, plotly_name, **kwargs):
        if False:
            return 10
        super(BaseLayoutHierarchyType, self).__init__(plotly_name, **kwargs)

    def _send_prop_set(self, prop_path_str, val):
        if False:
            i = 10
            return i + 15
        if self.parent:
            self.parent._relayout_child(self, prop_path_str, val)

class BaseLayoutType(BaseLayoutHierarchyType):
    """
    Base class for the layout type. The Layout class itself is a
    code-generated subclass.
    """

    @property
    def _subplotid_validators(self):
        if False:
            while True:
                i = 10
        '\n        dict of validator classes for each subplot type\n\n        Returns\n        -------\n        dict\n        '
        raise NotImplementedError()

    def _subplot_re_match(self, prop):
        if False:
            return 10
        raise NotImplementedError()

    def __init__(self, plotly_name, **kwargs):
        if False:
            return 10
        "\n        Construct a new BaseLayoutType object\n\n        Parameters\n        ----------\n        plotly_name : str\n            Name of the object (should always be 'layout')\n        kwargs : dict[str, any]\n            Properties that were not recognized by the Layout subclass.\n            These are subplot identifiers (xaxis2, geo4, etc.) or they are\n            invalid properties.\n        "
        assert plotly_name == 'layout'
        super(BaseLayoutHierarchyType, self).__init__(plotly_name)
        self._subplotid_props = set()
        self._process_kwargs(**kwargs)

    def _process_kwargs(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Process any extra kwargs that are not predefined as constructor params\n        '
        unknown_kwargs = {k: v for (k, v) in kwargs.items() if not self._subplot_re_match(k)}
        super(BaseLayoutHierarchyType, self)._process_kwargs(**unknown_kwargs)
        subplot_kwargs = {k: v for (k, v) in kwargs.items() if self._subplot_re_match(k)}
        for (prop, value) in subplot_kwargs.items():
            self._set_subplotid_prop(prop, value)

    def _set_subplotid_prop(self, prop, value):
        if False:
            while True:
                i = 10
        '\n        Set a subplot property on the layout\n\n        Parameters\n        ----------\n        prop : str\n            A valid subplot property\n        value\n            Subplot value\n        '
        match = self._subplot_re_match(prop)
        subplot_prop = match.group(1)
        suffix_digit = int(match.group(2))
        if suffix_digit == 0:
            raise TypeError('Subplot properties may only be suffixed by an integer >= 1\nReceived {k}'.format(k=prop))
        if suffix_digit == 1:
            prop = subplot_prop
        if prop not in self._valid_props:
            self._valid_props.add(prop)
        self._set_compound_prop(prop, value)
        self._subplotid_props.add(prop)

    def _strip_subplot_suffix_of_1(self, prop):
        if False:
            print('Hello World!')
        "\n        Strip the suffix for subplot property names that have a suffix of 1.\n        All other properties are returned unchanged\n\n        e.g. 'xaxis1' -> 'xaxis'\n\n        Parameters\n        ----------\n        prop : str|tuple\n\n        Returns\n        -------\n        str|tuple\n        "
        prop_tuple = BaseFigure._str_to_dict_path(prop)
        if len(prop_tuple) != 1 or not isinstance(prop_tuple[0], str):
            return prop
        else:
            prop = prop_tuple[0]
        match = self._subplot_re_match(prop)
        if match:
            subplot_prop = match.group(1)
            suffix_digit = int(match.group(2))
            if subplot_prop and suffix_digit == 1:
                prop = subplot_prop
        return prop

    def _get_prop_validator(self, prop):
        if False:
            print('Hello World!')
        '\n        Custom _get_prop_validator that handles subplot properties\n        '
        prop = self._strip_subplot_suffix_of_1(prop)
        return super(BaseLayoutHierarchyType, self)._get_prop_validator(prop)

    def __getattr__(self, prop):
        if False:
            for i in range(10):
                print('nop')
        '\n        Custom __getattr__ that handles dynamic subplot properties\n        '
        prop = self._strip_subplot_suffix_of_1(prop)
        if prop != '_subplotid_props' and prop in self._subplotid_props:
            validator = self._get_validator(prop)
            return validator.present(self._compound_props[prop])
        else:
            return super(BaseLayoutHierarchyType, self).__getattribute__(prop)

    def __getitem__(self, prop):
        if False:
            i = 10
            return i + 15
        '\n        Custom __getitem__ that handles dynamic subplot properties\n        '
        prop = self._strip_subplot_suffix_of_1(prop)
        return super(BaseLayoutHierarchyType, self).__getitem__(prop)

    def __contains__(self, prop):
        if False:
            print('Hello World!')
        '\n        Custom __contains__ that handles dynamic subplot properties\n        '
        prop = self._strip_subplot_suffix_of_1(prop)
        return super(BaseLayoutHierarchyType, self).__contains__(prop)

    def __setitem__(self, prop, value):
        if False:
            return 10
        '\n        Custom __setitem__ that handles dynamic subplot properties\n        '
        prop_tuple = BaseFigure._str_to_dict_path(prop)
        if len(prop_tuple) != 1 or not isinstance(prop_tuple[0], str):
            super(BaseLayoutHierarchyType, self).__setitem__(prop, value)
            return
        else:
            prop = prop_tuple[0]
        match = self._subplot_re_match(prop)
        if match is None:
            super(BaseLayoutHierarchyType, self).__setitem__(prop, value)
        else:
            self._set_subplotid_prop(prop, value)

    def __setattr__(self, prop, value):
        if False:
            i = 10
            return i + 15
        '\n        Custom __setattr__ that handles dynamic subplot properties\n        '
        match = self._subplot_re_match(prop)
        if match is None:
            super(BaseLayoutHierarchyType, self).__setattr__(prop, value)
        else:
            self._set_subplotid_prop(prop, value)

    def __dir__(self):
        if False:
            while True:
                i = 10
        '\n        Custom __dir__ that handles dynamic subplot properties\n        '
        return list(super(BaseLayoutHierarchyType, self).__dir__()) + sorted(self._subplotid_props)

class BaseTraceHierarchyType(BasePlotlyType):
    """
    Base class for all types in the trace hierarchy
    """

    def __init__(self, plotly_name, **kwargs):
        if False:
            while True:
                i = 10
        super(BaseTraceHierarchyType, self).__init__(plotly_name, **kwargs)

    def _send_prop_set(self, prop_path_str, val):
        if False:
            i = 10
            return i + 15
        if self.parent:
            self.parent._restyle_child(self, prop_path_str, val)

class BaseTraceType(BaseTraceHierarchyType):
    """
    Base class for the all trace types.

    Specific trace type classes (Scatter, Bar, etc.) are code generated as
    subclasses of this class.
    """

    def __init__(self, plotly_name, **kwargs):
        if False:
            while True:
                i = 10
        super(BaseTraceHierarchyType, self).__init__(plotly_name, **kwargs)
        self._hover_callbacks = []
        self._unhover_callbacks = []
        self._click_callbacks = []
        self._select_callbacks = []
        self._deselect_callbacks = []
        self._trace_ind = None

    @property
    def uid(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    @uid.setter
    def uid(self, val):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def on_hover(self, callback, append=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Register function to be called when the user hovers over one or more\n        points in this trace\n\n        Note: Callbacks will only be triggered when the trace belongs to a\n        instance of plotly.graph_objs.FigureWidget and it is displayed in an\n        ipywidget context. Callbacks will not be triggered on figures\n        that are displayed using plot/iplot.\n\n        Parameters\n        ----------\n        callback\n            Callable function that accepts 3 arguments\n\n            - this trace\n            - plotly.callbacks.Points object\n            - plotly.callbacks.InputDeviceState object\n\n        append : bool\n            If False (the default), this callback replaces any previously\n            defined on_hover callbacks for this trace. If True,\n            this callback is appended to the list of any previously defined\n            callbacks.\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n\n        >>> import plotly.graph_objects as go\n        >>> from plotly.callbacks import Points, InputDeviceState\n        >>> points, state = Points(), InputDeviceState()\n\n        >>> def hover_fn(trace, points, state):\n        ...     inds = points.point_inds\n        ...     # Do something\n\n        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])\n        >>> trace.on_hover(hover_fn)\n\n        Note: The creation of the `points` and `state` objects is optional,\n        it's simply a convenience to help the text editor perform completion\n        on the arguments inside `hover_fn`\n        "
        if not append:
            del self._hover_callbacks[:]
        if callback:
            self._hover_callbacks.append(callback)

    def _dispatch_on_hover(self, points, state):
        if False:
            print('Hello World!')
        '\n        Dispatch points and device state all all hover callbacks\n        '
        for callback in self._hover_callbacks:
            callback(self, points, state)

    def on_unhover(self, callback, append=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Register function to be called when the user unhovers away from one\n        or more points in this trace.\n\n        Note: Callbacks will only be triggered when the trace belongs to a\n        instance of plotly.graph_objs.FigureWidget and it is displayed in an\n        ipywidget context. Callbacks will not be triggered on figures\n        that are displayed using plot/iplot.\n\n        Parameters\n        ----------\n        callback\n            Callable function that accepts 3 arguments\n\n            - this trace\n            - plotly.callbacks.Points object\n            - plotly.callbacks.InputDeviceState object\n\n        append : bool\n            If False (the default), this callback replaces any previously\n            defined on_unhover callbacks for this trace. If True,\n            this callback is appended to the list of any previously defined\n            callbacks.\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n\n        >>> import plotly.graph_objects as go\n        >>> from plotly.callbacks import Points, InputDeviceState\n        >>> points, state = Points(), InputDeviceState()\n\n        >>> def unhover_fn(trace, points, state):\n        ...     inds = points.point_inds\n        ...     # Do something\n\n        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])\n        >>> trace.on_unhover(unhover_fn)\n\n        Note: The creation of the `points` and `state` objects is optional,\n        it's simply a convenience to help the text editor perform completion\n        on the arguments inside `unhover_fn`\n        "
        if not append:
            del self._unhover_callbacks[:]
        if callback:
            self._unhover_callbacks.append(callback)

    def _dispatch_on_unhover(self, points, state):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dispatch points and device state all all hover callbacks\n        '
        for callback in self._unhover_callbacks:
            callback(self, points, state)

    def on_click(self, callback, append=False):
        if False:
            return 10
        "\n        Register function to be called when the user clicks on one or more\n        points in this trace.\n\n        Note: Callbacks will only be triggered when the trace belongs to a\n        instance of plotly.graph_objs.FigureWidget and it is displayed in an\n        ipywidget context. Callbacks will not be triggered on figures\n        that are displayed using plot/iplot.\n\n        Parameters\n        ----------\n        callback\n            Callable function that accepts 3 arguments\n\n            - this trace\n            - plotly.callbacks.Points object\n            - plotly.callbacks.InputDeviceState object\n\n        append : bool\n            If False (the default), this callback replaces any previously\n            defined on_click callbacks for this trace. If True,\n            this callback is appended to the list of any previously defined\n            callbacks.\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n\n        >>> import plotly.graph_objects as go\n        >>> from plotly.callbacks import Points, InputDeviceState\n        >>> points, state = Points(), InputDeviceState()\n\n        >>> def click_fn(trace, points, state):\n        ...     inds = points.point_inds\n        ...     # Do something\n\n        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])\n        >>> trace.on_click(click_fn)\n\n        Note: The creation of the `points` and `state` objects is optional,\n        it's simply a convenience to help the text editor perform completion\n        on the arguments inside `click_fn`\n        "
        if not append:
            del self._click_callbacks[:]
        if callback:
            self._click_callbacks.append(callback)

    def _dispatch_on_click(self, points, state):
        if False:
            while True:
                i = 10
        '\n        Dispatch points and device state all all hover callbacks\n        '
        for callback in self._click_callbacks:
            callback(self, points, state)

    def on_selection(self, callback, append=False):
        if False:
            return 10
        "\n        Register function to be called when the user selects one or more\n        points in this trace.\n\n        Note: Callbacks will only be triggered when the trace belongs to a\n        instance of plotly.graph_objs.FigureWidget and it is displayed in an\n        ipywidget context. Callbacks will not be triggered on figures\n        that are displayed using plot/iplot.\n\n        Parameters\n        ----------\n        callback\n            Callable function that accepts 4 arguments\n\n            - this trace\n            - plotly.callbacks.Points object\n            - plotly.callbacks.BoxSelector or plotly.callbacks.LassoSelector\n\n        append : bool\n            If False (the default), this callback replaces any previously\n            defined on_selection callbacks for this trace. If True,\n            this callback is appended to the list of any previously defined\n            callbacks.\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n\n        >>> import plotly.graph_objects as go\n        >>> from plotly.callbacks import Points\n        >>> points = Points()\n\n        >>> def selection_fn(trace, points, selector):\n        ...     inds = points.point_inds\n        ...     # Do something\n\n        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])\n        >>> trace.on_selection(selection_fn)\n\n        Note: The creation of the `points` object is optional,\n        it's simply a convenience to help the text editor perform completion\n        on the `points` arguments inside `selection_fn`\n        "
        if not append:
            del self._select_callbacks[:]
        if callback:
            self._select_callbacks.append(callback)

    def _dispatch_on_selection(self, points, selector):
        if False:
            i = 10
            return i + 15
        '\n        Dispatch points and selector info to selection callbacks\n        '
        if 'selectedpoints' in self:
            self.selectedpoints = points.point_inds
        for callback in self._select_callbacks:
            callback(self, points, selector)

    def on_deselect(self, callback, append=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Register function to be called when the user deselects points\n        in this trace using doubleclick.\n\n        Note: Callbacks will only be triggered when the trace belongs to a\n        instance of plotly.graph_objs.FigureWidget and it is displayed in an\n        ipywidget context. Callbacks will not be triggered on figures\n        that are displayed using plot/iplot.\n\n        Parameters\n        ----------\n        callback\n            Callable function that accepts 3 arguments\n\n            - this trace\n            - plotly.callbacks.Points object\n\n        append : bool\n            If False (the default), this callback replaces any previously\n            defined on_deselect callbacks for this trace. If True,\n            this callback is appended to the list of any previously defined\n            callbacks.\n\n        Returns\n        -------\n        None\n\n        Examples\n        --------\n\n        >>> import plotly.graph_objects as go\n        >>> from plotly.callbacks import Points\n        >>> points = Points()\n\n        >>> def deselect_fn(trace, points):\n        ...     inds = points.point_inds\n        ...     # Do something\n\n        >>> trace = go.Scatter(x=[1, 2], y=[3, 0])\n        >>> trace.on_deselect(deselect_fn)\n\n        Note: The creation of the `points` object is optional,\n        it's simply a convenience to help the text editor perform completion\n        on the `points` arguments inside `selection_fn`\n        "
        if not append:
            del self._deselect_callbacks[:]
        if callback:
            self._deselect_callbacks.append(callback)

    def _dispatch_on_deselect(self, points):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dispatch points info to deselection callbacks\n        '
        if 'selectedpoints' in self:
            self.selectedpoints = None
        for callback in self._deselect_callbacks:
            callback(self, points)

class BaseFrameHierarchyType(BasePlotlyType):
    """
    Base class for all types in the trace hierarchy
    """

    def __init__(self, plotly_name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(BaseFrameHierarchyType, self).__init__(plotly_name, **kwargs)

    def _send_prop_set(self, prop_path_str, val):
        if False:
            i = 10
            return i + 15
        pass

    def _restyle_child(self, child, key_path_str, val):
        if False:
            return 10
        pass

    def on_change(self, callback, *args):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Change callbacks are not supported on Frames')

    def _get_child_props(self, child):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the properties dict for a child trace or child layout\n\n        Note: this method must match the name/signature of one on\n        BasePlotlyType\n\n        Parameters\n        ----------\n        child : BaseTraceType | BaseLayoutType\n\n        Returns\n        -------\n        dict\n        '
        try:
            trace_index = BaseFigure._index_is(self.data, child)
        except ValueError:
            trace_index = None
        if trace_index is not None:
            if 'data' in self._props:
                return self._props['data'][trace_index]
            else:
                return None
        elif child is self.layout:
            return self._props.get('layout', None)
        else:
            raise ValueError('Unrecognized child: %s' % child)