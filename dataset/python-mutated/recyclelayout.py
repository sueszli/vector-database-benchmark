"""
RecycleLayout
=============

.. versionadded:: 1.10.0

.. warning::
    This module is highly experimental, its API may change in the future and
    the documentation is not complete at this time.
"""
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior
from kivy.uix.layout import Layout
from kivy.properties import ObjectProperty, StringProperty, ReferenceListProperty, NumericProperty
from kivy.factory import Factory
__all__ = ('RecycleLayout',)

class RecycleLayout(RecycleLayoutManagerBehavior, Layout):
    """
    RecycleLayout provides the default layout for RecycleViews.
    """
    default_width = NumericProperty(100, allownone=True)
    'Default width for items\n\n    :attr:`default_width` is a NumericProperty and default to 100\n    '
    default_height = NumericProperty(100, allownone=True)
    'Default height for items\n\n    :attr:`default_height` is a :class:`~kivy.properties.NumericProperty` and\n    default to 100.\n    '
    default_size = ReferenceListProperty(default_width, default_height)
    'size (width, height). Each value can be None.\n\n    :attr:`default_size` is an :class:`~kivy.properties.ReferenceListProperty`\n    to [:attr:`default_width`, :attr:`default_height`].\n    '
    default_size_hint_x = NumericProperty(None, allownone=True)
    'Default size_hint_x for items\n\n    :attr:`default_size_hint_x` is a :class:`~kivy.properties.NumericProperty`\n    and default to None.\n    '
    default_size_hint_y = NumericProperty(None, allownone=True)
    'Default size_hint_y for items\n\n    :attr:`default_size_hint_y` is a :class:`~kivy.properties.NumericProperty`\n    and default to None.\n    '
    default_size_hint = ReferenceListProperty(default_size_hint_x, default_size_hint_y)
    'size (width, height). Each value can be None.\n\n    :attr:`default_size_hint` is an\n    :class:`~kivy.properties.ReferenceListProperty` to\n    [:attr:`default_size_hint_x`, :attr:`default_size_hint_y`].\n    '
    key_size = StringProperty(None, allownone=True)
    'If set, which key in the dict should be used to set the size property of\n    the item.\n\n    :attr:`key_size` is a :class:`~kivy.properties.StringProperty` and defaults\n    to None.\n    '
    key_size_hint = StringProperty(None, allownone=True)
    'If set, which key in the dict should be used to set the size_hint\n    property of the item.\n\n    :attr:`key_size_hint` is a :class:`~kivy.properties.StringProperty` and\n    defaults to None.\n    '
    key_size_hint_min = StringProperty(None, allownone=True)
    'If set, which key in the dict should be used to set the size_hint_min\n    property of the item.\n\n    :attr:`key_size_hint_min` is a :class:`~kivy.properties.StringProperty` and\n    defaults to None.\n    '
    default_size_hint_x_min = NumericProperty(None, allownone=True)
    'Default value for size_hint_x_min of items\n\n    :attr:`default_pos_hint_x_min` is a\n    :class:`~kivy.properties.NumericProperty` and defaults to None.\n    '
    default_size_hint_y_min = NumericProperty(None, allownone=True)
    'Default value for size_hint_y_min of items\n\n    :attr:`default_pos_hint_y_min` is a\n    :class:`~kivy.properties.NumericProperty` and defaults to None.\n    '
    default_size_hint_min = ReferenceListProperty(default_size_hint_x_min, default_size_hint_y_min)
    'Default value for size_hint_min of items\n\n    :attr:`default_size_min` is a\n    :class:`~kivy.properties.ReferenceListProperty` to\n    [:attr:`default_size_hint_x_min`, :attr:`default_size_hint_y_min`].\n    '
    key_size_hint_max = StringProperty(None, allownone=True)
    'If set, which key in the dict should be used to set the size_hint_max\n    property of the item.\n\n    :attr:`key_size_hint_max` is a :class:`~kivy.properties.StringProperty` and\n    defaults to None.\n    '
    default_size_hint_x_max = NumericProperty(None, allownone=True)
    'Default value for size_hint_x_max of items\n\n    :attr:`default_pos_hint_x_max` is a\n    :class:`~kivy.properties.NumericProperty` and defaults to None.\n    '
    default_size_hint_y_max = NumericProperty(None, allownone=True)
    'Default value for size_hint_y_max of items\n\n    :attr:`default_pos_hint_y_max` is a\n    :class:`~kivy.properties.NumericProperty` and defaults to None.\n    '
    default_size_hint_max = ReferenceListProperty(default_size_hint_x_max, default_size_hint_y_max)
    'Default value for size_hint_max of items\n\n    :attr:`default_size_max` is a\n    :class:`~kivy.properties.ReferenceListProperty` to\n    [:attr:`default_size_hint_x_max`, :attr:`default_size_hint_y_max`].\n    '
    default_pos_hint = ObjectProperty({})
    'Default pos_hint value for items\n\n    :attr:`default_pos_hint` is a :class:`~kivy.properties.DictProperty` and\n    defaults to {}.\n    '
    key_pos_hint = StringProperty(None, allownone=True)
    'If set, which key in the dict should be used to set the pos_hint of\n    items.\n\n    :attr:`key_pos_hint` is a :class:`~kivy.properties.StringProperty` and\n    defaults to None.\n    '
    initial_width = NumericProperty(100)
    'Initial width for the items.\n\n    :attr:`initial_width` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 100.\n    '
    initial_height = NumericProperty(100)
    'Initial height for the items.\n\n    :attr:`initial_height` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 100.\n    '
    initial_size = ReferenceListProperty(initial_width, initial_height)
    'Initial size of items\n\n    :attr:`initial_size` is a :class:`~kivy.properties.ReferenceListProperty`\n    to [:attr:`initial_width`, :attr:`initial_height`].\n    '
    view_opts = []
    _size_needs_update = False
    _changed_views = []
    view_indices = {}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.view_indices = {}
        self._updated_views = []
        self._trigger_layout = self._catch_layout_trigger
        super(RecycleLayout, self).__init__(**kwargs)

    def attach_recycleview(self, rv):
        if False:
            i = 10
            return i + 15
        super(RecycleLayout, self).attach_recycleview(rv)
        if rv:
            fbind = self.fbind
            fbind('default_size', rv.refresh_from_data)
            fbind('key_size', rv.refresh_from_data)
            fbind('default_size_hint', rv.refresh_from_data)
            fbind('key_size_hint', rv.refresh_from_data)
            fbind('default_size_hint_min', rv.refresh_from_data)
            fbind('key_size_hint_min', rv.refresh_from_data)
            fbind('default_size_hint_max', rv.refresh_from_data)
            fbind('key_size_hint_max', rv.refresh_from_data)
            fbind('default_pos_hint', rv.refresh_from_data)
            fbind('key_pos_hint', rv.refresh_from_data)

    def detach_recycleview(self):
        if False:
            return 10
        rv = self.recycleview
        if rv:
            funbind = self.funbind
            funbind('default_size', rv.refresh_from_data)
            funbind('key_size', rv.refresh_from_data)
            funbind('default_size_hint', rv.refresh_from_data)
            funbind('key_size_hint', rv.refresh_from_data)
            funbind('default_size_hint_min', rv.refresh_from_data)
            funbind('key_size_hint_min', rv.refresh_from_data)
            funbind('default_size_hint_max', rv.refresh_from_data)
            funbind('key_size_hint_max', rv.refresh_from_data)
            funbind('default_pos_hint', rv.refresh_from_data)
            funbind('key_pos_hint', rv.refresh_from_data)
        super(RecycleLayout, self).detach_recycleview()

    def _catch_layout_trigger(self, instance=None, value=None):
        if False:
            return 10
        rv = self.recycleview
        if rv is None:
            return
        idx = self.view_indices.get(instance)
        if idx is not None:
            if self._size_needs_update:
                return
            opt = self.view_opts[idx]
            if instance.size == opt['size'] and instance.size_hint == opt['size_hint'] and (instance.size_hint_min == opt['size_hint_min']) and (instance.size_hint_max == opt['size_hint_max']) and (instance.pos_hint == opt['pos_hint']):
                return
            self._size_needs_update = True
            rv.refresh_from_layout(view_size=True)
        else:
            rv.refresh_from_layout()

    def compute_sizes_from_data(self, data, flags):
        if False:
            print('Hello World!')
        if [f for f in flags if not f]:
            self.clear_layout()
            opts = self.view_opts = [None for _ in data]
        else:
            opts = self.view_opts
            changed = False
            for flag in flags:
                for (k, v) in flag.items():
                    changed = True
                    if k == 'removed':
                        del opts[v]
                    elif k == 'appended':
                        opts.extend([None] * (v.stop - v.start))
                    elif k == 'inserted':
                        opts.insert(v, None)
                    elif k == 'modified':
                        (start, stop, step) = (v.start, v.stop, v.step)
                        r = range(start, stop) if step is None else range(start, stop, step)
                        for i in r:
                            opts[i] = None
                    else:
                        raise Exception('Unrecognized data flag {}'.format(k))
            if changed:
                self.clear_layout()
        assert len(data) == len(opts)
        ph_key = self.key_pos_hint
        ph_def = self.default_pos_hint
        sh_key = self.key_size_hint
        sh_def = self.default_size_hint
        sh_min_key = self.key_size_hint_min
        sh_min_def = self.default_size_hint_min
        sh_max_key = self.key_size_hint_max
        sh_max_def = self.default_size_hint_max
        s_key = self.key_size
        s_def = self.default_size
        viewcls_def = self.viewclass
        viewcls_key = self.key_viewclass
        (iw, ih) = self.initial_size
        sh = []
        for (i, item) in enumerate(data):
            if opts[i] is not None:
                continue
            ph = ph_def if ph_key is None else item.get(ph_key, ph_def)
            ph = item.get('pos_hint', ph)
            sh = sh_def if sh_key is None else item.get(sh_key, sh_def)
            sh = item.get('size_hint', sh)
            sh = [item.get('size_hint_x', sh[0]), item.get('size_hint_y', sh[1])]
            sh_min = sh_min_def if sh_min_key is None else item.get(sh_min_key, sh_min_def)
            sh_min = item.get('size_hint_min', sh_min)
            sh_min = [item.get('size_hint_min_x', sh_min[0]), item.get('size_hint_min_y', sh_min[1])]
            sh_max = sh_max_def if sh_max_key is None else item.get(sh_max_key, sh_max_def)
            sh_max = item.get('size_hint_max', sh_max)
            sh_max = [item.get('size_hint_max_x', sh_max[0]), item.get('size_hint_max_y', sh_max[1])]
            s = s_def if s_key is None else item.get(s_key, s_def)
            s = item.get('size', s)
            (w, h) = s = (item.get('width', s[0]), item.get('height', s[1]))
            viewcls = None
            if viewcls_key is not None:
                viewcls = item.get(viewcls_key)
                if viewcls is not None:
                    viewcls = getattr(Factory, viewcls)
            if viewcls is None:
                viewcls = viewcls_def
            opts[i] = {'size': [iw if w is None else w, ih if h is None else h], 'size_hint': sh, 'size_hint_min': sh_min, 'size_hint_max': sh_max, 'pos': None, 'pos_hint': ph, 'viewclass': viewcls, 'width_none': w is None, 'height_none': h is None}

    def compute_layout(self, data, flags):
        if False:
            print('Hello World!')
        self._size_needs_update = False
        opts = self.view_opts
        changed = []
        for (widget, index) in self.view_indices.items():
            opt = opts[index]
            s = opt['size']
            (w, h) = sn = list(widget.size)
            sh = opt['size_hint']
            (shnw, shnh) = shn = list(widget.size_hint)
            sh_min = opt['size_hint_min']
            shn_min = list(widget.size_hint_min)
            sh_max = opt['size_hint_max']
            shn_max = list(widget.size_hint_max)
            ph = opt['pos_hint']
            phn = dict(widget.pos_hint)
            if s != sn or sh != shn or ph != phn or (sh_min != shn_min) or (sh_max != shn_max):
                changed.append((index, widget, s, sn, sh, shn, sh_min, shn_min, sh_max, shn_max, ph, phn))
                if shnw is None:
                    if shnh is None:
                        opt['size'] = sn
                    else:
                        opt['size'] = [w, s[1]]
                elif shnh is None:
                    opt['size'] = [s[0], h]
                opt['size_hint'] = shn
                opt['size_hint_min'] = shn_min
                opt['size_hint_max'] = shn_max
                opt['pos_hint'] = phn
        if [f for f in flags if not f]:
            self._changed_views = []
        else:
            self._changed_views = changed if changed else None

    def do_layout(self, *largs):
        if False:
            return 10
        assert False

    def set_visible_views(self, indices, data, viewport):
        if False:
            print('Hello World!')
        view_opts = self.view_opts
        (new, remaining, old) = self.recycleview.view_adapter.set_visible_views(indices, data, view_opts)
        remove = self.remove_widget
        view_indices = self.view_indices
        for (_, widget) in old:
            remove(widget)
            del view_indices[widget]
        refresh_view_layout = self.refresh_view_layout
        for (index, widget) in new:
            opt = view_opts[index].copy()
            del opt['width_none']
            del opt['height_none']
            refresh_view_layout(index, opt, widget, viewport)
        add = self.add_widget
        for (index, widget) in new:
            view_indices[widget] = index
            if widget.parent is None:
                add(widget)
        changed = False
        for (index, widget) in new:
            opt = view_opts[index]
            if changed or (widget.size == opt['size'] and widget.size_hint == opt['size_hint'] and (widget.size_hint_min == opt['size_hint_min']) and (widget.size_hint_max == opt['size_hint_max']) and (widget.pos_hint == opt['pos_hint'])):
                continue
            changed = True
        if changed:
            self._size_needs_update = True
            self.recycleview.refresh_from_layout(view_size=True)

    def refresh_view_layout(self, index, layout, view, viewport):
        if False:
            i = 10
            return i + 15
        opt = self.view_opts[index].copy()
        width_none = opt.pop('width_none')
        height_none = opt.pop('height_none')
        opt.update(layout)
        (w, h) = opt['size']
        (shw, shh) = opt['size_hint']
        if shw is None and width_none:
            w = None
        if shh is None and height_none:
            h = None
        opt['size'] = (w, h)
        super(RecycleLayout, self).refresh_view_layout(index, opt, view, viewport)

    def remove_views(self):
        if False:
            for i in range(10):
                print('nop')
        super(RecycleLayout, self).remove_views()
        self.clear_widgets()
        self.view_indices = {}

    def remove_view(self, view, index):
        if False:
            i = 10
            return i + 15
        super(RecycleLayout, self).remove_view(view, index)
        self.remove_widget(view)
        del self.view_indices[view]

    def clear_layout(self):
        if False:
            while True:
                i = 10
        super(RecycleLayout, self).clear_layout()
        self.clear_widgets()
        self.view_indices = {}
        self._size_needs_update = False