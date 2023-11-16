""" HVLayout

The HVLayout and its subclasses provide a simple mechanism to horizontally
or vertically stack child widgets. This can be done in different *modes*:
box mode is suited for aligning content where natural size matters. The
fix mode and split mode are more suited for high-level layout. See
the HVLayout class for details.


Interactive Box layout example:

.. UIExample:: 200

    from flexx import app, event, ui

    class Example(ui.HBox):
        def init(self):
            self.b1 = ui.Button(text='Horizontal', flex=0)
            self.b2 = ui.Button(text='Vertical', flex=1)
            self.b3 = ui.Button(text='Horizontal reversed', flex=2)
            self.b4 = ui.Button(text='Vertical reversed', flex=3)

        @event.reaction('b1.pointer_down')
        def _to_horizontal(self, *events):
            self.set_orientation('h')

        @event.reaction('b2.pointer_down')
        def _to_vertical(self, *events):
            self.set_orientation('v')

        @event.reaction('b3.pointer_down')
        def _to_horizontal_rev(self, *events):
            self.set_orientation('hr')

        @event.reaction('b4.pointer_down')
        def _to_vertical_r(self, *events):
            self.set_orientation('vr')

Also see examples: :ref:`app_layout.py`, :ref:`splitters.py`,
:ref:`box_vs_fix_layout.py`, :ref:`mondriaan.py`.

"""
'\n## Notes on performance and layout boundaries.\n\nIn layout one can see multiple streams of information:\n\n- Information about available size streams downward.\n- Information about minimum and maxium allowed sizes streams upward.\n- Information about natural sizes streams upward.\n\nThe first two streams are not problematic, as they are very much\none-directional, and minimum/maximum sizes are often quite static.\nThe flow of natural size is important to obtain good looking layouts, but\nadds complications because of its recursive effect; a change in size may\nneed several document reflows to get the layout right, which can cause\nsevere performance penalties if many elements are involved. Therefore it\nis important to introduce "layout boundaries" in the higher levels of a UI\nso that layout can be established within individual parts of the UI without\naffecting the other parts.\n\nThis module implements horizontal/vertical layouts that support natural sizes\n(box) and layouts that do not (fix and split). The former is implemented with\nCSS flexbox (the browser does all the work, and maintains the upward stream\nof natural sizes). The latter is implemented with absolute positioning (we make\nJavaScript do all the work). We realize good compatibility by maintaining the\nfirst two streams of information.\n\nTo clearify, it would be possible to implement split and fix with flexbox,\nand this could result in a "nicety" that a VSplit with content can still\nhave a natural horizontal size (and used as such in an HBox with flex 0).\nHowever, one can see how this will require additional document reflows\n(since a change in width can change the natural height and vice versa).\nSplit and Fix layouts provide an intuitive way to introduce layout boundaries.\n\nFor an element to be a layout boundary it must:\n\n- Not be display inline or inline-block\n- Not have a percentage height value.\n- Not have an implicit or auto height value.\n- Not have an implicit or auto width value.\n- Have an explicit overflow value (scroll, auto or hidden).\n- Not be a descendant of a <table> element.\n\nMost Widgets inside a HVLayout in split or fix mode conform to this:\nthey are typically not table elements, the Widget sets overflow, the layout\nitself uses CSS to set display, and sets height and weight.\n\nMore reading:\n\n- http://wilsonpage.co.uk/introducing-layout-boundaries/\n- https://css-tricks.com/snippets/css/a-guide-to-flexbox/\n\n'
from ... import event, app
from ...event import Property
from . import Layout

class OrientationProp(Property):
    """ A property that represents a pair of float values, which can also be
    set using a scalar.
    """
    _default = 'h'

    def _validate(self, v, name, data):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(v, str):
            v = v.lower().replace('-', '')
        v = {'horizontal': 'h', 0: 'h', 'lefttoright': 'h', 'vertical': 'v', 1: 'v', 'toptobottom': 'v', 'righttoleft': 'hr', 'bottomtotop': 'vr'}.get(v, v)
        if v not in ('h', 'v', 'hr', 'vr'):
            raise ValueError('%s.orientation got unknown value %r' % (self.id, v))
        return v

class HVLayout(Layout):
    """ A layout widget to distribute child widgets horizontally or vertically.

    This is a versatile layout class which can operate in different
    orientations (horizontal, vertical, reversed), and in different modes:

    In 'fix' mode, all available space is simply distributed corresponding
    to the children's flex values. This can be convenient to e.g. split
    a layout in two halves.

    In 'box' mode, each widget gets at least its natural size (if available),
    and any *additional* space is distributed corresponding to the children's
    flex values. This is convenient for low-level layout of widgets, e.g. to
    align  one or more buttons. It is common to use flex values of zero to
    give widgets just the size that they needs and use an empty widget with a
    flex of 1 to fill up any remaining space. This mode is based on CSS flexbox.

    In 'split' mode, all available space is initially distributed corresponding
    to the children's flex values. The splitters between the child widgets
    can be dragged by the user and positioned via an action. This is useful
    to give the user more control over the (high-level) layout.

    In all modes, the layout is constrained by the minimum and maximum size
    of the child widgets (as set via style/CSS). Note that flexbox (and thus
    box mode) may not honour min/max sizes of widgets in child layouts.

    Note that widgets with a flex value of zero may collapse if used inside
    a fix/split layout, or in a box layout but lacking a natural size. This
    can be resolved by assigning a minimum width/height to the widget. The
    exception is if all child widgets have a flex value of zero, in which
    case the available space is divided equally.

    The ``node`` of this widget is a
    `<div> <https://developer.mozilla.org/docs/Web/HTML/Element/div>`_. The
    outer nodes of the child widgets are layed-out using JavaScript of CSS,
    depending on the mode.
    
    Also see the convenience classes: HFix, VFix, HBox, VBox, HSplit, VSplit.
    """
    _DEFAULT_ORIENTATION = 'h'
    _DEFAULT_MODE = 'box'
    CSS = '\n\n    /* === for box layout === */\n\n    .flx-HVLayout > .flx-Widget {\n        margin: 0; /* the layout handles the margin */\n     }\n\n    .flx-box {\n        display: -webkit-flex;\n        display: -ms-flexbox;  /* IE 10 */\n        display: -ms-flex;     /* IE 11 */\n        display: -moz-flex;\n        display: flex;\n\n        /* How space is divided when all flex-factors are 0:\n           start, end, center, space-between, space-around */\n        -webkit-justify-content: space-around;\n        -ms-justify-content: space-around;\n        -moz-justify-content: space-around;\n        justify-content: space-around;\n\n        /* How items are aligned in the other direction:\n           center, stretch, baseline */\n        -webkit-align-items: stretch;\n        -ms-align-items: stretch;\n        -moz-align-items: stretch;\n        align-items: stretch;\n    }\n\n    .flx-box.flx-horizontal {\n        -webkit-flex-flow: row;\n        -ms-flex-flow: row;\n        -moz-flex-flow: row;\n        flex-flow: row;\n        width: 100%;\n    }\n    .flx-box.flx-vertical {\n        -webkit-flex-flow: column;\n        -ms-flex-flow: column;\n        -moz-flex-flow: column;\n        flex-flow: column;\n        height: 100%; width: 100%;\n    }\n    .flx-box.flx-horizontal.flx-reversed {\n        -webkit-flex-flow: row-reverse;\n        -ms-flex-flow: row-reverse;\n        -moz-flex-flow: row-reverse;\n        flex-flow: row-reverse;\n    }\n    .flx-box.flx-vertical.flx-reversed {\n        -webkit-flex-flow: column-reverse;\n        -ms-flex-flow: column-reverse;\n        -moz-flex-flow: column-reverse;\n        flex-flow: column-reverse;\n    }\n\n    /* Make child widgets (and layouts) size correctly */\n    .flx-box.flx-horizontal > .flx-Widget {\n        height: auto;\n        width: auto;\n    }\n    .flx-box.flx-vertical > .flx-Widget {\n        width: auto;\n        height: auto;\n    }\n\n    /* If a boxLayout is in a compound widget, we need to make that widget\n       a flex container (done with JS in Widget class), and scale here */\n    .flx-Widget > .flx-box {\n        flex-grow: 1;\n        flex-shrink: 1;\n    }\n\n    /* === For split and fix layout === */\n\n    .flx-split > .flx-Widget {\n        /* Let child widgets position well, and help them become a layout\n         * boundary. We cannot do "display: block;", as that would break stuff.\n        /* overflow is set in Widget.CSS, setting here breaks scrollable widgets\n         */\n        position: absolute;\n    }\n\n    .flx-split.flx-dragging { /* Fix for odd drag behavior on Chrome */\n        -webkit-user-select: none;\n        -moz-user-select: none;\n        -ms-user-select: none;\n        user-select: none;\n    }\n    .flx-split.flx-dragging iframe {  /* disable iframe during drag */\n        pointer-events: none;\n    }\n\n    .flx-split.flx-horizontal > .flx-split-sep,\n    .flx-split.flx-horizontal.flx-dragging {\n        cursor: ew-resize;\n    }\n    .flx-split.flx-vertical > .flx-split-sep,\n    .flx-split.flx-vertical.flx-dragging {\n        cursor: ns-resize;\n    }\n    .flx-split-sep {\n        z-index: 2;\n        position: absolute;\n        -webkit-user-select: none;\n        -moz-user-select: none;\n        -ms-user-select: none;\n        user-select: none;\n        box-sizing: border-box;\n        background: rgba(0, 0, 0, 0); /* transparent */\n        /* background: #fff;  /* hide underlying widgets */\n    }\n    '
    mode = event.EnumProp(('box', 'fix', 'split'), settable=True, doc="\n        The mode in which this layout operates:\n\n        * 'BOX': (default) each widget gets at least its natural size, and\n          additional space is distributed corresponding to the flex values.\n        * 'FIX': all available space is distributed corresponding to the flex values.\n        * 'SPLIT': available space is initially distributed correspondong to the\n          flex values, and can be modified by the user by dragging the splitters.\n        ")
    orientation = OrientationProp(settable=True, doc="\n        The orientation of the child widgets. 'h' or 'v' for horizontal and\n        vertical, or their reversed variants 'hr' and 'vr'. Settable with\n        values: 0, 1, 'h', 'v', 'hr', 'vr', 'horizontal', 'vertical',\n        'left-to-right', 'right-to-left', 'top-to-bottom', 'bottom-to-top'\n        (insensitive to case and use of dashes).\n        ")
    spacing = event.FloatProp(4, settable=True, doc='\n        The space between two child elements (in pixels).\n        ')
    padding = event.FloatProp(1, settable=True, doc='\n        The empty space around the layout (in pixels).\n        ')
    splitter_positions = app.LocalProperty(doc='\n        The preferred relative positions of the splitters. The actual\n        positions are subject to minsize and maxsize constraints\n        (and natural sizes for box-mode).\n        ')

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        kwargs['mode'] = kwargs.get('mode', self._DEFAULT_MODE)
        kwargs['orientation'] = kwargs.get('orientation', self._DEFAULT_ORIENTATION)
        self._seps = []
        self._dragging = None
        super().__init__(*args, **kwargs)
        if 'Split' in self._id and 'spacing' not in kwargs:
            self.set_spacing(8)

    @event.action
    def set_from_flex_values(self):
        if False:
            while True:
                i = 10
        " Set the divider positions corresponding to the children's flex values.\n        Only has a visual effect in split-mode.\n        "
        sizes = []
        dim = 0 if 'h' in self.orientation else 1
        for widget in self.children:
            sizes.append(widget.flex[dim])
        size_sum = 0 if len(sizes) == 0 else sum(sizes)
        if size_sum == 0:
            sizes = [1 / len(sizes) for i in sizes]
        else:
            sizes = [i / size_sum for i in sizes]
        positions = []
        pos = 0
        for i in range(len(sizes) - 1):
            pos = pos + sizes[i]
            positions.append(pos)
        self._mutate_splitter_positions(positions)

    @event.action
    def set_splitter_positions(self, *positions):
        if False:
            i = 10
            return i + 15
        ' Set relative splitter posisions (None or values between 0 and 1).\n        Only usable in split-mode.\n        '
        if self.mode != 'SPLIT':
            return
        positions2 = []
        for i in range(len(positions)):
            pos = positions[i]
            if pos is not None:
                pos = max(0.0, min(1.0, float(pos)))
            positions2.append(pos)
        self._mutate_splitter_positions(positions2)

    @event.emitter
    def user_splitter_positions(self, *positions):
        if False:
            for i in range(10):
                print('nop')
        ' Event emitted when the splitter is positioned by the user.\n        The event has a ``positions`` attribute.\n        '
        if self.mode != 'SPLIT':
            return None
        positions2 = []
        for i in range(len(positions)):
            pos = positions[i]
            if pos is not None:
                pos = max(0.0, min(1.0, float(pos)))
            positions2.append(pos)
        self.set_splitter_positions(*positions)
        return {'positions': positions}

    def _query_min_max_size(self):
        if False:
            print('Hello World!')
        ' Overload to also take child limits into account.\n        '
        hori = 'h' in self.orientation
        mima0 = super()._query_min_max_size()
        if hori is True:
            mima1 = [0, 0, 0, 1000000000.0]
        else:
            mima1 = [0, 1000000000.0, 0, 0]
        if self.minsize_from_children:
            for child in self.children:
                mima2 = child._size_limits
                if hori is True:
                    mima1[0] += mima2[0]
                    mima1[1] += mima2[1]
                    mima1[2] = max(mima1[2], mima2[2])
                    mima1[3] = min(mima1[3], mima2[3])
                else:
                    mima1[0] = max(mima1[0], mima2[0])
                    mima1[1] = min(mima1[1], mima2[1])
                    mima1[2] += mima2[2]
                    mima1[3] += mima2[3]
        if mima1[1] == 0:
            mima1[1] = 1000000000.0
        if mima1[3] == 0:
            mima1[3] = 1000000000.0
        if self.minsize_from_children:
            extra_padding = self.padding * 2
            extra_spacing = self.spacing * (len(self.children) - 1)
            for i in range(4):
                mima1[i] += extra_padding
            if hori is True:
                mima1[0] += extra_spacing
                mima1[1] += extra_spacing
            else:
                mima1[2] += extra_spacing
                mima1[3] += extra_spacing
        return [max(mima1[0], mima0[0]), min(mima1[1], mima0[1]), max(mima1[2], mima0[2]), min(mima1[3], mima0[3])]

    @event.reaction('size', '_size_limits', mode='greedy')
    def __size_changed(self, *events):
        if False:
            return 10
        self._rerender()

    @event.reaction('children*.size', mode='greedy')
    def __let_children_check_size(self, *events):
        if False:
            print('Hello World!')
        for child in self.children:
            child.check_real_size()

    @event.reaction('mode')
    def __set_mode(self, *events):
        if False:
            i = 10
            return i + 15
        for child in self.children:
            self._release_child(child)
        if self.mode == 'BOX':
            self.outernode.classList.remove('flx-split')
            self.outernode.classList.add('flx-box')
            self._set_box_child_flexes()
            self._set_box_spacing()
        else:
            self.outernode.classList.remove('flx-box')
            self.outernode.classList.add('flx-split')
            self._rerender()

    @event.reaction('orientation')
    def __set_orientation(self, *events):
        if False:
            return 10
        ori = self.orientation
        if 'h' in ori:
            self.outernode.classList.add('flx-horizontal')
            self.outernode.classList.remove('flx-vertical')
        else:
            self.outernode.classList.remove('flx-horizontal')
            self.outernode.classList.add('flx-vertical')
        if 'r' in ori:
            self.outernode.classList.add('flx-reversed')
        else:
            self.outernode.classList.remove('flx-reversed')
        for widget in self.children:
            widget.check_real_size()
        self._rerender()

    @event.reaction('padding')
    def __set_padding(self, *events):
        if False:
            while True:
                i = 10
        self.outernode.style['padding'] = self.padding + 'px'
        for widget in self.children:
            widget.check_real_size()
        self._rerender()

    def _release_child(self, widget):
        if False:
            return 10
        for n in ['margin', 'left', 'width', 'top', 'height']:
            widget.outernode.style[n] = ''

    def _render_dom(self):
        if False:
            print('Hello World!')
        children = self.children
        mode = self.mode
        use_seps = mode == 'SPLIT'
        if mode == 'BOX':
            self._ensure_seps(0)
        else:
            self._ensure_seps(len(children) - 1)
        nodes = []
        for i in range(len(children)):
            nodes.append(children[i].outernode)
            if use_seps and i < len(self._seps):
                nodes.append(self._seps[i])
        return nodes

    def _ensure_seps(self, n):
        if False:
            return 10
        ' Ensure that we have exactly n seperators.\n        '
        global window
        n = max(0, n)
        to_remove = self._seps[n:]
        self._seps = self._seps[:n]
        while len(self._seps) < n:
            sep = window.document.createElement('div')
            self._seps.append(sep)
            sep.i = len(self._seps) - 1
            sep.classList.add('flx-split-sep')
            sep.rel_pos = 0
            sep.abs_pos = 0

    @event.action
    def _rerender(self):
        if False:
            i = 10
            return i + 15
        ' Invoke a re-render. Only necessary for fix/split mode.\n        '
        if self.mode == 'BOX':
            for child in self.children:
                child.check_real_size()
        else:
            sp1 = ()
            sp2 = self.splitter_positions
            sp2 = () if sp2 is None else sp2
            if len(sp2) == 0:
                sp1 = (1,)
            self._mutate_splitter_positions(sp1)
            self._mutate_splitter_positions(sp2)

    @event.reaction('orientation', 'children', 'children*.flex', mode='greedy')
    def _set_box_child_flexes(self, *events):
        if False:
            while True:
                i = 10
        if self.mode != 'BOX':
            return
        ori = self.orientation
        i = 0 if ori in (0, 'h', 'hr') else 1
        for widget in self.children:
            _applyBoxStyle(widget.outernode, 'flex-grow', widget.flex[i])
            _applyBoxStyle(widget.outernode, 'flex-shrink', widget.flex[i] or 1)
        for widget in self.children:
            widget.check_real_size()

    @event.reaction('spacing', 'orientation', 'children', mode='greedy')
    def _set_box_spacing(self, *events):
        if False:
            print('Hello World!')
        if self.mode != 'BOX':
            return
        ori = self.orientation
        children_events = [ev for ev in events if ev.type == 'children']
        old_children = children_events[0].old_value if children_events else []
        children = self.children
        for child in children:
            child.outernode.style['margin-top'] = ''
            child.outernode.style['margin-left'] = ''
        for child in old_children:
            child.outernode.style['margin-top'] = ''
            child.outernode.style['margin-left'] = ''
        margin = 'margin-top' if ori in (1, 'v', 'vr') else 'margin-left'
        if children.length:
            if ori in ('vr', 'hr'):
                children[-1].outernode.style[margin] = '0px'
                for child in children[:-1]:
                    child.outernode.style[margin] = self.spacing + 'px'
            else:
                children[0].outernode.style[margin] = '0px'
                for child in children[1:]:
                    child.outernode.style[margin] = self.spacing + 'px'
        for widget in children:
            widget.check_real_size()

    def _get_available_size(self):
        if False:
            return 10
        bar_size = self.spacing
        pad_size = self.padding
        if 'h' in self.orientation:
            total_size = self.outernode.clientWidth
        else:
            total_size = self.outernode.clientHeight
        return (total_size, total_size - bar_size * len(self._seps) - 2 * pad_size)

    @event.reaction('spacing')
    def __spacing_changed(self, *events):
        if False:
            return 10
        self._rerender()

    @event.reaction('children', 'children*.flex', mode='greedy')
    def _set_split_from_flexes(self, *events):
        if False:
            for i in range(10):
                print('nop')
        self.set_from_flex_values()

    @event.reaction
    def __watch_splitter_positions(self):
        if False:
            while True:
                i = 10
        ' Set the slider positions, subject to constraints.\n        '
        if self.mode != 'BOX':
            self.splitter_positions
            self.emit('_render')

    def __apply_one_splitter_pos(self, index, pos):
        if False:
            print('Hello World!')
        ' Set the absolute position of one splitter. Called from move event.\n        '
        children = self.children
        (total_size, available_size) = self._get_available_size()
        ori = self.orientation
        if index >= len(self._seps):
            return
        if pos < 0:
            pos = available_size - pos
        pos = max(0, min(available_size, pos))
        abs_positions = [sep.abs_pos for sep in self._seps]
        abs_positions[index] = pos
        ref_pos = pos
        for i in reversed(range(0, index)):
            cur = abs_positions[i]
            (mi, ma) = _get_min_max(children[i + 1], ori)
            abs_positions[i] = ref_pos = max(ref_pos - ma, min(ref_pos - mi, cur))
        ref_pos = pos
        for i in range(index + 1, len(abs_positions)):
            cur = abs_positions[i]
            (mi, ma) = _get_min_max(children[i], ori)
            abs_positions[i] = ref_pos = max(ref_pos + mi, min(ref_pos + ma, cur))
        ref_pos = available_size
        for i in reversed(range(0, len(abs_positions))):
            cur = abs_positions[i]
            (mi, ma) = _get_min_max(children[i + 1], ori)
            abs_positions[i] = ref_pos = max(ref_pos - ma, min(ref_pos - mi, cur))
        ref_pos = 0
        for i in range(0, len(abs_positions)):
            cur = abs_positions[i]
            (mi, ma) = _get_min_max(children[i], ori)
            abs_positions[i] = ref_pos = max(ref_pos + mi, min(ref_pos + ma, cur))
        self.user_splitter_positions(*[pos / available_size for pos in abs_positions])

    def __apply_positions(self):
        if False:
            while True:
                i = 10
        ' Set sep.abs_pos and sep.rel_pos on each separator.\n        Called by __render_positions.\n        '
        children = self.children
        (total_size, available_size) = self._get_available_size()
        ori = self.orientation
        positions = self.splitter_positions
        if len(positions) != len(self._seps):
            return
        if len(children) != len(self._seps) + 1:
            return
        for i in range(len(positions)):
            self._seps[i].abs_pos = positions[i] * available_size
        ww = []
        ref_pos = 0
        for i in range(len(children)):
            w = {}
            ww.append(w)
            if i < len(self._seps):
                w.given = self._seps[i].abs_pos - ref_pos
                ref_pos = self._seps[i].abs_pos
            else:
                w.given = available_size - ref_pos
            (w.mi, w.ma) = _get_min_max(children[i], ori)
            w.can_give = w.given - w.mi
            w.can_receive = w.ma - w.given
            w.has = w.given
        net_size = 0
        for w in ww:
            if w.can_give < 0:
                net_size += w.can_give
                w.has = w.mi
                w.can_give = 0
                w.can_receive = w.ma - w.has
            elif w.can_receive < 0:
                net_size -= w.can_receive
                w.has = w.ma
                w.can_receive = 0
                w.can_give = w.has - w.mi
        ww2 = ww.copy()
        for iter in range(4):
            if abs(net_size) < 0.5 or len(ww2) == 0:
                break
            size_for_each = net_size / len(ww2)
            for i in reversed(range(len(ww2))):
                w = ww2[i]
                if net_size > 0:
                    if w.can_receive > 0:
                        gets = min(w.can_receive, size_for_each)
                        net_size -= gets
                        w.can_receive -= gets
                        w.has += gets
                    if w.can_receive <= 0:
                        ww2.pop(i)
                else:
                    if w.can_give > 0:
                        take = min(w.can_give, -size_for_each)
                        net_size += take
                        w.can_give -= take
                        w.has -= take
                    if w.can_give <= 0:
                        ww2.pop(i)
        ref_pos = 0
        for i in range(len(self._seps)):
            ref_pos += ww[i].has
            self._seps[i].abs_pos = ref_pos
        for i in range(0, len(self._seps)):
            self._seps[i].rel_pos = self._seps[i].abs_pos / available_size

    @event.reaction('!_render', mode='greedy')
    def __render_positions(self):
        if False:
            for i in range(10):
                print('nop')
        ' Use the absolute positions on the seps to apply positions to\n        the child elements and separators.\n        '
        children = self.children
        bar_size = self.spacing
        pad_size = self.padding
        (total_size, available_size) = self._get_available_size()
        ori = self.orientation
        if len(children) != len(self._seps) + 1:
            return
        self.__apply_positions()
        is_horizonal = 'h' in ori
        is_reversed = 'r' in ori
        offset = pad_size
        last_sep_pos = 0
        for i in range(len(children)):
            widget = children[i]
            ref_pos = self._seps[i].abs_pos if i < len(self._seps) else available_size
            size = ref_pos - last_sep_pos
            if True:
                pos = last_sep_pos + offset
                if is_reversed is True:
                    pos = total_size - pos - size
                if is_horizonal is True:
                    widget.outernode.style.left = pos + 'px'
                    widget.outernode.style.width = size + 'px'
                    widget.outernode.style.top = pad_size + 'px'
                    widget.outernode.style.height = 'calc(100% - ' + 2 * pad_size + 'px)'
                else:
                    widget.outernode.style.left = pad_size + 'px'
                    widget.outernode.style.width = 'calc(100% - ' + 2 * pad_size + 'px)'
                    widget.outernode.style.top = pos + 'px'
                    widget.outernode.style.height = size + 'px'
            if i < len(self._seps):
                sep = self._seps[i]
                pos = sep.abs_pos + offset
                if is_reversed is True:
                    pos = total_size - pos - bar_size
                if is_horizonal is True:
                    sep.style.left = pos + 'px'
                    sep.style.width = bar_size + 'px'
                    sep.style.top = '0'
                    sep.style.height = '100%'
                else:
                    sep.style.top = pos + 'px'
                    sep.style.height = bar_size + 'px'
                    sep.style.left = '0'
                    sep.style.width = '100%'
                offset += bar_size
                last_sep_pos = sep.abs_pos
        for child in children:
            child.check_real_size()

    @event.emitter
    def pointer_down(self, e):
        if False:
            print('Hello World!')
        if self.mode == 'SPLIT' and e.target.classList.contains('flx-split-sep'):
            e.stopPropagation()
            sep = e.target
            t = e.changedTouches[0] if e.changedTouches else e
            x_or_y1 = t.clientX if 'h' in self.orientation else t.clientY
            self._dragging = (self.orientation, sep.i, sep.abs_pos, x_or_y1)
            self.outernode.classList.add('flx-dragging')
        else:
            return super().pointer_down(e)

    @event.emitter
    def pointer_up(self, e):
        if False:
            i = 10
            return i + 15
        self._dragging = None
        self.outernode.classList.remove('flx-dragging')
        return super().pointer_down(e)

    @event.emitter
    def pointer_move(self, e):
        if False:
            while True:
                i = 10
        if self._dragging is not None:
            e.stopPropagation()
            e.preventDefault()
            (ori, i, ref_pos, x_or_y1) = self._dragging
            if ori == self.orientation:
                t = e.changedTouches[0] if e.changedTouches else e
                x_or_y2 = t.clientX if 'h' in self.orientation else t.clientY
                diff = x_or_y1 - x_or_y2 if 'r' in ori else x_or_y2 - x_or_y1
                self.__apply_one_splitter_pos(i, max(0, ref_pos + diff))
        else:
            return super().pointer_move(e)

def _applyBoxStyle(e, sty, value):
    if False:
        return 10
    for prefix in ['-webkit-', '-ms-', '-moz-', '']:
        e.style[prefix + sty] = value

def _get_min_max(widget, ori):
    if False:
        i = 10
        return i + 15
    mima = widget._size_limits
    if 'h' in ori:
        return (mima[0], mima[1])
    else:
        return (mima[2], mima[3])

class HBox(HVLayout):
    """ Horizontal layout that tries to give each widget its natural size and
    distributes any remaining space corresponding to the widget's flex values.
    (I.e. an HVLayout with orientation 'h' and mode 'box'.)
    """
    _DEFAULT_ORIENTATION = 'h'
    _DEFAULT_MODE = 'box'

class VBox(HVLayout):
    """ Vertical layout that tries to give each widget its natural size and
    distributes any remaining space corresponding to the widget's flex values.
    (I.e. an HVLayout with orientation 'v' and mode 'box'.)
    """
    _DEFAULT_ORIENTATION = 'v'
    _DEFAULT_MODE = 'box'

class HFix(HVLayout):
    """ Horizontal layout that distributes the available space corresponding
    to the widget's flex values.
    (I.e. an HVLayout with orientation 'h' and mode 'fix'.)
    """
    _DEFAULT_ORIENTATION = 'h'
    _DEFAULT_MODE = 'fix'

class VFix(HVLayout):
    """ Vertical layout that distributes the available space corresponding
    to the widget's flex values.
    (I.e. an HVLayout with orientation 'v' and mode 'fix'.)
    """
    _DEFAULT_ORIENTATION = 'v'
    _DEFAULT_MODE = 'fix'

class HSplit(HVLayout):
    """ Horizontal layout that initially distributes the available space
    corresponding to the widget's flex values, and has draggable splitters.
    By default, this layout has a slightly larger spacing between the widgets.
    (I.e. an HVLayout with orientation 'h' and mode 'split'.)
    """
    _DEFAULT_ORIENTATION = 'h'
    _DEFAULT_MODE = 'split'

class VSplit(HVLayout):
    """ Vertical layout that initially distributes the available space
    corresponding to the widget's flex values, and has draggable splitters.
    By default, this layout has a slightly larger spacing between the widgets.
    (I.e. an HVLayout with orientation 'v' and mode 'split'.)
    """
    _DEFAULT_ORIENTATION = 'v'
    _DEFAULT_MODE = 'split'