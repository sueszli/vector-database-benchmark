"""
MultiCall - a class which inherits its methods from a Tkinter widget (Text, for
example), but enables multiple calls of functions per virtual event - all
matching events will be called, not only the most specific one. This is done
by wrapping the event functions - event_add, event_delete and event_info.
MultiCall recognizes only a subset of legal event sequences. Sequences which
are not recognized are treated by the original Tk handling mechanism. A
more-specific event will be called before a less-specific event.

The recognized sequences are complete one-event sequences (no emacs-style
Ctrl-X Ctrl-C, no shortcuts like <3>), for all types of events.
Key/Button Press/Release events can have modifiers.
The recognized modifiers are Shift, Control, Option and Command for Mac, and
Control, Alt, Shift, Meta/M for other platforms.

For all events which were handled by MultiCall, a new member is added to the
event instance passed to the binded functions - mc_type. This is one of the
event type constants defined in this module (such as MC_KEYPRESS).
For Key/Button events (which are handled by MultiCall and may receive
modifiers), another member is added - mc_state. This member gives the state
of the recognized modifiers, as a combination of the modifier constants
also defined in this module (for example, MC_SHIFT).
Using these members is absolutely portable.

The order by which events are called is defined by these rules:
1. A more-specific event will be called before a less-specific event.
2. A recently-binded event will be called before a previously-binded event,
   unless this conflicts with the first rule.
Each function will be called at most once for each event.
"""
import re
import sys
import tkinter
MC_KEYPRESS = 0
MC_KEYRELEASE = 1
MC_BUTTONPRESS = 2
MC_BUTTONRELEASE = 3
MC_ACTIVATE = 4
MC_CIRCULATE = 5
MC_COLORMAP = 6
MC_CONFIGURE = 7
MC_DEACTIVATE = 8
MC_DESTROY = 9
MC_ENTER = 10
MC_EXPOSE = 11
MC_FOCUSIN = 12
MC_FOCUSOUT = 13
MC_GRAVITY = 14
MC_LEAVE = 15
MC_MAP = 16
MC_MOTION = 17
MC_MOUSEWHEEL = 18
MC_PROPERTY = 19
MC_REPARENT = 20
MC_UNMAP = 21
MC_VISIBILITY = 22
MC_SHIFT = 1 << 0
MC_CONTROL = 1 << 2
MC_ALT = 1 << 3
MC_META = 1 << 5
MC_OPTION = 1 << 6
MC_COMMAND = 1 << 7
if sys.platform == 'darwin':
    _modifiers = (('Shift',), ('Control',), ('Option',), ('Command',))
    _modifier_masks = (MC_SHIFT, MC_CONTROL, MC_OPTION, MC_COMMAND)
else:
    _modifiers = (('Control',), ('Alt',), ('Shift',), ('Meta', 'M'))
    _modifier_masks = (MC_CONTROL, MC_ALT, MC_SHIFT, MC_META)
_modifier_names = dict([(name, number) for number in range(len(_modifiers)) for name in _modifiers[number]])
APPLICATION_GONE = 'application has been destroyed'

class _SimpleBinder:

    def __init__(self, type, widget, widgetinst):
        if False:
            print('Hello World!')
        self.type = type
        self.sequence = '<' + _types[type][0] + '>'
        self.widget = widget
        self.widgetinst = widgetinst
        self.bindedfuncs = []
        self.handlerid = None

    def bind(self, triplet, func):
        if False:
            i = 10
            return i + 15
        if not self.handlerid:

            def handler(event, l=self.bindedfuncs, mc_type=self.type):
                if False:
                    print('Hello World!')
                event.mc_type = mc_type
                wascalled = {}
                for i in range(len(l) - 1, -1, -1):
                    func = l[i]
                    if func not in wascalled:
                        wascalled[func] = True
                        r = func(event)
                        if r:
                            return r
            self.handlerid = self.widget.bind(self.widgetinst, self.sequence, handler)
        self.bindedfuncs.append(func)

    def unbind(self, triplet, func):
        if False:
            return 10
        self.bindedfuncs.remove(func)
        if not self.bindedfuncs:
            self.widget.unbind(self.widgetinst, self.sequence, self.handlerid)
            self.handlerid = None

    def __del__(self):
        if False:
            while True:
                i = 10
        if self.handlerid:
            try:
                self.widget.unbind(self.widgetinst, self.sequence, self.handlerid)
            except tkinter.TclError as e:
                if not APPLICATION_GONE in e.args[0]:
                    raise
_states = range(1 << len(_modifiers))
_state_names = [''.join((m[0] + '-' for (i, m) in enumerate(_modifiers) if 1 << i & s)) for s in _states]

def expand_substates(states):
    if False:
        i = 10
        return i + 15
    'For each item of states return a list containing all combinations of\n    that item with individual bits reset, sorted by the number of set bits.\n    '

    def nbits(n):
        if False:
            while True:
                i = 10
        'number of bits set in n base 2'
        nb = 0
        while n:
            (n, rem) = divmod(n, 2)
            nb += rem
        return nb
    statelist = []
    for state in states:
        substates = list(set((state & x for x in states)))
        substates.sort(key=nbits, reverse=True)
        statelist.append(substates)
    return statelist
_state_subsets = expand_substates(_states)
_state_codes = []
for s in _states:
    r = 0
    for i in range(len(_modifiers)):
        if 1 << i & s:
            r |= _modifier_masks[i]
    _state_codes.append(r)

class _ComplexBinder:

    def __create_handler(self, lists, mc_type, mc_state):
        if False:
            for i in range(10):
                print('nop')

        def handler(event, lists=lists, mc_type=mc_type, mc_state=mc_state, ishandlerrunning=self.ishandlerrunning, doafterhandler=self.doafterhandler):
            if False:
                return 10
            ishandlerrunning[:] = [True]
            event.mc_type = mc_type
            event.mc_state = mc_state
            wascalled = {}
            r = None
            for l in lists:
                for i in range(len(l) - 1, -1, -1):
                    func = l[i]
                    if func not in wascalled:
                        wascalled[func] = True
                        r = l[i](event)
                        if r:
                            break
                if r:
                    break
            ishandlerrunning[:] = []
            for f in doafterhandler:
                f()
            doafterhandler[:] = []
            if r:
                return r
        return handler

    def __init__(self, type, widget, widgetinst):
        if False:
            print('Hello World!')
        self.type = type
        self.typename = _types[type][0]
        self.widget = widget
        self.widgetinst = widgetinst
        self.bindedfuncs = {None: [[] for s in _states]}
        self.handlerids = []
        self.ishandlerrunning = []
        self.doafterhandler = []
        for s in _states:
            lists = [self.bindedfuncs[None][i] for i in _state_subsets[s]]
            handler = self.__create_handler(lists, type, _state_codes[s])
            seq = '<' + _state_names[s] + self.typename + '>'
            self.handlerids.append((seq, self.widget.bind(self.widgetinst, seq, handler)))

    def bind(self, triplet, func):
        if False:
            while True:
                i = 10
        if triplet[2] not in self.bindedfuncs:
            self.bindedfuncs[triplet[2]] = [[] for s in _states]
            for s in _states:
                lists = [self.bindedfuncs[detail][i] for detail in (triplet[2], None) for i in _state_subsets[s]]
                handler = self.__create_handler(lists, self.type, _state_codes[s])
                seq = '<%s%s-%s>' % (_state_names[s], self.typename, triplet[2])
                self.handlerids.append((seq, self.widget.bind(self.widgetinst, seq, handler)))
        doit = lambda : self.bindedfuncs[triplet[2]][triplet[0]].append(func)
        if not self.ishandlerrunning:
            doit()
        else:
            self.doafterhandler.append(doit)

    def unbind(self, triplet, func):
        if False:
            for i in range(10):
                print('nop')
        doit = lambda : self.bindedfuncs[triplet[2]][triplet[0]].remove(func)
        if not self.ishandlerrunning:
            doit()
        else:
            self.doafterhandler.append(doit)

    def __del__(self):
        if False:
            return 10
        for (seq, id) in self.handlerids:
            try:
                self.widget.unbind(self.widgetinst, seq, id)
            except tkinter.TclError as e:
                if not APPLICATION_GONE in e.args[0]:
                    raise
_types = (('KeyPress', 'Key'), ('KeyRelease',), ('ButtonPress', 'Button'), ('ButtonRelease',), ('Activate',), ('Circulate',), ('Colormap',), ('Configure',), ('Deactivate',), ('Destroy',), ('Enter',), ('Expose',), ('FocusIn',), ('FocusOut',), ('Gravity',), ('Leave',), ('Map',), ('Motion',), ('MouseWheel',), ('Property',), ('Reparent',), ('Unmap',), ('Visibility',))
_binder_classes = (_ComplexBinder,) * 4 + (_SimpleBinder,) * (len(_types) - 4)
_type_names = dict([(name, number) for number in range(len(_types)) for name in _types[number]])
_keysym_re = re.compile('^\\w+$')
_button_re = re.compile('^[1-5]$')

def _parse_sequence(sequence):
    if False:
        print('Hello World!')
    'Get a string which should describe an event sequence. If it is\n    successfully parsed as one, return a tuple containing the state (as an int),\n    the event type (as an index of _types), and the detail - None if none, or a\n    string if there is one. If the parsing is unsuccessful, return None.\n    '
    if not sequence or sequence[0] != '<' or sequence[-1] != '>':
        return None
    words = sequence[1:-1].split('-')
    modifiers = 0
    while words and words[0] in _modifier_names:
        modifiers |= 1 << _modifier_names[words[0]]
        del words[0]
    if words and words[0] in _type_names:
        type = _type_names[words[0]]
        del words[0]
    else:
        return None
    if _binder_classes[type] is _SimpleBinder:
        if modifiers or words:
            return None
        else:
            detail = None
    else:
        if type in [_type_names[s] for s in ('KeyPress', 'KeyRelease')]:
            type_re = _keysym_re
        else:
            type_re = _button_re
        if not words:
            detail = None
        elif len(words) == 1 and type_re.match(words[0]):
            detail = words[0]
        else:
            return None
    return (modifiers, type, detail)

def _triplet_to_sequence(triplet):
    if False:
        return 10
    if triplet[2]:
        return '<' + _state_names[triplet[0]] + _types[triplet[1]][0] + '-' + triplet[2] + '>'
    else:
        return '<' + _state_names[triplet[0]] + _types[triplet[1]][0] + '>'
_multicall_dict = {}

def MultiCallCreator(widget):
    if False:
        while True:
            i = 10
    'Return a MultiCall class which inherits its methods from the\n    given widget class (for example, Tkinter.Text). This is used\n    instead of a templating mechanism.\n    '
    if widget in _multicall_dict:
        return _multicall_dict[widget]

    class MultiCall(widget):
        assert issubclass(widget, tkinter.Misc)

        def __init__(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            widget.__init__(self, *args, **kwargs)
            self.__eventinfo = {}
            self.__binders = [_binder_classes[i](i, widget, self) for i in range(len(_types))]

        def bind(self, sequence=None, func=None, add=None):
            if False:
                print('Hello World!')
            if type(sequence) is str and len(sequence) > 2 and (sequence[:2] == '<<') and (sequence[-2:] == '>>'):
                if sequence in self.__eventinfo:
                    ei = self.__eventinfo[sequence]
                    if ei[0] is not None:
                        for triplet in ei[1]:
                            self.__binders[triplet[1]].unbind(triplet, ei[0])
                    ei[0] = func
                    if ei[0] is not None:
                        for triplet in ei[1]:
                            self.__binders[triplet[1]].bind(triplet, func)
                else:
                    self.__eventinfo[sequence] = [func, []]
            return widget.bind(self, sequence, func, add)

        def unbind(self, sequence, funcid=None):
            if False:
                return 10
            if type(sequence) is str and len(sequence) > 2 and (sequence[:2] == '<<') and (sequence[-2:] == '>>') and (sequence in self.__eventinfo):
                (func, triplets) = self.__eventinfo[sequence]
                if func is not None:
                    for triplet in triplets:
                        self.__binders[triplet[1]].unbind(triplet, func)
                    self.__eventinfo[sequence][0] = None
            return widget.unbind(self, sequence, funcid)

        def event_add(self, virtual, *sequences):
            if False:
                print('Hello World!')
            if virtual not in self.__eventinfo:
                self.__eventinfo[virtual] = [None, []]
            (func, triplets) = self.__eventinfo[virtual]
            for seq in sequences:
                triplet = _parse_sequence(seq)
                if triplet is None:
                    widget.event_add(self, virtual, seq)
                else:
                    if func is not None:
                        self.__binders[triplet[1]].bind(triplet, func)
                    triplets.append(triplet)

        def event_delete(self, virtual, *sequences):
            if False:
                while True:
                    i = 10
            if virtual not in self.__eventinfo:
                return
            (func, triplets) = self.__eventinfo[virtual]
            for seq in sequences:
                triplet = _parse_sequence(seq)
                if triplet is None:
                    widget.event_delete(self, virtual, seq)
                else:
                    if func is not None:
                        self.__binders[triplet[1]].unbind(triplet, func)
                    triplets.remove(triplet)

        def event_info(self, virtual=None):
            if False:
                i = 10
                return i + 15
            if virtual is None or virtual not in self.__eventinfo:
                return widget.event_info(self, virtual)
            else:
                return tuple(map(_triplet_to_sequence, self.__eventinfo[virtual][1])) + widget.event_info(self, virtual)

        def __del__(self):
            if False:
                return 10
            for virtual in self.__eventinfo:
                (func, triplets) = self.__eventinfo[virtual]
                if func:
                    for triplet in triplets:
                        try:
                            self.__binders[triplet[1]].unbind(triplet, func)
                        except tkinter.TclError as e:
                            if not APPLICATION_GONE in e.args[0]:
                                raise
    _multicall_dict[widget] = MultiCall
    return MultiCall

def _multi_call(parent):
    if False:
        while True:
            i = 10
    top = tkinter.Toplevel(parent)
    top.title('Test MultiCall')
    (x, y) = map(int, parent.geometry().split('+')[1:])
    top.geometry('+%d+%d' % (x, y + 175))
    text = MultiCallCreator(tkinter.Text)(top)
    text.pack()

    def bindseq(seq, n=[0]):
        if False:
            i = 10
            return i + 15

        def handler(event):
            if False:
                print('Hello World!')
            print(seq)
        text.bind('<<handler%d>>' % n[0], handler)
        text.event_add('<<handler%d>>' % n[0], seq)
        n[0] += 1
    bindseq('<Key>')
    bindseq('<Control-Key>')
    bindseq('<Alt-Key-a>')
    bindseq('<Control-Key-a>')
    bindseq('<Alt-Control-Key-a>')
    bindseq('<Key-b>')
    bindseq('<Control-Button-1>')
    bindseq('<Button-2>')
    bindseq('<Alt-Button-1>')
    bindseq('<FocusOut>')
    bindseq('<Enter>')
    bindseq('<Leave>')
if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_mainmenu', verbosity=2, exit=False)
    from idlelib.idle_test.htest import run
    run(_multi_call)