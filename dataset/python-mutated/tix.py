import os
import warnings
import tkinter
from tkinter import *
from tkinter import _cnfmerge
warnings.warn('The Tix Tk extension is unmaintained, and the tkinter.tix wrapper module is deprecated in favor of tkinter.ttk', DeprecationWarning, stacklevel=2)
WINDOW = 'window'
TEXT = 'text'
STATUS = 'status'
IMMEDIATE = 'immediate'
IMAGE = 'image'
IMAGETEXT = 'imagetext'
BALLOON = 'balloon'
AUTO = 'auto'
ACROSSTOP = 'acrosstop'
ASCII = 'ascii'
CELL = 'cell'
COLUMN = 'column'
DECREASING = 'decreasing'
INCREASING = 'increasing'
INTEGER = 'integer'
MAIN = 'main'
MAX = 'max'
REAL = 'real'
ROW = 'row'
S_REGION = 's-region'
X_REGION = 'x-region'
Y_REGION = 'y-region'
TCL_DONT_WAIT = 1 << 1
TCL_WINDOW_EVENTS = 1 << 2
TCL_FILE_EVENTS = 1 << 3
TCL_TIMER_EVENTS = 1 << 4
TCL_IDLE_EVENTS = 1 << 5
TCL_ALL_EVENTS = 0

class tixCommand:
    """The tix commands provide access to miscellaneous  elements
    of  Tix's  internal state and the Tix application context.
    Most of the information manipulated by these  commands pertains
    to  the  application  as a whole, or to a screen or
    display, rather than to a particular window.

    This is a mixin class, assumed to be mixed to Tkinter.Tk
    that supports the self.tk.call method.
    """

    def tix_addbitmapdir(self, directory):
        if False:
            print('Hello World!')
        'Tix maintains a list of directories under which\n        the  tix_getimage  and tix_getbitmap commands will\n        search for image files. The standard bitmap  directory\n        is $TIX_LIBRARY/bitmaps. The addbitmapdir command\n        adds directory into this list. By  using  this\n        command, the  image  files  of an applications can\n        also be located using the tix_getimage or tix_getbitmap\n        command.\n        '
        return self.tk.call('tix', 'addbitmapdir', directory)

    def tix_cget(self, option):
        if False:
            while True:
                i = 10
        'Returns  the  current  value  of the configuration\n        option given by option. Option may be  any  of  the\n        options described in the CONFIGURATION OPTIONS section.\n        '
        return self.tk.call('tix', 'cget', option)

    def tix_configure(self, cnf=None, **kw):
        if False:
            return 10
        'Query or modify the configuration options of the Tix application\n        context. If no option is specified, returns a dictionary all of the\n        available options.  If option is specified with no value, then the\n        command returns a list describing the one named option (this list\n        will be identical to the corresponding sublist of the value\n        returned if no option is specified).  If one or more option-value\n        pairs are specified, then the command modifies the given option(s)\n        to have the given value(s); in this case the command returns an\n        empty string. Option may be any of the configuration options.\n        '
        if kw:
            cnf = _cnfmerge((cnf, kw))
        elif cnf:
            cnf = _cnfmerge(cnf)
        if cnf is None:
            return self._getconfigure('tix', 'configure')
        if isinstance(cnf, str):
            return self._getconfigure1('tix', 'configure', '-' + cnf)
        return self.tk.call(('tix', 'configure') + self._options(cnf))

    def tix_filedialog(self, dlgclass=None):
        if False:
            i = 10
            return i + 15
        'Returns the file selection dialog that may be shared among\n        different calls from this application.  This command will create a\n        file selection dialog widget when it is called the first time. This\n        dialog will be returned by all subsequent calls to tix_filedialog.\n        An optional dlgclass parameter can be passed to specified what type\n        of file selection dialog widget is desired. Possible options are\n        tix FileSelectDialog or tixExFileSelectDialog.\n        '
        if dlgclass is not None:
            return self.tk.call('tix', 'filedialog', dlgclass)
        else:
            return self.tk.call('tix', 'filedialog')

    def tix_getbitmap(self, name):
        if False:
            return 10
        "Locates a bitmap file of the name name.xpm or name in one of the\n        bitmap directories (see the tix_addbitmapdir command above).  By\n        using tix_getbitmap, you can avoid hard coding the pathnames of the\n        bitmap files in your application. When successful, it returns the\n        complete pathname of the bitmap file, prefixed with the character\n        '@'.  The returned value can be used to configure the -bitmap\n        option of the TK and Tix widgets.\n        "
        return self.tk.call('tix', 'getbitmap', name)

    def tix_getimage(self, name):
        if False:
            return 10
        'Locates an image file of the name name.xpm, name.xbm or name.ppm\n        in one of the bitmap directories (see the addbitmapdir command\n        above). If more than one file with the same name (but different\n        extensions) exist, then the image type is chosen according to the\n        depth of the X display: xbm images are chosen on monochrome\n        displays and color images are chosen on color displays. By using\n        tix_ getimage, you can avoid hard coding the pathnames of the\n        image files in your application. When successful, this command\n        returns the name of the newly created image, which can be used to\n        configure the -image option of the Tk and Tix widgets.\n        '
        return self.tk.call('tix', 'getimage', name)

    def tix_option_get(self, name):
        if False:
            while True:
                i = 10
        'Gets  the options  maintained  by  the  Tix\n        scheme mechanism. Available options include:\n\n            active_bg       active_fg      bg\n            bold_font       dark1_bg       dark1_fg\n            dark2_bg        dark2_fg       disabled_fg\n            fg              fixed_font     font\n            inactive_bg     inactive_fg    input1_bg\n            input2_bg       italic_font    light1_bg\n            light1_fg       light2_bg      light2_fg\n            menu_font       output1_bg     output2_bg\n            select_bg       select_fg      selector\n            '
        return self.tk.call('tix', 'option', 'get', name)

    def tix_resetoptions(self, newScheme, newFontSet, newScmPrio=None):
        if False:
            i = 10
            return i + 15
        'Resets the scheme and fontset of the Tix application to\n        newScheme and newFontSet, respectively.  This affects only those\n        widgets created after this call. Therefore, it is best to call the\n        resetoptions command before the creation of any widgets in a Tix\n        application.\n\n        The optional parameter newScmPrio can be given to reset the\n        priority level of the Tk options set by the Tix schemes.\n\n        Because of the way Tk handles the X option database, after Tix has\n        been has imported and inited, it is not possible to reset the color\n        schemes and font sets using the tix config command.  Instead, the\n        tix_resetoptions command must be used.\n        '
        if newScmPrio is not None:
            return self.tk.call('tix', 'resetoptions', newScheme, newFontSet, newScmPrio)
        else:
            return self.tk.call('tix', 'resetoptions', newScheme, newFontSet)

class Tk(tkinter.Tk, tixCommand):
    """Toplevel widget of Tix which represents mostly the main window
    of an application. It has an associated Tcl interpreter."""

    def __init__(self, screenName=None, baseName=None, className='Tix'):
        if False:
            print('Hello World!')
        tkinter.Tk.__init__(self, screenName, baseName, className)
        tixlib = os.environ.get('TIX_LIBRARY')
        self.tk.eval('global auto_path; lappend auto_path [file dir [info nameof]]')
        if tixlib is not None:
            self.tk.eval('global auto_path; lappend auto_path {%s}' % tixlib)
            self.tk.eval('global tcl_pkgPath; lappend tcl_pkgPath {%s}' % tixlib)
        self.tk.eval('package require Tix')

    def destroy(self):
        if False:
            return 10
        self.protocol('WM_DELETE_WINDOW', '')
        tkinter.Tk.destroy(self)

class Form:
    """The Tix Form geometry manager

    Widgets can be arranged by specifying attachments to other widgets.
    See Tix documentation for complete details"""

    def config(self, cnf={}, **kw):
        if False:
            return 10
        self.tk.call('tixForm', self._w, *self._options(cnf, kw))
    form = config

    def __setitem__(self, key, value):
        if False:
            return 10
        Form.form(self, {key: value})

    def check(self):
        if False:
            i = 10
            return i + 15
        return self.tk.call('tixForm', 'check', self._w)

    def forget(self):
        if False:
            return 10
        self.tk.call('tixForm', 'forget', self._w)

    def grid(self, xsize=0, ysize=0):
        if False:
            print('Hello World!')
        if not xsize and (not ysize):
            x = self.tk.call('tixForm', 'grid', self._w)
            y = self.tk.splitlist(x)
            z = ()
            for x in y:
                z = z + (self.tk.getint(x),)
            return z
        return self.tk.call('tixForm', 'grid', self._w, xsize, ysize)

    def info(self, option=None):
        if False:
            while True:
                i = 10
        if not option:
            return self.tk.call('tixForm', 'info', self._w)
        if option[0] != '-':
            option = '-' + option
        return self.tk.call('tixForm', 'info', self._w, option)

    def slaves(self):
        if False:
            for i in range(10):
                print('nop')
        return [self._nametowidget(x) for x in self.tk.splitlist(self.tk.call('tixForm', 'slaves', self._w))]
tkinter.Widget.__bases__ = tkinter.Widget.__bases__ + (Form,)

class TixWidget(tkinter.Widget):
    """A TixWidget class is used to package all (or most) Tix widgets.

    Widget initialization is extended in two ways:
       1) It is possible to give a list of options which must be part of
       the creation command (so called Tix 'static' options). These cannot be
       given as a 'config' command later.
       2) It is possible to give the name of an existing TK widget. These are
       child widgets created automatically by a Tix mega-widget. The Tk call
       to create these widgets is therefore bypassed in TixWidget.__init__

    Both options are for use by subclasses only.
    """

    def __init__(self, master=None, widgetName=None, static_options=None, cnf={}, kw={}):
        if False:
            print('Hello World!')
        if kw:
            cnf = _cnfmerge((cnf, kw))
        else:
            cnf = _cnfmerge(cnf)
        extra = ()
        if static_options:
            static_options.append('options')
        else:
            static_options = ['options']
        for (k, v) in list(cnf.items()):
            if k in static_options:
                extra = extra + ('-' + k, v)
                del cnf[k]
        self.widgetName = widgetName
        Widget._setup(self, master, cnf)
        if widgetName:
            self.tk.call(widgetName, self._w, *extra)
        if cnf:
            Widget.config(self, cnf)
        self.subwidget_list = {}

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        if name in self.subwidget_list:
            return self.subwidget_list[name]
        raise AttributeError(name)

    def set_silent(self, value):
        if False:
            i = 10
            return i + 15
        'Set a variable without calling its action routine'
        self.tk.call('tixSetSilent', self._w, value)

    def subwidget(self, name):
        if False:
            i = 10
            return i + 15
        'Return the named subwidget (which must have been created by\n        the sub-class).'
        n = self._subwidget_name(name)
        if not n:
            raise TclError('Subwidget ' + name + ' not child of ' + self._name)
        n = n[len(self._w) + 1:]
        return self._nametowidget(n)

    def subwidgets_all(self):
        if False:
            while True:
                i = 10
        'Return all subwidgets.'
        names = self._subwidget_names()
        if not names:
            return []
        retlist = []
        for name in names:
            name = name[len(self._w) + 1:]
            try:
                retlist.append(self._nametowidget(name))
            except:
                pass
        return retlist

    def _subwidget_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Get a subwidget name (returns a String, not a Widget !)'
        try:
            return self.tk.call(self._w, 'subwidget', name)
        except TclError:
            return None

    def _subwidget_names(self):
        if False:
            while True:
                i = 10
        'Return the name of all subwidgets.'
        try:
            x = self.tk.call(self._w, 'subwidgets', '-all')
            return self.tk.splitlist(x)
        except TclError:
            return None

    def config_all(self, option, value):
        if False:
            return 10
        'Set configuration options for all subwidgets (and self).'
        if option == '':
            return
        elif not isinstance(option, str):
            option = repr(option)
        if not isinstance(value, str):
            value = repr(value)
        names = self._subwidget_names()
        for name in names:
            self.tk.call(name, 'configure', '-' + option, value)

    def image_create(self, imgtype, cnf={}, master=None, **kw):
        if False:
            print('Hello World!')
        if master is None:
            master = self
        if kw and cnf:
            cnf = _cnfmerge((cnf, kw))
        elif kw:
            cnf = kw
        options = ()
        for (k, v) in cnf.items():
            if callable(v):
                v = self._register(v)
            options = options + ('-' + k, v)
        return master.tk.call(('image', 'create', imgtype) + options)

    def image_delete(self, imgname):
        if False:
            while True:
                i = 10
        try:
            self.tk.call('image', 'delete', imgname)
        except TclError:
            pass

class TixSubWidget(TixWidget):
    """Subwidget class.

    This is used to mirror child widgets automatically created
    by Tix/Tk as part of a mega-widget in Python (which is not informed
    of this)"""

    def __init__(self, master, name, destroy_physically=1, check_intermediate=1):
        if False:
            i = 10
            return i + 15
        if check_intermediate:
            path = master._subwidget_name(name)
            try:
                path = path[len(master._w) + 1:]
                plist = path.split('.')
            except:
                plist = []
        if not check_intermediate:
            TixWidget.__init__(self, master, None, None, {'name': name})
        else:
            parent = master
            for i in range(len(plist) - 1):
                n = '.'.join(plist[:i + 1])
                try:
                    w = master._nametowidget(n)
                    parent = w
                except KeyError:
                    parent = TixSubWidget(parent, plist[i], destroy_physically=0, check_intermediate=0)
            if plist:
                name = plist[-1]
            TixWidget.__init__(self, parent, None, None, {'name': name})
        self.destroy_physically = destroy_physically

    def destroy(self):
        if False:
            return 10
        for c in list(self.children.values()):
            c.destroy()
        if self._name in self.master.children:
            del self.master.children[self._name]
        if self._name in self.master.subwidget_list:
            del self.master.subwidget_list[self._name]
        if self.destroy_physically:
            self.tk.call('destroy', self._w)

class DisplayStyle:
    """DisplayStyle - handle configuration options shared by
    (multiple) Display Items"""

    def __init__(self, itemtype, cnf={}, *, master=None, **kw):
        if False:
            while True:
                i = 10
        if master is None:
            if 'refwindow' in kw:
                master = kw['refwindow']
            elif 'refwindow' in cnf:
                master = cnf['refwindow']
            else:
                master = tkinter._get_default_root('create display style')
        self.tk = master.tk
        self.stylename = self.tk.call('tixDisplayStyle', itemtype, *self._options(cnf, kw))

    def __str__(self):
        if False:
            print('Hello World!')
        return self.stylename

    def _options(self, cnf, kw):
        if False:
            for i in range(10):
                print('nop')
        if kw and cnf:
            cnf = _cnfmerge((cnf, kw))
        elif kw:
            cnf = kw
        opts = ()
        for (k, v) in cnf.items():
            opts = opts + ('-' + k, v)
        return opts

    def delete(self):
        if False:
            i = 10
            return i + 15
        self.tk.call(self.stylename, 'delete')

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        self.tk.call(self.stylename, 'configure', '-%s' % key, value)

    def config(self, cnf={}, **kw):
        if False:
            print('Hello World!')
        return self._getconfigure(self.stylename, 'configure', *self._options(cnf, kw))

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.tk.call(self.stylename, 'cget', '-%s' % key)

class Balloon(TixWidget):
    """Balloon help widget.

    Subwidget       Class
    ---------       -----
    label           Label
    message         Message"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        static = ['options', 'installcolormap', 'initwait', 'statusbar', 'cursor']
        TixWidget.__init__(self, master, 'tixBalloon', static, cnf, kw)
        self.subwidget_list['label'] = _dummyLabel(self, 'label', destroy_physically=0)
        self.subwidget_list['message'] = _dummyLabel(self, 'message', destroy_physically=0)

    def bind_widget(self, widget, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        'Bind balloon widget to another.\n        One balloon widget may be bound to several widgets at the same time'
        self.tk.call(self._w, 'bind', widget._w, *self._options(cnf, kw))

    def unbind_widget(self, widget):
        if False:
            while True:
                i = 10
        self.tk.call(self._w, 'unbind', widget._w)

class ButtonBox(TixWidget):
    """ButtonBox - A container for pushbuttons.
    Subwidgets are the buttons added with the add method.
    """

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            return 10
        TixWidget.__init__(self, master, 'tixButtonBox', ['orientation', 'options'], cnf, kw)

    def add(self, name, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        'Add a button with given name to box.'
        btn = self.tk.call(self._w, 'add', name, *self._options(cnf, kw))
        self.subwidget_list[name] = _dummyButton(self, name)
        return btn

    def invoke(self, name):
        if False:
            return 10
        if name in self.subwidget_list:
            self.tk.call(self._w, 'invoke', name)

class ComboBox(TixWidget):
    """ComboBox - an Entry field with a dropdown menu. The user can select a
    choice by either typing in the entry subwidget or selecting from the
    listbox subwidget.

    Subwidget       Class
    ---------       -----
    entry       Entry
    arrow       Button
    slistbox    ScrolledListBox
    tick        Button
    cross       Button : present if created with the fancy option"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        TixWidget.__init__(self, master, 'tixComboBox', ['editable', 'dropdown', 'fancy', 'options'], cnf, kw)
        self.subwidget_list['label'] = _dummyLabel(self, 'label')
        self.subwidget_list['entry'] = _dummyEntry(self, 'entry')
        self.subwidget_list['arrow'] = _dummyButton(self, 'arrow')
        self.subwidget_list['slistbox'] = _dummyScrolledListBox(self, 'slistbox')
        try:
            self.subwidget_list['tick'] = _dummyButton(self, 'tick')
            self.subwidget_list['cross'] = _dummyButton(self, 'cross')
        except TypeError:
            pass

    def add_history(self, str):
        if False:
            return 10
        self.tk.call(self._w, 'addhistory', str)

    def append_history(self, str):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'appendhistory', str)

    def insert(self, index, str):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'insert', index, str)

    def pick(self, index):
        if False:
            return 10
        self.tk.call(self._w, 'pick', index)

class Control(TixWidget):
    """Control - An entry field with value change arrows.  The user can
    adjust the value by pressing the two arrow buttons or by entering
    the value directly into the entry. The new value will be checked
    against the user-defined upper and lower limits.

    Subwidget       Class
    ---------       -----
    incr       Button
    decr       Button
    entry       Entry
    label       Label"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            print('Hello World!')
        TixWidget.__init__(self, master, 'tixControl', ['options'], cnf, kw)
        self.subwidget_list['incr'] = _dummyButton(self, 'incr')
        self.subwidget_list['decr'] = _dummyButton(self, 'decr')
        self.subwidget_list['label'] = _dummyLabel(self, 'label')
        self.subwidget_list['entry'] = _dummyEntry(self, 'entry')

    def decrement(self):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'decr')

    def increment(self):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'incr')

    def invoke(self):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'invoke')

    def update(self):
        if False:
            return 10
        self.tk.call(self._w, 'update')

class DirList(TixWidget):
    """DirList - displays a list view of a directory, its previous
    directories and its sub-directories. The user can choose one of
    the directories displayed in the list or change to another directory.

    Subwidget       Class
    ---------       -----
    hlist       HList
    hsb              Scrollbar
    vsb              Scrollbar"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            print('Hello World!')
        TixWidget.__init__(self, master, 'tixDirList', ['options'], cnf, kw)
        self.subwidget_list['hlist'] = _dummyHList(self, 'hlist')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

    def chdir(self, dir):
        if False:
            return 10
        self.tk.call(self._w, 'chdir', dir)

class DirTree(TixWidget):
    """DirTree - Directory Listing in a hierarchical view.
    Displays a tree view of a directory, its previous directories and its
    sub-directories. The user can choose one of the directories displayed
    in the list or change to another directory.

    Subwidget       Class
    ---------       -----
    hlist           HList
    hsb             Scrollbar
    vsb             Scrollbar"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            return 10
        TixWidget.__init__(self, master, 'tixDirTree', ['options'], cnf, kw)
        self.subwidget_list['hlist'] = _dummyHList(self, 'hlist')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

    def chdir(self, dir):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'chdir', dir)

class DirSelectBox(TixWidget):
    """DirSelectBox - Motif style file select box.
    It is generally used for
    the user to choose a file. FileSelectBox stores the files mostly
    recently selected into a ComboBox widget so that they can be quickly
    selected again.

    Subwidget       Class
    ---------       -----
    selection       ComboBox
    filter          ComboBox
    dirlist         ScrolledListBox
    filelist        ScrolledListBox"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        TixWidget.__init__(self, master, 'tixDirSelectBox', ['options'], cnf, kw)
        self.subwidget_list['dirlist'] = _dummyDirList(self, 'dirlist')
        self.subwidget_list['dircbx'] = _dummyFileComboBox(self, 'dircbx')

class ExFileSelectBox(TixWidget):
    """ExFileSelectBox - MS Windows style file select box.
    It provides a convenient method for the user to select files.

    Subwidget       Class
    ---------       -----
    cancel       Button
    ok              Button
    hidden       Checkbutton
    types       ComboBox
    dir              ComboBox
    file       ComboBox
    dirlist       ScrolledListBox
    filelist       ScrolledListBox"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            while True:
                i = 10
        TixWidget.__init__(self, master, 'tixExFileSelectBox', ['options'], cnf, kw)
        self.subwidget_list['cancel'] = _dummyButton(self, 'cancel')
        self.subwidget_list['ok'] = _dummyButton(self, 'ok')
        self.subwidget_list['hidden'] = _dummyCheckbutton(self, 'hidden')
        self.subwidget_list['types'] = _dummyComboBox(self, 'types')
        self.subwidget_list['dir'] = _dummyComboBox(self, 'dir')
        self.subwidget_list['dirlist'] = _dummyDirList(self, 'dirlist')
        self.subwidget_list['file'] = _dummyComboBox(self, 'file')
        self.subwidget_list['filelist'] = _dummyScrolledListBox(self, 'filelist')

    def filter(self):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'filter')

    def invoke(self):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'invoke')

class DirSelectDialog(TixWidget):
    """The DirSelectDialog widget presents the directories in the file
    system in a dialog window. The user can use this dialog window to
    navigate through the file system to select the desired directory.

    Subwidgets       Class
    ----------       -----
    dirbox       DirSelectDialog"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            while True:
                i = 10
        TixWidget.__init__(self, master, 'tixDirSelectDialog', ['options'], cnf, kw)
        self.subwidget_list['dirbox'] = _dummyDirSelectBox(self, 'dirbox')

    def popup(self):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'popup')

    def popdown(self):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'popdown')

class ExFileSelectDialog(TixWidget):
    """ExFileSelectDialog - MS Windows style file select dialog.
    It provides a convenient method for the user to select files.

    Subwidgets       Class
    ----------       -----
    fsbox       ExFileSelectBox"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            print('Hello World!')
        TixWidget.__init__(self, master, 'tixExFileSelectDialog', ['options'], cnf, kw)
        self.subwidget_list['fsbox'] = _dummyExFileSelectBox(self, 'fsbox')

    def popup(self):
        if False:
            while True:
                i = 10
        self.tk.call(self._w, 'popup')

    def popdown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'popdown')

class FileSelectBox(TixWidget):
    """ExFileSelectBox - Motif style file select box.
    It is generally used for
    the user to choose a file. FileSelectBox stores the files mostly
    recently selected into a ComboBox widget so that they can be quickly
    selected again.

    Subwidget       Class
    ---------       -----
    selection       ComboBox
    filter          ComboBox
    dirlist         ScrolledListBox
    filelist        ScrolledListBox"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            while True:
                i = 10
        TixWidget.__init__(self, master, 'tixFileSelectBox', ['options'], cnf, kw)
        self.subwidget_list['dirlist'] = _dummyScrolledListBox(self, 'dirlist')
        self.subwidget_list['filelist'] = _dummyScrolledListBox(self, 'filelist')
        self.subwidget_list['filter'] = _dummyComboBox(self, 'filter')
        self.subwidget_list['selection'] = _dummyComboBox(self, 'selection')

    def apply_filter(self):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'filter')

    def invoke(self):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'invoke')

class FileSelectDialog(TixWidget):
    """FileSelectDialog - Motif style file select dialog.

    Subwidgets       Class
    ----------       -----
    btns       StdButtonBox
    fsbox       FileSelectBox"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            return 10
        TixWidget.__init__(self, master, 'tixFileSelectDialog', ['options'], cnf, kw)
        self.subwidget_list['btns'] = _dummyStdButtonBox(self, 'btns')
        self.subwidget_list['fsbox'] = _dummyFileSelectBox(self, 'fsbox')

    def popup(self):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'popup')

    def popdown(self):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'popdown')

class FileEntry(TixWidget):
    """FileEntry - Entry field with button that invokes a FileSelectDialog.
    The user can type in the filename manually. Alternatively, the user can
    press the button widget that sits next to the entry, which will bring
    up a file selection dialog.

    Subwidgets       Class
    ----------       -----
    button       Button
    entry       Entry"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            print('Hello World!')
        TixWidget.__init__(self, master, 'tixFileEntry', ['dialogtype', 'options'], cnf, kw)
        self.subwidget_list['button'] = _dummyButton(self, 'button')
        self.subwidget_list['entry'] = _dummyEntry(self, 'entry')

    def invoke(self):
        if False:
            return 10
        self.tk.call(self._w, 'invoke')

    def file_dialog(self):
        if False:
            while True:
                i = 10
        pass

class HList(TixWidget, XView, YView):
    """HList - Hierarchy display  widget can be used to display any data
    that have a hierarchical structure, for example, file system directory
    trees. The list entries are indented and connected by branch lines
    according to their places in the hierarchy.

    Subwidgets - None"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            while True:
                i = 10
        TixWidget.__init__(self, master, 'tixHList', ['columns', 'options'], cnf, kw)

    def add(self, entry, cnf={}, **kw):
        if False:
            while True:
                i = 10
        return self.tk.call(self._w, 'add', entry, *self._options(cnf, kw))

    def add_child(self, parent=None, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        if parent is None:
            parent = ''
        return self.tk.call(self._w, 'addchild', parent, *self._options(cnf, kw))

    def anchor_set(self, entry):
        if False:
            while True:
                i = 10
        self.tk.call(self._w, 'anchor', 'set', entry)

    def anchor_clear(self):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'anchor', 'clear')

    def column_width(self, col=0, width=None, chars=None):
        if False:
            for i in range(10):
                print('nop')
        if not chars:
            return self.tk.call(self._w, 'column', 'width', col, width)
        else:
            return self.tk.call(self._w, 'column', 'width', col, '-char', chars)

    def delete_all(self):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'delete', 'all')

    def delete_entry(self, entry):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'delete', 'entry', entry)

    def delete_offsprings(self, entry):
        if False:
            return 10
        self.tk.call(self._w, 'delete', 'offsprings', entry)

    def delete_siblings(self, entry):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'delete', 'siblings', entry)

    def dragsite_set(self, index):
        if False:
            return 10
        self.tk.call(self._w, 'dragsite', 'set', index)

    def dragsite_clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'dragsite', 'clear')

    def dropsite_set(self, index):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'dropsite', 'set', index)

    def dropsite_clear(self):
        if False:
            return 10
        self.tk.call(self._w, 'dropsite', 'clear')

    def header_create(self, col, cnf={}, **kw):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'header', 'create', col, *self._options(cnf, kw))

    def header_configure(self, col, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        if cnf is None:
            return self._getconfigure(self._w, 'header', 'configure', col)
        self.tk.call(self._w, 'header', 'configure', col, *self._options(cnf, kw))

    def header_cget(self, col, opt):
        if False:
            return 10
        return self.tk.call(self._w, 'header', 'cget', col, opt)

    def header_exists(self, col):
        if False:
            for i in range(10):
                print('nop')
        return self.tk.getboolean(self.tk.call(self._w, 'header', 'exist', col))
    header_exist = header_exists

    def header_delete(self, col):
        if False:
            return 10
        self.tk.call(self._w, 'header', 'delete', col)

    def header_size(self, col):
        if False:
            return 10
        return self.tk.call(self._w, 'header', 'size', col)

    def hide_entry(self, entry):
        if False:
            return 10
        self.tk.call(self._w, 'hide', 'entry', entry)

    def indicator_create(self, entry, cnf={}, **kw):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'indicator', 'create', entry, *self._options(cnf, kw))

    def indicator_configure(self, entry, cnf={}, **kw):
        if False:
            print('Hello World!')
        if cnf is None:
            return self._getconfigure(self._w, 'indicator', 'configure', entry)
        self.tk.call(self._w, 'indicator', 'configure', entry, *self._options(cnf, kw))

    def indicator_cget(self, entry, opt):
        if False:
            i = 10
            return i + 15
        return self.tk.call(self._w, 'indicator', 'cget', entry, opt)

    def indicator_exists(self, entry):
        if False:
            for i in range(10):
                print('nop')
        return self.tk.call(self._w, 'indicator', 'exists', entry)

    def indicator_delete(self, entry):
        if False:
            while True:
                i = 10
        self.tk.call(self._w, 'indicator', 'delete', entry)

    def indicator_size(self, entry):
        if False:
            return 10
        return self.tk.call(self._w, 'indicator', 'size', entry)

    def info_anchor(self):
        if False:
            while True:
                i = 10
        return self.tk.call(self._w, 'info', 'anchor')

    def info_bbox(self, entry):
        if False:
            return 10
        return self._getints(self.tk.call(self._w, 'info', 'bbox', entry)) or None

    def info_children(self, entry=None):
        if False:
            return 10
        c = self.tk.call(self._w, 'info', 'children', entry)
        return self.tk.splitlist(c)

    def info_data(self, entry):
        if False:
            i = 10
            return i + 15
        return self.tk.call(self._w, 'info', 'data', entry)

    def info_dragsite(self):
        if False:
            return 10
        return self.tk.call(self._w, 'info', 'dragsite')

    def info_dropsite(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tk.call(self._w, 'info', 'dropsite')

    def info_exists(self, entry):
        if False:
            return 10
        return self.tk.call(self._w, 'info', 'exists', entry)

    def info_hidden(self, entry):
        if False:
            return 10
        return self.tk.call(self._w, 'info', 'hidden', entry)

    def info_next(self, entry):
        if False:
            return 10
        return self.tk.call(self._w, 'info', 'next', entry)

    def info_parent(self, entry):
        if False:
            while True:
                i = 10
        return self.tk.call(self._w, 'info', 'parent', entry)

    def info_prev(self, entry):
        if False:
            print('Hello World!')
        return self.tk.call(self._w, 'info', 'prev', entry)

    def info_selection(self):
        if False:
            i = 10
            return i + 15
        c = self.tk.call(self._w, 'info', 'selection')
        return self.tk.splitlist(c)

    def item_cget(self, entry, col, opt):
        if False:
            i = 10
            return i + 15
        return self.tk.call(self._w, 'item', 'cget', entry, col, opt)

    def item_configure(self, entry, col, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        if cnf is None:
            return self._getconfigure(self._w, 'item', 'configure', entry, col)
        self.tk.call(self._w, 'item', 'configure', entry, col, *self._options(cnf, kw))

    def item_create(self, entry, col, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'item', 'create', entry, col, *self._options(cnf, kw))

    def item_exists(self, entry, col):
        if False:
            for i in range(10):
                print('nop')
        return self.tk.call(self._w, 'item', 'exists', entry, col)

    def item_delete(self, entry, col):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'item', 'delete', entry, col)

    def entrycget(self, entry, opt):
        if False:
            for i in range(10):
                print('nop')
        return self.tk.call(self._w, 'entrycget', entry, opt)

    def entryconfigure(self, entry, cnf={}, **kw):
        if False:
            print('Hello World!')
        if cnf is None:
            return self._getconfigure(self._w, 'entryconfigure', entry)
        self.tk.call(self._w, 'entryconfigure', entry, *self._options(cnf, kw))

    def nearest(self, y):
        if False:
            print('Hello World!')
        return self.tk.call(self._w, 'nearest', y)

    def see(self, entry):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'see', entry)

    def selection_clear(self, cnf={}, **kw):
        if False:
            return 10
        self.tk.call(self._w, 'selection', 'clear', *self._options(cnf, kw))

    def selection_includes(self, entry):
        if False:
            print('Hello World!')
        return self.tk.call(self._w, 'selection', 'includes', entry)

    def selection_set(self, first, last=None):
        if False:
            return 10
        self.tk.call(self._w, 'selection', 'set', first, last)

    def show_entry(self, entry):
        if False:
            i = 10
            return i + 15
        return self.tk.call(self._w, 'show', 'entry', entry)

class InputOnly(TixWidget):
    """InputOnly - Invisible widget. Unix only.

    Subwidgets - None"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        TixWidget.__init__(self, master, 'tixInputOnly', None, cnf, kw)

class LabelEntry(TixWidget):
    """LabelEntry - Entry field with label. Packages an entry widget
    and a label into one mega widget. It can be used to simplify the creation
    of ``entry-form'' type of interface.

    Subwidgets       Class
    ----------       -----
    label       Label
    entry       Entry"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            return 10
        TixWidget.__init__(self, master, 'tixLabelEntry', ['labelside', 'options'], cnf, kw)
        self.subwidget_list['label'] = _dummyLabel(self, 'label')
        self.subwidget_list['entry'] = _dummyEntry(self, 'entry')

class LabelFrame(TixWidget):
    """LabelFrame - Labelled Frame container. Packages a frame widget
    and a label into one mega widget. To create widgets inside a
    LabelFrame widget, one creates the new widgets relative to the
    frame subwidget and manage them inside the frame subwidget.

    Subwidgets       Class
    ----------       -----
    label       Label
    frame       Frame"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            print('Hello World!')
        TixWidget.__init__(self, master, 'tixLabelFrame', ['labelside', 'options'], cnf, kw)
        self.subwidget_list['label'] = _dummyLabel(self, 'label')
        self.subwidget_list['frame'] = _dummyFrame(self, 'frame')

class ListNoteBook(TixWidget):
    """A ListNoteBook widget is very similar to the TixNoteBook widget:
    it can be used to display many windows in a limited space using a
    notebook metaphor. The notebook is divided into a stack of pages
    (windows). At one time only one of these pages can be shown.
    The user can navigate through these pages by
    choosing the name of the desired page in the hlist subwidget."""

    def __init__(self, master, cnf={}, **kw):
        if False:
            print('Hello World!')
        TixWidget.__init__(self, master, 'tixListNoteBook', ['options'], cnf, kw)
        self.subwidget_list['pane'] = _dummyPanedWindow(self, 'pane', destroy_physically=0)
        self.subwidget_list['hlist'] = _dummyHList(self, 'hlist')
        self.subwidget_list['shlist'] = _dummyScrolledHList(self, 'shlist')

    def add(self, name, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'add', name, *self._options(cnf, kw))
        self.subwidget_list[name] = TixSubWidget(self, name)
        return self.subwidget_list[name]

    def page(self, name):
        if False:
            while True:
                i = 10
        return self.subwidget(name)

    def pages(self):
        if False:
            i = 10
            return i + 15
        names = self.tk.splitlist(self.tk.call(self._w, 'pages'))
        ret = []
        for x in names:
            ret.append(self.subwidget(x))
        return ret

    def raise_page(self, name):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'raise', name)

class Meter(TixWidget):
    """The Meter widget can be used to show the progress of a background
    job which may take a long time to execute.
    """

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            return 10
        TixWidget.__init__(self, master, 'tixMeter', ['options'], cnf, kw)

class NoteBook(TixWidget):
    """NoteBook - Multi-page container widget (tabbed notebook metaphor).

    Subwidgets       Class
    ----------       -----
    nbframe       NoteBookFrame
    <pages>       page widgets added dynamically with the add method"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            while True:
                i = 10
        TixWidget.__init__(self, master, 'tixNoteBook', ['options'], cnf, kw)
        self.subwidget_list['nbframe'] = TixSubWidget(self, 'nbframe', destroy_physically=0)

    def add(self, name, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'add', name, *self._options(cnf, kw))
        self.subwidget_list[name] = TixSubWidget(self, name)
        return self.subwidget_list[name]

    def delete(self, name):
        if False:
            while True:
                i = 10
        self.tk.call(self._w, 'delete', name)
        self.subwidget_list[name].destroy()
        del self.subwidget_list[name]

    def page(self, name):
        if False:
            while True:
                i = 10
        return self.subwidget(name)

    def pages(self):
        if False:
            i = 10
            return i + 15
        names = self.tk.splitlist(self.tk.call(self._w, 'pages'))
        ret = []
        for x in names:
            ret.append(self.subwidget(x))
        return ret

    def raise_page(self, name):
        if False:
            while True:
                i = 10
        self.tk.call(self._w, 'raise', name)

    def raised(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tk.call(self._w, 'raised')

class NoteBookFrame(TixWidget):
    pass

class OptionMenu(TixWidget):
    """OptionMenu - creates a menu button of options.

    Subwidget       Class
    ---------       -----
    menubutton      Menubutton
    menu            Menu"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        TixWidget.__init__(self, master, 'tixOptionMenu', ['options'], cnf, kw)
        self.subwidget_list['menubutton'] = _dummyMenubutton(self, 'menubutton')
        self.subwidget_list['menu'] = _dummyMenu(self, 'menu')

    def add_command(self, name, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'add', 'command', name, *self._options(cnf, kw))

    def add_separator(self, name, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'add', 'separator', name, *self._options(cnf, kw))

    def delete(self, name):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'delete', name)

    def disable(self, name):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'disable', name)

    def enable(self, name):
        if False:
            return 10
        self.tk.call(self._w, 'enable', name)

class PanedWindow(TixWidget):
    """PanedWindow - Multi-pane container widget
    allows the user to interactively manipulate the sizes of several
    panes. The panes can be arranged either vertically or horizontally.The
    user changes the sizes of the panes by dragging the resize handle
    between two panes.

    Subwidgets       Class
    ----------       -----
    <panes>       g/p widgets added dynamically with the add method."""

    def __init__(self, master, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        TixWidget.__init__(self, master, 'tixPanedWindow', ['orientation', 'options'], cnf, kw)

    def add(self, name, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'add', name, *self._options(cnf, kw))
        self.subwidget_list[name] = TixSubWidget(self, name, check_intermediate=0)
        return self.subwidget_list[name]

    def delete(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'delete', name)
        self.subwidget_list[name].destroy()
        del self.subwidget_list[name]

    def forget(self, name):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'forget', name)

    def panecget(self, entry, opt):
        if False:
            return 10
        return self.tk.call(self._w, 'panecget', entry, opt)

    def paneconfigure(self, entry, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        if cnf is None:
            return self._getconfigure(self._w, 'paneconfigure', entry)
        self.tk.call(self._w, 'paneconfigure', entry, *self._options(cnf, kw))

    def panes(self):
        if False:
            return 10
        names = self.tk.splitlist(self.tk.call(self._w, 'panes'))
        return [self.subwidget(x) for x in names]

class PopupMenu(TixWidget):
    """PopupMenu widget can be used as a replacement of the tk_popup command.
    The advantage of the Tix PopupMenu widget is it requires less application
    code to manipulate.


    Subwidgets       Class
    ----------       -----
    menubutton       Menubutton
    menu       Menu"""

    def __init__(self, master, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        TixWidget.__init__(self, master, 'tixPopupMenu', ['options'], cnf, kw)
        self.subwidget_list['menubutton'] = _dummyMenubutton(self, 'menubutton')
        self.subwidget_list['menu'] = _dummyMenu(self, 'menu')

    def bind_widget(self, widget):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'bind', widget._w)

    def unbind_widget(self, widget):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'unbind', widget._w)

    def post_widget(self, widget, x, y):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'post', widget._w, x, y)

class ResizeHandle(TixWidget):
    """Internal widget to draw resize handles on Scrolled widgets."""

    def __init__(self, master, cnf={}, **kw):
        if False:
            return 10
        flags = ['options', 'command', 'cursorfg', 'cursorbg', 'handlesize', 'hintcolor', 'hintwidth', 'x', 'y']
        TixWidget.__init__(self, master, 'tixResizeHandle', flags, cnf, kw)

    def attach_widget(self, widget):
        if False:
            return 10
        self.tk.call(self._w, 'attachwidget', widget._w)

    def detach_widget(self, widget):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'detachwidget', widget._w)

    def hide(self, widget):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'hide', widget._w)

    def show(self, widget):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'show', widget._w)

class ScrolledHList(TixWidget):
    """ScrolledHList - HList with automatic scrollbars."""

    def __init__(self, master, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        TixWidget.__init__(self, master, 'tixScrolledHList', ['options'], cnf, kw)
        self.subwidget_list['hlist'] = _dummyHList(self, 'hlist')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

class ScrolledListBox(TixWidget):
    """ScrolledListBox - Listbox with automatic scrollbars."""

    def __init__(self, master, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        TixWidget.__init__(self, master, 'tixScrolledListBox', ['options'], cnf, kw)
        self.subwidget_list['listbox'] = _dummyListbox(self, 'listbox')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

class ScrolledText(TixWidget):
    """ScrolledText - Text with automatic scrollbars."""

    def __init__(self, master, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        TixWidget.__init__(self, master, 'tixScrolledText', ['options'], cnf, kw)
        self.subwidget_list['text'] = _dummyText(self, 'text')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

class ScrolledTList(TixWidget):
    """ScrolledTList - TList with automatic scrollbars."""

    def __init__(self, master, cnf={}, **kw):
        if False:
            return 10
        TixWidget.__init__(self, master, 'tixScrolledTList', ['options'], cnf, kw)
        self.subwidget_list['tlist'] = _dummyTList(self, 'tlist')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

class ScrolledWindow(TixWidget):
    """ScrolledWindow - Window with automatic scrollbars."""

    def __init__(self, master, cnf={}, **kw):
        if False:
            return 10
        TixWidget.__init__(self, master, 'tixScrolledWindow', ['options'], cnf, kw)
        self.subwidget_list['window'] = _dummyFrame(self, 'window')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

class Select(TixWidget):
    """Select - Container of button subwidgets. It can be used to provide
    radio-box or check-box style of selection options for the user.

    Subwidgets are buttons added dynamically using the add method."""

    def __init__(self, master, cnf={}, **kw):
        if False:
            return 10
        TixWidget.__init__(self, master, 'tixSelect', ['allowzero', 'radio', 'orientation', 'labelside', 'options'], cnf, kw)
        self.subwidget_list['label'] = _dummyLabel(self, 'label')

    def add(self, name, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'add', name, *self._options(cnf, kw))
        self.subwidget_list[name] = _dummyButton(self, name)
        return self.subwidget_list[name]

    def invoke(self, name):
        if False:
            print('Hello World!')
        self.tk.call(self._w, 'invoke', name)

class Shell(TixWidget):
    """Toplevel window.

    Subwidgets - None"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            print('Hello World!')
        TixWidget.__init__(self, master, 'tixShell', ['options', 'title'], cnf, kw)

class DialogShell(TixWidget):
    """Toplevel window, with popup popdown and center methods.
    It tells the window manager that it is a dialog window and should be
    treated specially. The exact treatment depends on the treatment of
    the window manager.

    Subwidgets - None"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        TixWidget.__init__(self, master, 'tixDialogShell', ['options', 'title', 'mapped', 'minheight', 'minwidth', 'parent', 'transient'], cnf, kw)

    def popdown(self):
        if False:
            return 10
        self.tk.call(self._w, 'popdown')

    def popup(self):
        if False:
            return 10
        self.tk.call(self._w, 'popup')

    def center(self):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'center')

class StdButtonBox(TixWidget):
    """StdButtonBox - Standard Button Box (OK, Apply, Cancel and Help) """

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        TixWidget.__init__(self, master, 'tixStdButtonBox', ['orientation', 'options'], cnf, kw)
        self.subwidget_list['ok'] = _dummyButton(self, 'ok')
        self.subwidget_list['apply'] = _dummyButton(self, 'apply')
        self.subwidget_list['cancel'] = _dummyButton(self, 'cancel')
        self.subwidget_list['help'] = _dummyButton(self, 'help')

    def invoke(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name in self.subwidget_list:
            self.tk.call(self._w, 'invoke', name)

class TList(TixWidget, XView, YView):
    """TList - Hierarchy display widget which can be
    used to display data in a tabular format. The list entries of a TList
    widget are similar to the entries in the Tk listbox widget. The main
    differences are (1) the TList widget can display the list entries in a
    two dimensional format and (2) you can use graphical images as well as
    multiple colors and fonts for the list entries.

    Subwidgets - None"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            print('Hello World!')
        TixWidget.__init__(self, master, 'tixTList', ['options'], cnf, kw)

    def active_set(self, index):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'active', 'set', index)

    def active_clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'active', 'clear')

    def anchor_set(self, index):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'anchor', 'set', index)

    def anchor_clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'anchor', 'clear')

    def delete(self, from_, to=None):
        if False:
            return 10
        self.tk.call(self._w, 'delete', from_, to)

    def dragsite_set(self, index):
        if False:
            return 10
        self.tk.call(self._w, 'dragsite', 'set', index)

    def dragsite_clear(self):
        if False:
            while True:
                i = 10
        self.tk.call(self._w, 'dragsite', 'clear')

    def dropsite_set(self, index):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'dropsite', 'set', index)

    def dropsite_clear(self):
        if False:
            while True:
                i = 10
        self.tk.call(self._w, 'dropsite', 'clear')

    def insert(self, index, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'insert', index, *self._options(cnf, kw))

    def info_active(self):
        if False:
            while True:
                i = 10
        return self.tk.call(self._w, 'info', 'active')

    def info_anchor(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tk.call(self._w, 'info', 'anchor')

    def info_down(self, index):
        if False:
            return 10
        return self.tk.call(self._w, 'info', 'down', index)

    def info_left(self, index):
        if False:
            while True:
                i = 10
        return self.tk.call(self._w, 'info', 'left', index)

    def info_right(self, index):
        if False:
            i = 10
            return i + 15
        return self.tk.call(self._w, 'info', 'right', index)

    def info_selection(self):
        if False:
            print('Hello World!')
        c = self.tk.call(self._w, 'info', 'selection')
        return self.tk.splitlist(c)

    def info_size(self):
        if False:
            return 10
        return self.tk.call(self._w, 'info', 'size')

    def info_up(self, index):
        if False:
            print('Hello World!')
        return self.tk.call(self._w, 'info', 'up', index)

    def nearest(self, x, y):
        if False:
            return 10
        return self.tk.call(self._w, 'nearest', x, y)

    def see(self, index):
        if False:
            i = 10
            return i + 15
        self.tk.call(self._w, 'see', index)

    def selection_clear(self, cnf={}, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'selection', 'clear', *self._options(cnf, kw))

    def selection_includes(self, index):
        if False:
            while True:
                i = 10
        return self.tk.call(self._w, 'selection', 'includes', index)

    def selection_set(self, first, last=None):
        if False:
            for i in range(10):
                print('nop')
        self.tk.call(self._w, 'selection', 'set', first, last)

class Tree(TixWidget):
    """Tree - The tixTree widget can be used to display hierarchical
    data in a tree form. The user can adjust
    the view of the tree by opening or closing parts of the tree."""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            while True:
                i = 10
        TixWidget.__init__(self, master, 'tixTree', ['options'], cnf, kw)
        self.subwidget_list['hlist'] = _dummyHList(self, 'hlist')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

    def autosetmode(self):
        if False:
            print('Hello World!')
        'This command calls the setmode method for all the entries in this\n     Tree widget: if an entry has no child entries, its mode is set to\n     none. Otherwise, if the entry has any hidden child entries, its mode is\n     set to open; otherwise its mode is set to close.'
        self.tk.call(self._w, 'autosetmode')

    def close(self, entrypath):
        if False:
            print('Hello World!')
        'Close the entry given by entryPath if its mode is close.'
        self.tk.call(self._w, 'close', entrypath)

    def getmode(self, entrypath):
        if False:
            return 10
        'Returns the current mode of the entry given by entryPath.'
        return self.tk.call(self._w, 'getmode', entrypath)

    def open(self, entrypath):
        if False:
            for i in range(10):
                print('nop')
        'Open the entry given by entryPath if its mode is open.'
        self.tk.call(self._w, 'open', entrypath)

    def setmode(self, entrypath, mode='none'):
        if False:
            for i in range(10):
                print('nop')
        'This command is used to indicate whether the entry given by\n     entryPath has children entries and whether the children are visible. mode\n     must be one of open, close or none. If mode is set to open, a (+)\n     indicator is drawn next the entry. If mode is set to close, a (-)\n     indicator is drawn next the entry. If mode is set to none, no\n     indicators will be drawn for this entry. The default mode is none. The\n     open mode indicates the entry has hidden children and this entry can be\n     opened by the user. The close mode indicates that all the children of the\n     entry are now visible and the entry can be closed by the user.'
        self.tk.call(self._w, 'setmode', entrypath, mode)

class CheckList(TixWidget):
    """The CheckList widget
    displays a list of items to be selected by the user. CheckList acts
    similarly to the Tk checkbutton or radiobutton widgets, except it is
    capable of handling many more items than checkbuttons or radiobuttons.
    """

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            while True:
                i = 10
        TixWidget.__init__(self, master, 'tixCheckList', ['options', 'radio'], cnf, kw)
        self.subwidget_list['hlist'] = _dummyHList(self, 'hlist')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

    def autosetmode(self):
        if False:
            for i in range(10):
                print('nop')
        'This command calls the setmode method for all the entries in this\n     Tree widget: if an entry has no child entries, its mode is set to\n     none. Otherwise, if the entry has any hidden child entries, its mode is\n     set to open; otherwise its mode is set to close.'
        self.tk.call(self._w, 'autosetmode')

    def close(self, entrypath):
        if False:
            while True:
                i = 10
        'Close the entry given by entryPath if its mode is close.'
        self.tk.call(self._w, 'close', entrypath)

    def getmode(self, entrypath):
        if False:
            for i in range(10):
                print('nop')
        'Returns the current mode of the entry given by entryPath.'
        return self.tk.call(self._w, 'getmode', entrypath)

    def open(self, entrypath):
        if False:
            print('Hello World!')
        'Open the entry given by entryPath if its mode is open.'
        self.tk.call(self._w, 'open', entrypath)

    def getselection(self, mode='on'):
        if False:
            i = 10
            return i + 15
        'Returns a list of items whose status matches status. If status is\n     not specified, the list of items in the "on" status will be returned.\n     Mode can be on, off, default'
        return self.tk.splitlist(self.tk.call(self._w, 'getselection', mode))

    def getstatus(self, entrypath):
        if False:
            print('Hello World!')
        'Returns the current status of entryPath.'
        return self.tk.call(self._w, 'getstatus', entrypath)

    def setstatus(self, entrypath, mode='on'):
        if False:
            return 10
        'Sets the status of entryPath to be status. A bitmap will be\n     displayed next to the entry its status is on, off or default.'
        self.tk.call(self._w, 'setstatus', entrypath, mode)

class _dummyButton(Button, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            print('Hello World!')
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyCheckbutton(Checkbutton, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            return 10
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyEntry(Entry, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            while True:
                i = 10
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyFrame(Frame, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            return 10
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyLabel(Label, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            while True:
                i = 10
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyListbox(Listbox, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            while True:
                i = 10
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyMenu(Menu, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            i = 10
            return i + 15
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyMenubutton(Menubutton, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            print('Hello World!')
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyScrollbar(Scrollbar, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            i = 10
            return i + 15
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyText(Text, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            while True:
                i = 10
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyScrolledListBox(ScrolledListBox, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            while True:
                i = 10
        TixSubWidget.__init__(self, master, name, destroy_physically)
        self.subwidget_list['listbox'] = _dummyListbox(self, 'listbox')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

class _dummyHList(HList, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            i = 10
            return i + 15
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyScrolledHList(ScrolledHList, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            print('Hello World!')
        TixSubWidget.__init__(self, master, name, destroy_physically)
        self.subwidget_list['hlist'] = _dummyHList(self, 'hlist')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

class _dummyTList(TList, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            print('Hello World!')
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyComboBox(ComboBox, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            return 10
        TixSubWidget.__init__(self, master, name, ['fancy', destroy_physically])
        self.subwidget_list['label'] = _dummyLabel(self, 'label')
        self.subwidget_list['entry'] = _dummyEntry(self, 'entry')
        self.subwidget_list['arrow'] = _dummyButton(self, 'arrow')
        self.subwidget_list['slistbox'] = _dummyScrolledListBox(self, 'slistbox')
        try:
            self.subwidget_list['tick'] = _dummyButton(self, 'tick')
            self.subwidget_list['cross'] = _dummyButton(self, 'cross')
        except TypeError:
            pass

class _dummyDirList(DirList, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            i = 10
            return i + 15
        TixSubWidget.__init__(self, master, name, destroy_physically)
        self.subwidget_list['hlist'] = _dummyHList(self, 'hlist')
        self.subwidget_list['vsb'] = _dummyScrollbar(self, 'vsb')
        self.subwidget_list['hsb'] = _dummyScrollbar(self, 'hsb')

class _dummyDirSelectBox(DirSelectBox, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            while True:
                i = 10
        TixSubWidget.__init__(self, master, name, destroy_physically)
        self.subwidget_list['dirlist'] = _dummyDirList(self, 'dirlist')
        self.subwidget_list['dircbx'] = _dummyFileComboBox(self, 'dircbx')

class _dummyExFileSelectBox(ExFileSelectBox, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            while True:
                i = 10
        TixSubWidget.__init__(self, master, name, destroy_physically)
        self.subwidget_list['cancel'] = _dummyButton(self, 'cancel')
        self.subwidget_list['ok'] = _dummyButton(self, 'ok')
        self.subwidget_list['hidden'] = _dummyCheckbutton(self, 'hidden')
        self.subwidget_list['types'] = _dummyComboBox(self, 'types')
        self.subwidget_list['dir'] = _dummyComboBox(self, 'dir')
        self.subwidget_list['dirlist'] = _dummyScrolledListBox(self, 'dirlist')
        self.subwidget_list['file'] = _dummyComboBox(self, 'file')
        self.subwidget_list['filelist'] = _dummyScrolledListBox(self, 'filelist')

class _dummyFileSelectBox(FileSelectBox, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            i = 10
            return i + 15
        TixSubWidget.__init__(self, master, name, destroy_physically)
        self.subwidget_list['dirlist'] = _dummyScrolledListBox(self, 'dirlist')
        self.subwidget_list['filelist'] = _dummyScrolledListBox(self, 'filelist')
        self.subwidget_list['filter'] = _dummyComboBox(self, 'filter')
        self.subwidget_list['selection'] = _dummyComboBox(self, 'selection')

class _dummyFileComboBox(ComboBox, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            return 10
        TixSubWidget.__init__(self, master, name, destroy_physically)
        self.subwidget_list['dircbx'] = _dummyComboBox(self, 'dircbx')

class _dummyStdButtonBox(StdButtonBox, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            while True:
                i = 10
        TixSubWidget.__init__(self, master, name, destroy_physically)
        self.subwidget_list['ok'] = _dummyButton(self, 'ok')
        self.subwidget_list['apply'] = _dummyButton(self, 'apply')
        self.subwidget_list['cancel'] = _dummyButton(self, 'cancel')
        self.subwidget_list['help'] = _dummyButton(self, 'help')

class _dummyNoteBookFrame(NoteBookFrame, TixSubWidget):

    def __init__(self, master, name, destroy_physically=0):
        if False:
            print('Hello World!')
        TixSubWidget.__init__(self, master, name, destroy_physically)

class _dummyPanedWindow(PanedWindow, TixSubWidget):

    def __init__(self, master, name, destroy_physically=1):
        if False:
            for i in range(10):
                print('nop')
        TixSubWidget.__init__(self, master, name, destroy_physically)

def OptionName(widget):
    if False:
        while True:
            i = 10
    'Returns the qualified path name for the widget. Normally used to set\n    default options for subwidgets. See tixwidgets.py'
    return widget.tk.call('tixOptionName', widget._w)

def FileTypeList(dict):
    if False:
        for i in range(10):
            print('nop')
    s = ''
    for type in dict.keys():
        s = s + '{{' + type + '} {' + type + ' - ' + dict[type] + '}} '
    return s

class CObjView(TixWidget):
    """This file implements the Canvas Object View widget. This is a base
    class of IconView. It implements automatic placement/adjustment of the
    scrollbars according to the canvas objects inside the canvas subwidget.
    The scrollbars are adjusted so that the canvas is just large enough
    to see all the objects.
    """
    pass

class Grid(TixWidget, XView, YView):
    """The Tix Grid command creates a new window  and makes it into a
    tixGrid widget. Additional options, may be specified on the command
    line or in the option database to configure aspects such as its cursor
    and relief.

    A Grid widget displays its contents in a two dimensional grid of cells.
    Each cell may contain one Tix display item, which may be in text,
    graphics or other formats. See the DisplayStyle class for more information
    about Tix display items. Individual cells, or groups of cells, can be
    formatted with a wide range of attributes, such as its color, relief and
    border.

    Subwidgets - None"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            print('Hello World!')
        static = []
        self.cnf = cnf
        TixWidget.__init__(self, master, 'tixGrid', static, cnf, kw)

    def anchor_clear(self):
        if False:
            while True:
                i = 10
        'Removes the selection anchor.'
        self.tk.call(self, 'anchor', 'clear')

    def anchor_get(self):
        if False:
            print('Hello World!')
        'Get the (x,y) coordinate of the current anchor cell'
        return self._getints(self.tk.call(self, 'anchor', 'get'))

    def anchor_set(self, x, y):
        if False:
            i = 10
            return i + 15
        'Set the selection anchor to the cell at (x, y).'
        self.tk.call(self, 'anchor', 'set', x, y)

    def delete_row(self, from_, to=None):
        if False:
            return 10
        'Delete rows between from_ and to inclusive.\n        If to is not provided,  delete only row at from_'
        if to is None:
            self.tk.call(self, 'delete', 'row', from_)
        else:
            self.tk.call(self, 'delete', 'row', from_, to)

    def delete_column(self, from_, to=None):
        if False:
            print('Hello World!')
        'Delete columns between from_ and to inclusive.\n        If to is not provided,  delete only column at from_'
        if to is None:
            self.tk.call(self, 'delete', 'column', from_)
        else:
            self.tk.call(self, 'delete', 'column', from_, to)

    def edit_apply(self):
        if False:
            while True:
                i = 10
        'If any cell is being edited, de-highlight the cell  and  applies\n        the changes.'
        self.tk.call(self, 'edit', 'apply')

    def edit_set(self, x, y):
        if False:
            while True:
                i = 10
        'Highlights  the  cell  at  (x, y) for editing, if the -editnotify\n        command returns True for this cell.'
        self.tk.call(self, 'edit', 'set', x, y)

    def entrycget(self, x, y, option):
        if False:
            i = 10
            return i + 15
        'Get the option value for cell at (x,y)'
        if option and option[0] != '-':
            option = '-' + option
        return self.tk.call(self, 'entrycget', x, y, option)

    def entryconfigure(self, x, y, cnf=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        return self._configure(('entryconfigure', x, y), cnf, kw)

    def info_exists(self, x, y):
        if False:
            while True:
                i = 10
        'Return True if display item exists at (x,y)'
        return self._getboolean(self.tk.call(self, 'info', 'exists', x, y))

    def info_bbox(self, x, y):
        if False:
            i = 10
            return i + 15
        return self.tk.call(self, 'info', 'bbox', x, y)

    def move_column(self, from_, to, offset):
        if False:
            print('Hello World!')
        'Moves the range of columns from position FROM through TO by\n        the distance indicated by OFFSET. For example, move_column(2, 4, 1)\n        moves the columns 2,3,4 to columns 3,4,5.'
        self.tk.call(self, 'move', 'column', from_, to, offset)

    def move_row(self, from_, to, offset):
        if False:
            print('Hello World!')
        'Moves the range of rows from position FROM through TO by\n        the distance indicated by OFFSET.\n        For example, move_row(2, 4, 1) moves the rows 2,3,4 to rows 3,4,5.'
        self.tk.call(self, 'move', 'row', from_, to, offset)

    def nearest(self, x, y):
        if False:
            i = 10
            return i + 15
        'Return coordinate of cell nearest pixel coordinate (x,y)'
        return self._getints(self.tk.call(self, 'nearest', x, y))

    def set(self, x, y, itemtype=None, **kw):
        if False:
            return 10
        args = self._options(self.cnf, kw)
        if itemtype is not None:
            args = ('-itemtype', itemtype) + args
        self.tk.call(self, 'set', x, y, *args)

    def size_column(self, index, **kw):
        if False:
            for i in range(10):
                print('nop')
        'Queries or sets the size of the column given by\n        INDEX.  INDEX may be any non-negative\n        integer that gives the position of a given column.\n        INDEX can also be the string "default"; in this case, this command\n        queries or sets the default size of all columns.\n        When no option-value pair is given, this command returns a tuple\n        containing the current size setting of the given column.  When\n        option-value pairs are given, the corresponding options of the\n        size setting of the given column are changed. Options may be one\n        of the following:\n              pad0 pixels\n                     Specifies the paddings to the left of a column.\n              pad1 pixels\n                     Specifies the paddings to the right of a column.\n              size val\n                     Specifies the width of a column.  Val may be:\n                     "auto" -- the width of the column is set to the\n                     width of the widest cell in the column;\n                     a valid Tk screen distance unit;\n                     or a real number following by the word chars\n                     (e.g. 3.4chars) that sets the width of the column to the\n                     given number of characters.'
        return self.tk.splitlist(self.tk.call(self._w, 'size', 'column', index, *self._options({}, kw)))

    def size_row(self, index, **kw):
        if False:
            i = 10
            return i + 15
        'Queries or sets the size of the row given by\n        INDEX. INDEX may be any non-negative\n        integer that gives the position of a given row .\n        INDEX can also be the string "default"; in this case, this command\n        queries or sets the default size of all rows.\n        When no option-value pair is given, this command returns a list con-\n        taining the current size setting of the given row . When option-value\n        pairs are given, the corresponding options of the size setting of the\n        given row are changed. Options may be one of the following:\n              pad0 pixels\n                     Specifies the paddings to the top of a row.\n              pad1 pixels\n                     Specifies the paddings to the bottom of a row.\n              size val\n                     Specifies the height of a row.  Val may be:\n                     "auto" -- the height of the row is set to the\n                     height of the highest cell in the row;\n                     a valid Tk screen distance unit;\n                     or a real number following by the word chars\n                     (e.g. 3.4chars) that sets the height of the row to the\n                     given number of characters.'
        return self.tk.splitlist(self.tk.call(self, 'size', 'row', index, *self._options({}, kw)))

    def unset(self, x, y):
        if False:
            return 10
        'Clears the cell at (x, y) by removing its display item.'
        self.tk.call(self._w, 'unset', x, y)

class ScrolledGrid(Grid):
    """Scrolled Grid widgets"""

    def __init__(self, master=None, cnf={}, **kw):
        if False:
            i = 10
            return i + 15
        static = []
        self.cnf = cnf
        TixWidget.__init__(self, master, 'tixScrolledGrid', static, cnf, kw)