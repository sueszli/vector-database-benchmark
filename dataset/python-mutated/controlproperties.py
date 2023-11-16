"""Wrap"""
from .windows.win32structures import RECT, LOGFONTW
from . import deprecated

class FuncWrapper(object):
    """Little class to allow attribute access to return a callable object"""

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self.value = value

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Return the saved value'
        return self.value

class ControlProps(dict):
    """Wrap controls read from a file to resemble hwnd controls"""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        dict.__init__(self, *args, **kwargs)
        self.ref = None

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        if attr not in self and attr + 's' in self:
            return FuncWrapper(self[attr + 's'][0])
        return FuncWrapper(self[attr])

    def window_text(self):
        if False:
            for i in range(10):
                print('nop')
        return self['texts'][0]
    WindowText = deprecated(window_text)

    def has_style(self, style):
        if False:
            i = 10
            return i + 15
        return self['style'] & style == style
    HasStyle = deprecated(has_style)

    def has_exstyle(self, exstyle):
        if False:
            print('Hello World!')
        return self['exstyle'] & exstyle == exstyle
    HasExStyle = deprecated(has_exstyle, deprecated_name='HasExStyle')

def GetMenuBlocks(ctrls):
    if False:
        for i in range(10):
            print('nop')
    allMenuBlocks = []
    for ctrl in ctrls:
        if 'menu_items' in ctrl.keys():
            menuBlocks = MenuBlockAsControls(ctrl.menu_items())
            allMenuBlocks.extend(menuBlocks)
    return allMenuBlocks

def MenuBlockAsControls(menuItems, parentage=None):
    if False:
        for i in range(10):
            print('nop')
    if parentage is None:
        parentage = []
    blocks = []
    curBlock = []
    for item in menuItems:
        itemAsCtrl = MenuItemAsControl(item)
        if parentage:
            itemPath = '%s->%s' % ('->'.join(parentage), item['text'])
        else:
            itemPath = item['text']
        curBlock.append(itemAsCtrl)
        if 'menu_items' in item.keys():
            parentage.append(item['text'])
            blocks.extend(MenuBlockAsControls(item['menu_items']['menu_items'], parentage))
            del parentage[-1]
    blocks.append(curBlock)
    return blocks

def MenuItemAsControl(menuItem):
    if False:
        while True:
            i = 10
    'Make a menu item look like a control for tests'
    itemAsCtrl = ControlProps()
    itemAsCtrl['texts'] = [menuItem['text']]
    itemAsCtrl['control_id'] = menuItem['id']
    itemAsCtrl['type'] = menuItem['type']
    itemAsCtrl['state'] = menuItem['state']
    itemAsCtrl['class_name'] = 'MenuItem'
    itemAsCtrl['friendly_class_name'] = 'MenuItem'
    itemAsCtrl['rectangle'] = RECT(0, 0, 999, 999)
    itemAsCtrl['fonts'] = [LOGFONTW()]
    itemAsCtrl['client_rects'] = [RECT(0, 0, 999, 999)]
    itemAsCtrl['context_help_id'] = 0
    itemAsCtrl['user_data'] = 0
    itemAsCtrl['style'] = 0
    itemAsCtrl['exstyle'] = 0
    itemAsCtrl['is_visible'] = 1
    return itemAsCtrl

def SetReferenceControls(controls, refControls):
    if False:
        print('Hello World!')
    "Set the reference controls for the controls passed in\n\n    This does some minor checking as following:\n     * test that there are the same number of reference controls as\n       controls - fails with an exception if there are not\n     * test if all the ID's are the same or not\n    "
    if len(controls) != len(refControls):
        raise RuntimeError('Numbers of controls on ref. dialog does not match Loc. dialog')
    for (i, ctrl) in enumerate(controls):
        ctrl.ref = refControls[i]
    toRet = 1
    allIDsSameFlag = 2
    allClassesSameFlag = 4
    if [ctrl.control_id() for ctrl in controls] == [ctrl.control_id() for ctrl in refControls]:
        toRet += allIDsSameFlag
    if [ctrl.class_name() for ctrl in controls] == [ctrl.class_name() for ctrl in refControls]:
        toRet += allClassesSameFlag
    return toRet