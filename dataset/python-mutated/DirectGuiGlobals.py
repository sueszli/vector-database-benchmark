"""
Global definitions used by Direct Gui Classes and handy constants
that can be used during widget construction
"""
__all__ = ()
from panda3d.core import KeyboardButton, MouseButton, PGButton, PGEntry, PGFrameStyle, PGSliderBar, TextNode
defaultFont = None
defaultFontFunc = TextNode.getDefaultFont
defaultClickSound = None
defaultRolloverSound = None
defaultDialogGeom = None
defaultDialogRelief = PGFrameStyle.TBevelOut
drawOrder = 100
panel = None
INITOPT = ['initopt']
LMB = 0
MMB = 1
RMB = 2
NORMAL = 'normal'
DISABLED = 'disabled'
FLAT = PGFrameStyle.TFlat
RAISED = PGFrameStyle.TBevelOut
SUNKEN = PGFrameStyle.TBevelIn
GROOVE = PGFrameStyle.TGroove
RIDGE = PGFrameStyle.TRidge
TEXTUREBORDER = PGFrameStyle.TTextureBorder
FrameStyleDict = {'flat': FLAT, 'raised': RAISED, 'sunken': SUNKEN, 'groove': GROOVE, 'ridge': RIDGE, 'texture_border': TEXTUREBORDER}
HORIZONTAL = 'horizontal'
VERTICAL = 'vertical'
VERTICAL_INVERTED = 'vertical_inverted'
DIALOG_NO = 0
DIALOG_OK = DIALOG_YES = DIALOG_RETRY = 1
DIALOG_CANCEL = -1
DESTROY = 'destroy-'
PRINT = 'print-'
ENTER = PGButton.getEnterPrefix()
EXIT = PGButton.getExitPrefix()
WITHIN = PGButton.getWithinPrefix()
WITHOUT = PGButton.getWithoutPrefix()
B1CLICK = PGButton.getClickPrefix() + MouseButton.one().getName() + '-'
B2CLICK = PGButton.getClickPrefix() + MouseButton.two().getName() + '-'
B3CLICK = PGButton.getClickPrefix() + MouseButton.three().getName() + '-'
B1PRESS = PGButton.getPressPrefix() + MouseButton.one().getName() + '-'
B2PRESS = PGButton.getPressPrefix() + MouseButton.two().getName() + '-'
B3PRESS = PGButton.getPressPrefix() + MouseButton.three().getName() + '-'
B1RELEASE = PGButton.getReleasePrefix() + MouseButton.one().getName() + '-'
B2RELEASE = PGButton.getReleasePrefix() + MouseButton.two().getName() + '-'
B3RELEASE = PGButton.getReleasePrefix() + MouseButton.three().getName() + '-'
WHEELUP = PGButton.getReleasePrefix() + MouseButton.wheelUp().getName() + '-'
WHEELDOWN = PGButton.getReleasePrefix() + MouseButton.wheelDown().getName() + '-'
OVERFLOW = PGEntry.getOverflowPrefix()
ACCEPT = PGEntry.getAcceptPrefix() + KeyboardButton.enter().getName() + '-'
ACCEPTFAILED = PGEntry.getAcceptFailedPrefix() + KeyboardButton.enter().getName() + '-'
TYPE = PGEntry.getTypePrefix()
ERASE = PGEntry.getErasePrefix()
CURSORMOVE = PGEntry.getCursormovePrefix()
ADJUST = PGSliderBar.getAdjustPrefix()
IMAGE_SORT_INDEX = 10
GEOM_SORT_INDEX = 20
TEXT_SORT_INDEX = 30
FADE_SORT_INDEX = 1000
NO_FADE_SORT_INDEX = 2000
BACKGROUND_SORT_INDEX = -100
MIDGROUND_SORT_INDEX = 0
FOREGROUND_SORT_INDEX = 100
_OPT_DEFAULT = 0
_OPT_VALUE = 1
_OPT_FUNCTION = 2
BUTTON_READY_STATE = PGButton.SReady
BUTTON_DEPRESSED_STATE = PGButton.SDepressed
BUTTON_ROLLOVER_STATE = PGButton.SRollover
BUTTON_INACTIVE_STATE = PGButton.SInactive

def getDefaultRolloverSound():
    if False:
        i = 10
        return i + 15
    return defaultRolloverSound

def setDefaultRolloverSound(newSound):
    if False:
        print('Hello World!')
    global defaultRolloverSound
    defaultRolloverSound = newSound

def getDefaultClickSound():
    if False:
        i = 10
        return i + 15
    return defaultClickSound

def setDefaultClickSound(newSound):
    if False:
        for i in range(10):
            print('nop')
    global defaultClickSound
    defaultClickSound = newSound

def getDefaultFont():
    if False:
        return 10
    global defaultFont
    if defaultFont is None:
        defaultFont = defaultFontFunc()
    return defaultFont

def setDefaultFont(newFont):
    if False:
        i = 10
        return i + 15
    'Changes the default font for DirectGUI items.  To change the default\n    font across the board, see :meth:`.TextNode.setDefaultFont`. '
    global defaultFont
    defaultFont = newFont

def setDefaultFontFunc(newFontFunc):
    if False:
        for i in range(10):
            print('nop')
    global defaultFontFunc
    defaultFontFunc = newFontFunc

def getDefaultDialogGeom():
    if False:
        return 10
    return defaultDialogGeom

def getDefaultDialogRelief():
    if False:
        print('Hello World!')
    return defaultDialogRelief

def setDefaultDialogGeom(newDialogGeom, relief=None):
    if False:
        for i in range(10):
            print('nop')
    global defaultDialogGeom, defaultDialogRelief
    defaultDialogGeom = newDialogGeom
    defaultDialogRelief = relief

def getDefaultDrawOrder():
    if False:
        for i in range(10):
            print('nop')
    return drawOrder

def setDefaultDrawOrder(newDrawOrder):
    if False:
        print('Hello World!')
    global drawOrder
    drawOrder = newDrawOrder

def getDefaultPanel():
    if False:
        while True:
            i = 10
    return panel

def setDefaultPanel(newPanel):
    if False:
        return 10
    global panel
    panel = newPanel
get_default_rollover_sound = getDefaultRolloverSound
set_default_rollover_sound = setDefaultRolloverSound
get_default_click_sound = getDefaultClickSound
set_default_click_sound = setDefaultClickSound
get_default_font = getDefaultFont
set_default_font = setDefaultFont
get_default_dialog_geom = getDefaultDialogGeom
get_default_dialog_relief = getDefaultDialogRelief
set_default_dialog_geom = setDefaultDialogGeom
get_default_draw_order = getDefaultDrawOrder
set_default_draw_order = setDefaultDrawOrder
get_default_panel = getDefaultPanel
set_default_panel = setDefaultPanel