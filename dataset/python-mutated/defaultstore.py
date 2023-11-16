from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
from renpy.minstore import *
import renpy
import renpy.display.im as im
import renpy.display.anim as anim
_restart = None
_return = None
_args = None
_kwargs = None
_window = False
_window_subtitle = ''
_rollback = True
_begin_rollback = True
_skipping = True
_dismiss_pause = True
_config = renpy.config
_widget_by_id = None
_widget_properties = {}
_text_rect = None
_menu = False
main_menu = False
_autosave = True
_live2d_fade = True

class _Config(object):

    def __getstate__(self):
        if False:
            while True:
                i = 10
        return None

    def __setstate__(self, data):
        if False:
            print('Hello World!')
        return

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        cvars = vars(_config)
        if name not in cvars:
            raise Exception('config.%s is not a known configuration variable.' % name)
        return cvars[name]

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        cvars = _config.__dict__
        if name not in cvars and renpy.config.locked:
            raise Exception('config.%s is not a known configuration variable.' % name)
        if name == 'script_version':
            renpy.store._set_script_version(value)
        if name == 'developer':
            if value == 'auto':
                renpy.config.original_developer = value
                renpy.config.developer = renpy.config.default_developer
                return
        cvars[name] = value

    def __delattr__(self, name):
        if False:
            while True:
                i = 10
        if renpy.config.locked:
            raise Exception('Deleting configuration variables is not supported.')
        else:
            delattr(renpy.config, name)
style = None
config = _Config()
library = config
eval = renpy.python.py_eval
Bar = renpy.display.behavior.Bar
Button = renpy.display.behavior.Button
ImageButton = renpy.display.behavior.ImageButton
Input = renpy.display.behavior.Input
TextButton = renpy.display.behavior.TextButton
ImageReference = renpy.display.image.ImageReference
DynamicImage = renpy.display.image.DynamicImage
Image = renpy.display.im.image
Frame = renpy.display.imagelike.Frame
Borders = renpy.display.imagelike.Borders
Solid = renpy.display.imagelike.Solid
FileCurrentScreenshot = renpy.display.imagelike.FileCurrentScreenshot
LiveComposite = renpy.display.layout.LiveComposite
LiveCrop = renpy.display.layout.LiveCrop
LiveTile = renpy.display.layout.LiveTile
Composite = renpy.display.layout.Composite
Crop = renpy.display.layout.Crop
Tile = renpy.display.layout.Tile
Flatten = renpy.display.layout.Flatten
Null = renpy.display.layout.Null
Window = renpy.display.layout.Window
Viewport = renpy.display.viewport.Viewport
DynamicDisplayable = renpy.display.layout.DynamicDisplayable
ConditionSwitch = renpy.display.layout.ConditionSwitch
ShowingSwitch = renpy.display.layout.ShowingSwitch
AlphaMask = renpy.display.layout.AlphaMask
Layer = renpy.display.layout.Layer
Transform = renpy.display.transform.Transform
Camera = renpy.display.transform.Camera
Animation = anim.Animation
Movie = renpy.display.video.Movie
Particles = renpy.display.particle.Particles
SnowBlossom = renpy.display.particle.SnowBlossom
Text = renpy.text.text.Text
ParameterizedText = renpy.text.extras.ParameterizedText
FontGroup = renpy.text.font.FontGroup
Drag = renpy.display.dragdrop.Drag
DragGroup = renpy.display.dragdrop.DragGroup
Sprite = renpy.display.particle.Sprite
SpriteManager = renpy.display.particle.SpriteManager
Matrix = renpy.display.matrix.Matrix
Live2D = renpy.gl2.live2d.Live2D
Model = renpy.display.model.Model
Alpha = renpy.curry.curry(renpy.display.layout.Alpha)
Position = renpy.curry.curry(renpy.display.layout.Position)
Pan = renpy.curry.curry(renpy.display.motion.Pan)
Move = renpy.curry.curry(renpy.display.motion.Move)
Motion = renpy.curry.curry(renpy.display.motion.Motion)
Revolve = renpy.curry.curry(renpy.display.motion.Revolve)
Zoom = renpy.curry.curry(renpy.display.motion.Zoom)
RotoZoom = renpy.curry.curry(renpy.display.motion.RotoZoom)
FactorZoom = renpy.curry.curry(renpy.display.motion.FactorZoom)
SizeZoom = renpy.curry.curry(renpy.display.motion.SizeZoom)
Fade = renpy.curry.curry(renpy.display.transition.Fade)
Dissolve = renpy.curry.curry(renpy.display.transition.Dissolve)
ImageDissolve = renpy.curry.curry(renpy.display.transition.ImageDissolve)
AlphaDissolve = renpy.curry.curry(renpy.display.transition.AlphaDissolve)
CropMove = renpy.curry.curry(renpy.display.transition.CropMove)
PushMove = renpy.curry.curry(renpy.display.transition.PushMove)
Pixellate = renpy.curry.curry(renpy.display.transition.Pixellate)
OldMoveTransition = renpy.curry.curry(renpy.display.movetransition.OldMoveTransition)
MoveTransition = renpy.curry.curry(renpy.display.movetransition.MoveTransition)
MoveFactory = renpy.curry.curry(renpy.display.movetransition.MoveFactory)
MoveIn = renpy.curry.curry(renpy.display.movetransition.MoveIn)
MoveOut = renpy.curry.curry(renpy.display.movetransition.MoveOut)
ZoomInOut = renpy.curry.curry(renpy.display.movetransition.ZoomInOut)
RevolveInOut = renpy.curry.curry(renpy.display.movetransition.RevolveInOut)
MultipleTransition = renpy.curry.curry(renpy.display.transition.MultipleTransition)
ComposeTransition = renpy.curry.curry(renpy.display.transition.ComposeTransition)
Pause = renpy.curry.curry(renpy.display.transition.NoTransition)
SubTransition = renpy.curry.curry(renpy.display.transition.SubTransition)
ADVSpeaker = ADVCharacter = renpy.character.ADVCharacter
Speaker = Character = renpy.character.Character
DynamicCharacter = renpy.character.DynamicCharacter
MultiPersistent = renpy.persistent.MultiPersistent
Action = renpy.ui.Action
BarValue = renpy.ui.BarValue
AudioData = renpy.audio.audio.AudioData
Style = renpy.style.Style
SlottedNoRollback = renpy.rollback.SlottedNoRollback
NoRollback = renpy.rollback.NoRollback

class _layout_class(__builtins__['object']):
    """
    This is used to generate declarative versions of MultiBox and Grid.
    """

    def __init__(self, cls, doc, nargs=0, **extra_kwargs):
        if False:
            i = 10
            return i + 15
        self.cls = cls
        self.nargs = nargs
        self.extra_kwargs = extra_kwargs
        self.__doc__ = doc

    def __call__(self, *args, **properties):
        if False:
            while True:
                i = 10
        conargs = args[:self.nargs]
        kids = args[self.nargs:]
        kwargs = self.extra_kwargs.copy()
        kwargs.update(properties)
        rv = self.cls(*conargs, **kwargs)
        for i in kids:
            rv.add(renpy.easy.displayable(i))
        return rv
Fixed = _layout_class(renpy.display.layout.MultiBox, '\n:name: Fixed\n:doc: disp_box\n:args: (*args, **properties)\n\nA box that fills the screen. Its members are laid out\nfrom back to front, with their position properties\ncontrolling their position.\n', layout='fixed')
HBox = _layout_class(renpy.display.layout.MultiBox, '\n:doc: disp_box\n:args: (*args, **properties)\n\nA box that lays out its members from left to right.\n', layout='horizontal')
VBox = _layout_class(renpy.display.layout.MultiBox, '\n:doc: disp_box\n:args: (*args, **properties)\n\nA layout that lays out its members from top to bottom.\n', layout='vertical')
Grid = _layout_class(renpy.display.layout.Grid, '\n:doc: disp_grid\n:args: (cols, rows, *args, **properties)\n\nLays out displayables in a grid. The first two positional arguments\nare the number of columns and rows in the grid. This must be followed\nby `columns * rows` positional arguments giving the displayables that\nfill the grid.\n', nargs=2, layout='vertical')

def AlphaBlend(control, old, new, alpha=False):
    if False:
        while True:
            i = 10
    "\n    :doc: disp_effects\n\n    This transition uses a `control` displayable (almost always some sort of\n    animated transform) to transition from one displayable to another. The\n    transform is evaluated. The `new` displayable is used where the transform\n    is opaque, and the `old` displayable is used when it is transparent.\n\n    `alpha`\n        If true, the image is composited with what's behind it. If false,\n        the default, the image is opaque and overwrites what's behind it.\n    "
    return renpy.display.transition.AlphaDissolve(control, 0.0, old_widget=old, new_widget=new, alpha=alpha)

def At(d, *args):
    if False:
        while True:
            i = 10
    '\n    :doc: disp_at\n    :name: At\n\n    Given a displayable `d`, applies each of the transforms in `args`\n    to it. The transforms are applied in left-to-right order, so that\n    the outermost transform is the rightmost argument. ::\n\n        transform birds_transform:\n            xpos -200\n            linear 10 xpos 800\n            pause 20\n            repeat\n\n        image birds = At("birds.png", birds_transform)\n        '
    rv = renpy.easy.displayable(d)
    for i in args:
        if isinstance(i, renpy.display.motion.Transform):
            rv = i(child=rv)
        else:
            rv = i(rv)
    return rv
Color = renpy.color.Color
color = renpy.color.Color
menu = renpy.exports.display_menu
predict_menu = renpy.exports.predict_menu
default_transition = None
mouse_visible = True
suppress_overlay = False
adv = ADVCharacter(None, who_prefix='', who_suffix='', what_prefix='', what_suffix='', show_function=renpy.exports.show_display_say, predict_function=renpy.exports.predict_show_display_say, condition=None, dynamic=False, image=None, interact=True, slow=True, slow_abortable=True, afm=True, ctc=None, ctc_pause=None, ctc_timedpause=None, ctc_position='nestled', all_at_once=False, with_none=None, callback=None, type='say', advance=True, retain=False, who_style='say_label', what_style='say_dialogue', window_style='say_window', screen='say', mode='say', voice_tag=None, kind=False)
name_only = adv

def predict_say(who, what):
    if False:
        return 10
    who = Character(who, kind=adv)
    try:
        who.predict(what)
    except Exception:
        pass

def say(who, what, interact=True, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    who = Character(who, kind=adv)
    who(what, *args, interact=interact, **kwargs)
_last_say_who = None
_last_say_what = None
_last_say_args = ()
_last_say_kwargs = {}
_cache_pin_set = set()
_predict_set = set()
_predict_screen = dict()
_overlay_screens = None
_in_replay = None
_side_image_attributes = None
_side_image_attributes_reset = False
main_menu = False
_ignore_action = None
_quit_slot = None
_screenshot_pattern = None
import sys
import os
globals()['renpy'] = renpy.exports