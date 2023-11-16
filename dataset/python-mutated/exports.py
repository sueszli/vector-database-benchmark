from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import gc
import io
import re
import renpy
from renpy.pyanalysis import const, pure, not_const
try:
    import emscripten
except ImportError:
    pass

def renpy_pure(fn):
    if False:
        print('Hello World!')
    '\n    Marks renpy.`fn` as a pure function.\n    '
    name = fn
    if not isinstance(name, basestring):
        name = fn.__name__
    pure('renpy.' + name)
    return fn
import pygame_sdl2
from renpy.text.extras import ParameterizedText, filter_text_tags
from renpy.text.font import register_sfont, register_mudgefont, register_bmfont
from renpy.text.text import language_tailor, BASELINE
from renpy.display.behavior import Keymap
from renpy.display.behavior import run, run as run_action, run_unhovered, run_periodic
from renpy.display.behavior import map_event, queue_event, clear_keymap_cache
from renpy.display.behavior import is_selected, is_sensitive
from renpy.display.minigame import Minigame
from renpy.display.screen import define_screen, show_screen, hide_screen, use_screen, current_screen
from renpy.display.screen import has_screen, get_screen, get_displayable, get_widget, ScreenProfile as profile_screen
from renpy.display.screen import get_displayable_properties, get_widget_properties
from renpy.display.focus import focus_coordinates, capture_focus, clear_capture_focus, get_focus_rect
from renpy.display.predict import screen as predict_screen
from renpy.display.image import image_exists, image_exists as has_image, list_images
from renpy.display.image import get_available_image_tags, get_available_image_attributes, check_image_attributes, get_ordered_image_attributes
from renpy.display.image import get_registered_image
from renpy.display.im import load_surface, load_image, load_rgba
from renpy.curry import curry, partial
from renpy.display.video import movie_start_fullscreen, movie_start_displayable, movie_stop
from renpy.loadsave import load, save, list_saved_games, can_load, rename_save, copy_save, unlink_save, scan_saved_game
from renpy.loadsave import list_slots, newest_slot, slot_mtime, slot_json, slot_screenshot, force_autosave
from renpy.savetoken import get_save_token_keys
from renpy.python import py_eval as eval
from renpy.rollback import rng as random
from renpy.atl import atl_warper
from renpy.easy import predict, displayable, split_properties
from renpy.lexer import unelide_filename
from renpy.parser import get_parse_errors
from renpy.translation import change_language, known_languages, translate_string, get_translation_identifier
from renpy.translation.generation import generic_filter as transform_text
from renpy.persistent import register_persistent
from renpy.character import show_display_say, predict_show_display_say, display_say
import renpy.audio.sound as sound
import renpy.audio.music as music
from renpy.statements import register as register_statement
from renpy.text.extras import check_text_tags
from renpy.memory import profile_memory, diff_memory, profile_rollback
from renpy.text.font import variable_font_info
from renpy.text.textsupport import TAG as TEXT_TAG, TEXT as TEXT_TEXT, PARAGRAPH as TEXT_PARAGRAPH, DISPLAYABLE as TEXT_DISPLAYABLE
from renpy.execution import not_infinite_loop, reset_all_contexts
from renpy.sl2.slparser import CustomParser as register_sl_statement, register_sl_displayable
from renpy.ast import eval_who
from renpy.loader import add_python_directory
from renpy.lint import try_compile, try_eval
from renpy.gl2.gl2shadercache import register_shader
from renpy.gl2.live2d import has_live2d
from renpy.bootstrap import get_alternate_base
renpy_pure('ParameterizedText')
renpy_pure('Keymap')
renpy_pure('has_screen')
renpy_pure('image_exists')
renpy_pure('curry')
renpy_pure('partial')
renpy_pure('unelide_filename')
renpy_pure('known_languages')
renpy_pure('check_text_tags')
renpy_pure('filter_text_tags')
renpy_pure('split_properties')
import time
import sys
import threading
import fnmatch
if sys.maxsize > 2 << 32:
    bits = 64
else:
    bits = 32

def roll_forward_info():
    if False:
        while True:
            i = 10
    '\n    :doc: rollback\n\n    When in rollback, returns the data that was supplied to :func:`renpy.checkpoint`\n    the last time this statement executed. Outside of rollback, returns None.\n    '
    if not renpy.game.context().rollback:
        return None
    return renpy.game.log.forward_info()

def roll_forward_core(value=None):
    if False:
        while True:
            i = 10
    '\n    :undocumented:\n\n    To cause a roll_forward to occur, return the value of this function\n    from an event handler.\n    '
    if value is None:
        value = roll_forward_info()
    if value is None:
        return
    renpy.game.interface.suppress_transition = True
    renpy.game.after_rollback = True
    renpy.game.log.rolled_forward = True
    return value

def in_rollback():
    if False:
        i = 10
        return i + 15
    '\n    :doc: rollback\n\n    Returns true if the game has been rolled back.\n    '
    return renpy.game.log.in_rollback() or renpy.game.after_rollback

def can_rollback():
    if False:
        while True:
            i = 10
    '\n    :doc: rollback\n\n    Returns true if we can rollback.\n    '
    if not renpy.config.rollback_enabled:
        return False
    return renpy.game.log.can_rollback()

def in_fixed_rollback():
    if False:
        print('Hello World!')
    '\n    :doc: blockrollback\n\n    Returns true if rollback is currently occurring and the current\n    context is before an executed renpy.fix_rollback() statement.\n    '
    return renpy.game.log.in_fixed_rollback()

def checkpoint(data=None, keep_rollback=None, hard=True):
    if False:
        return 10
    '\n    :doc: rollback\n    :args: (data=None, *, hard=True)\n\n    Makes the current statement a checkpoint that the user can rollback to. Once\n    this function has been called, there should be no more interaction with the\n    user in the current statement.\n\n    This will also clear the current screenshot used by saved games.\n\n    `data`\n        This data is returned by :func:`renpy.roll_forward_info` when the\n        game is being rolled back.\n\n    `hard`\n        If true, this is a hard checkpoint that rollback will stop at. If false,\n        this is a soft checkpoint that will not stop rollback.\n    '
    if keep_rollback is None:
        keep_rollback = renpy.config.keep_rollback_data
    renpy.game.log.checkpoint(data, keep_rollback=keep_rollback, hard=renpy.store._rollback and hard)
    if renpy.store._rollback and renpy.config.auto_clear_screenshot:
        renpy.game.interface.clear_screenshot = True

def block_rollback(purge=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: blockrollback\n    :args: ()\n\n    Prevents the game from rolling back to before the current\n    statement.\n    '
    renpy.game.log.block(purge=purge)

def suspend_rollback(flag):
    if False:
        while True:
            i = 10
    '\n    :doc: rollback\n    :args: (flag)\n\n    Rollback will skip sections of the game where rollback has been\n    suspended.\n\n    `flag`:\n        When `flag` is true, rollback is suspended. When false,\n        rollback is resumed.\n    '
    renpy.game.log.suspend_checkpointing(flag)

def fix_rollback():
    if False:
        print('Hello World!')
    '\n    :doc: blockrollback\n\n    Prevents the user from changing decisions made before the current\n    statement.\n    '
    renpy.game.log.fix_rollback()

def retain_after_load():
    if False:
        while True:
            i = 10
    '\n    :doc: retain_after_load\n\n    Causes data modified between the current statement and the statement\n    containing the next checkpoint to be retained when a load occurs.\n    '
    renpy.game.log.retain_after_load()
scene_lists = renpy.display.core.scene_lists

def count_displayables_in_layer(layer):
    if False:
        print('Hello World!')
    '\n    Returns how many displayables are in the supplied layer.\n    '
    sls = scene_lists()
    return len(sls.layers[layer])

def image(name, d):
    if False:
        i = 10
        return i + 15
    '\n    :doc: se_images\n\n    Defines an image. This function is the Python equivalent of the\n    image statement.\n\n    `name`\n        The name of the image to display, a string.\n\n    `d`\n        The displayable to associate with that image name.\n\n    This function may only be run from inside an init block. It is an\n    error to run this function once the game has started.\n    '
    if d is None:
        raise Exception('Images may not be declared to be None.')
    if not renpy.game.context().init_phase:
        raise Exception('Images may only be declared inside init blocks.')
    if not isinstance(name, tuple):
        name = tuple(name.split())
    d = renpy.easy.displayable(d)
    renpy.display.image.register_image(name, d)

def copy_images(old, new):
    if False:
        while True:
            i = 10
    '\n    :doc: image_func\n\n    Copies images beginning with one prefix to images beginning with\n    another. For example::\n\n        renpy.copy_images("eileen", "eileen2")\n\n    will create an image beginning with "eileen2" for every image beginning\n    with "eileen". If "eileen happy" exists, "eileen2 happy" will be\n    created.\n\n    `old`\n        A space-separated string giving the components of the old image\n        name.\n\n    `new`\n        A space-separated string giving the components of the new image\n        name.\n    '
    if not isinstance(old, tuple):
        old = tuple(old.split())
    if not isinstance(new, tuple):
        new = tuple(new.split())
    lenold = len(old)
    for (k, v) in renpy.display.image.images.items():
        if len(k) < lenold:
            continue
        if k[:lenold] == old:
            renpy.display.image.register_image(new + k[lenold:], v)

def default_layer(layer, tag, expression=False):
    if False:
        i = 10
        return i + 15
    '\n    :undocumented:\n\n    If layer is not None, returns it. Otherwise, interprets `tag` as a name\n    or tag, then looks up what the default layer for that tag is, and returns\n    the result.\n    '
    if layer is not None:
        return layer
    if tag is None or expression:
        return renpy.config.default_tag_layer
    if isinstance(tag, tuple):
        tag = tag[0]
    elif ' ' in tag:
        tag = tag.split()[0]
    return scene_lists().sticky_tags.get(tag, None) or renpy.config.tag_layer.get(tag, renpy.config.default_tag_layer)

def can_show(name, layer=None, tag=None):
    if False:
        print('Hello World!')
    '\n    :doc: image_func\n\n    Determines if `name` can be used to show an image. This interprets `name`\n    as a tag and attributes. This is combined with the attributes of the\n    currently-showing image with `tag` on `layer` to try to determine a unique image\n    to show. If a unique image can be show, returns the name of that image as\n    a tuple. Otherwise, returns None.\n\n    `tag`\n        The image tag to get attributes from. If not given, defaults to the first\n        component of `name`.\n\n    `layer`\n        The layer to check. If None, uses the default layer for `tag`.\n    '
    if not isinstance(name, tuple):
        name = tuple(name.split())
    if tag is None:
        tag = name[0]
    layer = default_layer(layer, tag)
    try:
        return renpy.game.context().images.apply_attributes(layer, tag, name)
    except Exception:
        return None

def showing(name, layer=None):
    if False:
        while True:
            i = 10
    '\n    :doc: image_func\n\n    Returns true if an image with the same tag as `name` is showing on\n    `layer`.\n\n    `image`\n        May be a string giving the image name or a tuple giving each\n        component of the image name. It may also be a string giving\n        only the image tag.\n\n    `layer`\n        The layer to check. If None, uses the default layer for `tag`.\n    '
    if not isinstance(name, tuple):
        name = tuple(name.split())
    layer = default_layer(layer, name)
    return renpy.game.context().images.showing(layer, name)

def get_showing_tags(layer='master', sort=False):
    if False:
        while True:
            i = 10
    '\n    :doc: image_func\n\n    Returns the set of image tags that are currently being shown on `layer`. If\n    sort is true, returns a list of the tags from back to front.\n    '
    if sort:
        return scene_lists().get_sorted_tags(layer)
    return renpy.game.context().images.get_showing_tags(layer)

def get_hidden_tags(layer='master'):
    if False:
        print('Hello World!')
    '\n    :doc: image_func\n\n    Returns the set of image tags on `layer` that are currently hidden, but\n    still have attribute information associated with them.\n    '
    return renpy.game.context().images.get_hidden_tags(layer)

def get_attributes(tag, layer=None, if_hidden=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: image_func\n\n    Return a tuple giving the image attributes for the image `tag`. If\n    the image tag has not had any attributes associated since the last\n    time it was hidden, returns `if_hidden`.\n\n    `layer`\n        The layer to check. If None, uses the default layer for `tag`.\n    '
    layer = default_layer(layer, tag)
    return renpy.game.context().images.get_attributes(layer, tag, if_hidden)

def clear_attributes(tag, layer=None):
    if False:
        print('Hello World!')
    '\n    :doc: image_func\n\n    Clears all image attributes for the `tag` image.\n    If the tag had no attached image attributes, this does nothing.\n\n    `layer`\n        The layer to check. If None, uses the default layer for `tag`.\n    '
    current = get_attributes(tag, layer, None)
    if not current:
        return
    shown = showing(tag, default_layer(layer, tag))
    current = tuple(('-' + a for a in current))
    set_tag_attributes((tag,) + current, layer)
    if shown:
        show(tag, layer=layer)

def _find_image(layer, key, name, what):
    if False:
        return 10
    '\n    :undocumented:\n\n    Finds an image to show.\n    '
    if what is not None:
        if isinstance(what, basestring):
            what = tuple(what.split())
        return (name, what)
    if renpy.config.image_attributes:
        new_image = renpy.game.context().images.apply_attributes(layer, key, name)
        if new_image is not None:
            image = new_image
            name = (key,) + new_image[1:]
            return (name, new_image)
    f = renpy.config.adjust_attributes.get(name[0], None) or renpy.config.adjust_attributes.get(None, None)
    if f is not None:
        new_image = f(name)
        name = (key,) + new_image[1:]
        return (name, new_image)
    return (name, name)

def predict_show(name, layer=None, what=None, tag=None, at_list=[]):
    if False:
        return 10
    '\n    :undocumented:\n\n    Predicts a scene or show statement.\n\n    `name`\n        The name of the image to show, a string.\n\n    `layer`\n        The layer the image is being shown on.\n\n    `what`\n        What is being show - if given, overrides `name`.\n\n    `tag`\n        The tag of the thing being shown.\n\n    `at_list`\n        A list of transforms to apply to the displayable.\n    '
    key = tag or name[0]
    layer = default_layer(layer, key)
    if isinstance(what, renpy.display.displayable.Displayable):
        base = img = what
    else:
        (name, what) = _find_image(layer, key, name, what)
        base = img = renpy.display.image.ImageReference(what, style='image_placement')
        if not base.find_target():
            return
    for i in at_list:
        if isinstance(i, renpy.display.motion.Transform):
            img = i(child=img)
        else:
            img = i(img)
        img._unique()
    renpy.game.context().images.predict_show(layer, name, True)
    renpy.display.predict.displayable(img)

def set_tag_attributes(name, layer=None):
    if False:
        print('Hello World!')
    '\n    :doc: side\n\n    This sets the attributes associated with an image tag when that image\n    tag is not showing. The main use of this would be to directly set the\n    attributes used by a side image.\n\n    For example::\n\n        $ renpy.set_tag_attributes("lucy mad")\n        $ renpy.say(l, "I\'m rather cross.")\n\n    and::\n\n        l mad "I\'m rather cross."\n\n    are equivalent.\n    '
    if not isinstance(name, tuple):
        name = tuple(name.split())
    tag = name[0]
    name = renpy.game.context().images.apply_attributes(layer, tag, name)
    if name is not None:
        renpy.game.context().images.predict_show(layer, name, False)

def show(name, at_list=[], layer=None, what=None, zorder=None, tag=None, behind=[], atl=None, transient=False, munge_name=True):
    if False:
        print('Hello World!')
    '\n    :doc: se_images\n    :args: (name, at_list=[], layer=None, what=None, zorder=0, tag=None, behind=[], **kwargs)\n\n    Shows an image on a layer. This is the programmatic equivalent of the show\n    statement.\n\n    `name`\n        The name of the image to show, a string.\n\n    `at_list`\n        A list of transforms that are applied to the image.\n        The equivalent of the ``at`` property.\n\n    `layer`\n        A string, giving the name of the layer on which the image will be shown.\n        The equivalent of the ``onlayer`` property. If None, uses the default\n        layer associated with the tag.\n\n    `what`\n        If not None, this is a displayable that will be shown in lieu of\n        looking on the image. (This is the equivalent of the show expression\n        statement.) When a `what` parameter is given, `name` can be used to\n        associate a tag with the image.\n\n    `zorder`\n        An integer, the equivalent of the ``zorder`` property. If None, the\n        zorder is preserved if it exists, and is otherwise set to 0.\n\n    `tag`\n        A string, used to specify the image tag of the shown image. The\n        equivalent of the ``as`` property.\n\n    `behind`\n        A list of strings, giving image tags that this image is shown behind.\n        The equivalent of the ``behind`` property.\n\n    ::\n\n        show a\n        $ renpy.show("a")\n\n        show expression w\n        # anonymous show expression : no equivalent\n\n        show expression w as a\n        $ renpy.show("a", what=w)\n        $ renpy.show("y", what=w, tag="a") # in this case, name is ignored\n\n        show a at T, T2\n        $ renpy.show("a", at_list=(T, T2))\n\n        show a onlayer b behind c zorder d as e\n        $ renpy.show("a", layer="b", behind=["c"], zorder="d", tag="e")\n    '
    default_transform = renpy.config.default_transform
    if renpy.game.context().init_phase:
        raise Exception('Show may not run while in init phase.')
    if not isinstance(name, tuple):
        name = tuple(name.split())
    if zorder is None and (not renpy.config.preserve_zorder):
        zorder = 0
    sls = scene_lists()
    key = tag or name[0]
    layer = default_layer(layer, key)
    if renpy.config.sticky_positions:
        if not at_list and key in sls.at_list[layer]:
            at_list = sls.at_list[layer][key]
    if not at_list:
        tt = renpy.config.tag_transform.get(key, None)
        if tt is not None:
            at_list = renpy.easy.to_list(tt, copy=True)
    if isinstance(what, renpy.display.displayable.Displayable):
        if renpy.config.wrap_shown_transforms and isinstance(what, renpy.display.motion.Transform):
            base = img = renpy.display.image.ImageReference(what, style='image_placement')
            default_transform = None
        else:
            base = img = what
    else:
        (name, what) = _find_image(layer, key, name, what)
        base = img = renpy.display.image.ImageReference(what, style='image_placement')
        if not base.find_target() and renpy.config.missing_show:
            result = renpy.config.missing_show(name, what, layer)
            if isinstance(result, renpy.display.displayable.Displayable):
                base = img = result
            elif result:
                return
    for i in at_list:
        if isinstance(i, renpy.display.motion.Transform):
            img = i(child=img)
        else:
            img = i(img)
        img._unique()
    renpy.game.persistent._seen_images[tuple((str(i) for i in name))] = True
    if tag and munge_name:
        name = (tag,) + name[1:]
    if renpy.config.missing_hide:
        renpy.config.missing_hide(name, layer)
    sls.add(layer, img, key, zorder, behind, at_list=at_list, name=name, atl=atl, default_transform=default_transform, transient=transient)

def hide(name, layer=None):
    if False:
        while True:
            i = 10
    '\n    :doc: se_images\n\n    Hides an image from a layer. The Python equivalent of the hide statement.\n\n    `name`\n        The name of the image to hide. Only the image tag is used, and\n        any image with the tag is hidden (the precise name does not matter).\n\n    `layer`\n        The layer on which this function operates. If None, uses the default\n        layer associated with the tag.\n    '
    if renpy.game.context().init_phase:
        raise Exception('Hide may not run while in init phase.')
    if not isinstance(name, tuple):
        name = tuple(name.split())
    sls = scene_lists()
    key = name[0]
    layer = default_layer(layer, key)
    sls.remove(layer, key)
    if renpy.config.missing_hide:
        renpy.config.missing_hide(name, layer)

def scene(layer='master'):
    if False:
        while True:
            i = 10
    '\n    :doc: se_images\n\n    Removes all displayables from `layer`. This is equivalent to the scene\n    statement, when the scene statement is not given an image to show.\n\n    A full scene statement is equivalent to a call to renpy.scene followed by a\n    call to :func:`renpy.show`. For example::\n\n        scene bg beach\n\n    is equivalent to::\n\n        $ renpy.scene()\n        $ renpy.show("bg beach")\n    '
    if layer is None:
        layer = 'master'
    if renpy.game.context().init_phase:
        raise Exception('Scene may not run while in init phase.')
    sls = scene_lists()
    sls.clear(layer)
    if renpy.config.missing_scene:
        renpy.config.missing_scene(layer)
    renpy.display.interface.ongoing_transition.pop(layer, None)
    for i in renpy.config.scene_callbacks:
        i(layer)

def web_input(prompt, default='', allow=None, exclude='{}', length=None, mask=False):
    if False:
        print('Hello World!')
    '\n    :undocumented:\n\n    This provides input in the web environment, when config.web_input is True.\n    '
    renpy.exports.mode('input')
    renpy.game.preferences.fullscreen = False
    prompt = renpy.text.extras.filter_text_tags(prompt, allow=set())
    roll_forward = renpy.exports.roll_forward_info()
    if not isinstance(roll_forward, basestring):
        roll_forward = None
    if roll_forward is not None:
        default = roll_forward
    wi = renpy.display.behavior.WebInput(substitute(prompt), default, length=length, allow=allow, exclude=exclude, mask=mask)
    renpy.ui.add(wi)
    renpy.exports.shown_window()
    if renpy.config.autosave_on_input and (not renpy.game.after_rollback):
        renpy.loadsave.force_autosave(True)
    rv = renpy.ui.interact(mouse='prompt', type='input', roll_forward=roll_forward)
    renpy.exports.checkpoint(rv)
    with_none = renpy.config.implicit_with_none
    if with_none:
        renpy.game.interface.do_with(None, None)
    return rv

def input(prompt, default='', allow=None, exclude='{}', length=None, with_none=None, pixel_width=None, screen='input', mask=None, copypaste=True, multiline=False, **kwargs):
    if False:
        print('Hello World!')
    '\n    :doc: input\n\n    Calling this function pops up a window asking the player to enter some\n    text. It returns the entered text.\n\n    `prompt`\n        A string giving a prompt to display to the player.\n\n    `default`\n        A string giving the initial text that will be edited by the player.\n\n    `allow`\n        If not None, a string giving a list of characters that will\n        be allowed in the text.\n\n    `exclude`\n        If not None, if a character is present in this string, it is not\n        allowed in the text.\n\n    `length`\n        If not None, this must be an integer giving the maximum length\n        of the input string.\n\n    `pixel_width`\n        If not None, the input is limited to being this many pixels wide,\n        in the font used by the input to display text.\n\n    `screen`\n        The name of the screen that takes input. If not given, the ``input``\n        screen is used.\n\n    `mask`\n        If not None, a single-character string that replaces the input text that\n        is shown to the player, such as to conceal a password.\n\n    `copypaste`\n        When true, copying from and pasting to this input is allowed.\n\n    `multiline`\n        When true, move caret to next line is allowed.\n\n    If :var:`config.disable_input` is True, this function only returns\n    `default`.\n\n    Keywords prefixed with ``show_`` have the prefix stripped and\n    are passed to the screen.\n\n    Due to limitations in supporting libraries, on Android and the web platform\n    this function is limited to alphabetic characters.\n    '
    if renpy.config.disable_input:
        return default
    fixed = in_fixed_rollback()
    if not PY2 and renpy.emscripten and renpy.config.web_input and (not fixed):
        return web_input(prompt, default, allow, exclude, length, bool(mask))
    renpy.exports.mode('input')
    roll_forward = renpy.exports.roll_forward_info()
    if not isinstance(roll_forward, basestring):
        roll_forward = None
    if roll_forward is not None:
        default = roll_forward
    (show_properties, kwargs) = renpy.easy.split_properties(kwargs, 'show_', '')
    if kwargs:
        raise TypeError('renpy.input() got unexpected keyword argument(s): {}'.format(', '.join(kwargs.keys())))
    if has_screen(screen):
        widget_properties = {}
        widget_properties['input'] = dict(default=default, length=length, allow=allow, exclude=exclude, editable=not fixed, pixel_width=pixel_width, mask=mask, copypaste=copypaste, multiline=multiline)
        show_screen(screen, _transient=True, _widget_properties=widget_properties, prompt=prompt, **show_properties)
    else:
        if screen != 'input':
            raise Exception("The '{}' screen does not exist.".format(screen))
        renpy.ui.window(style='input_window')
        renpy.ui.vbox()
        renpy.ui.text(prompt, style='input_prompt')
        inputwidget = renpy.ui.input(default, length=length, style='input_text', allow=allow, exclude=exclude)
        if fixed:
            inputwidget.disable()
        renpy.ui.close()
    renpy.exports.shown_window()
    if renpy.config.autosave_on_input and (not renpy.game.after_rollback):
        renpy.loadsave.force_autosave(True)
    if fixed:
        renpy.ui.saybehavior()
    rv = renpy.ui.interact(mouse='prompt', type='input', roll_forward=roll_forward)
    renpy.exports.checkpoint(rv)
    if with_none is None:
        with_none = renpy.config.implicit_with_none
    if with_none:
        renpy.game.interface.do_with(None, None)
    return rv
menu_args = None
menu_kwargs = None

def get_menu_args():
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    Returns a tuple giving the arguments (as a tuple) and the keyword arguments\n    (as a dict) passed to the current menu statement.\n    '
    if menu_args is None:
        return ((), {})
    return (menu_args, menu_kwargs)

def menu(items, set_expr, args=None, kwargs=None, item_arguments=None):
    if False:
        print('Hello World!')
    '\n    :undocumented:\n\n    Displays a menu, and returns to the user the value of the selected\n    choice. Also handles conditions and the menuset.\n    '
    global menu_args
    global menu_kwargs
    args = args or ()
    kwargs = kwargs or {}
    nvl = kwargs.pop('nvl', False)
    if renpy.config.menu_arguments_callback is not None:
        (args, kwargs) = renpy.config.menu_arguments_callback(*args, **kwargs)
    if renpy.config.old_substitutions:

        def substitute(s):
            if False:
                return 10
            return s % tag_quoting_dict
    else:

        def substitute(s):
            if False:
                for i in range(10):
                    print('nop')
            return s
    if item_arguments is None:
        item_arguments = [((), {})] * len(items)
    if set_expr:
        set = renpy.python.py_eval(set_expr)
        new_items = []
        new_item_arguments = []
        for (i, ia) in zip(items, item_arguments):
            if i[0] not in set:
                new_items.append(i)
                new_item_arguments.append(ia)
        items = new_items
        item_arguments = new_item_arguments
    else:
        set = None
    if renpy.config.menu_actions:
        location = renpy.game.context().current
        new_items = []
        for ((label, condition, value), (item_args, item_kwargs)) in zip(items, item_arguments):
            label = substitute(label)
            condition = renpy.python.py_eval(condition)
            if not renpy.config.menu_include_disabled and (not condition):
                continue
            if value is not None:
                new_items.append((label, renpy.ui.ChoiceReturn(label, value, location, sensitive=condition, args=item_args, kwargs=item_kwargs)))
            else:
                new_items.append((label, None))
    else:
        new_items = [(substitute(label), value) for (label, condition, value) in items if renpy.python.py_eval(condition)]
    choices = [value for (label, value) in new_items if value is not None]
    if not choices:
        return None
    old_menu_args = menu_args
    old_menu_kwargs = menu_kwargs
    try:
        menu_args = args
        menu_kwargs = kwargs
        if nvl:
            rv = renpy.store.nvl_menu(new_items)
        else:
            rv = renpy.store.menu(new_items)
    finally:
        menu_args = old_menu_args
        menu_kwargs = old_menu_kwargs
    if set is not None and rv is not None:
        for (label, condition, value) in items:
            if value == rv:
                try:
                    set.append(label)
                except AttributeError:
                    set.add(label)
    return rv

def choice_for_skipping():
    if False:
        while True:
            i = 10
    "\n    :doc: other\n\n    Tells Ren'Py that a choice is coming up soon. This currently has\n    two effects:\n\n    * If Ren'Py is skipping, and the Skip After Choices preferences is set\n      to stop skipping, skipping is terminated.\n\n    * An auto-save is triggered.\n    "
    if renpy.config.skipping and (not renpy.game.preferences.skip_after_choices):
        renpy.config.skipping = None
    if renpy.config.autosave_on_choice and (not renpy.game.after_rollback):
        renpy.loadsave.force_autosave(True)

def predict_menu():
    if False:
        while True:
            i = 10
    '\n    :undocumented:\n\n    Predicts widgets that are used by the menu.\n    '
    if not renpy.config.choice_screen_chosen:
        return
    items = [('Menu Prediction', True, False)]
    predict_screen('choice', items=items)

class MenuEntry(tuple):
    """
    The object passed into the choice screen.
    """

def display_menu(items, window_style='menu_window', interact=True, with_none=None, caption_style='menu_caption', choice_style='menu_choice', choice_chosen_style='menu_choice_chosen', choice_button_style='menu_choice_button', choice_chosen_button_style='menu_choice_chosen_button', scope={}, widget_properties=None, screen='choice', type='menu', predict_only=False, **kwargs):
    if False:
        return 10
    '\n    :doc: se_menu\n    :name: renpy.display_menu\n    :args: (items, *, interact=True, screen="choice", **kwargs)\n\n    This displays a menu to the user. `items` should be a list of 2-item tuples.\n    In each tuple, the first item is a textual label, and the second item is\n    the value to be returned if that item is selected. If the value is None,\n    the first item is used as a menu caption.\n\n    This function takes many arguments, of which only a few are documented.\n    Except for `items`, all arguments should be given as keyword arguments.\n\n    `interact`\n        If false, the menu is displayed, but no interaction is performed.\n\n    `screen`\n        The name of the screen used to display the menu.\n\n    Note that most Ren\'Py games do not use menu captions, but use narration\n    instead. To display a menu using narration, write::\n\n        $ narrator("Which direction would you like to go?", interact=False)\n        $ result = renpy.display_menu([ ("East", "east"), ("West", "west") ])\n    '
    (menu_args, menu_kwargs) = get_menu_args()
    screen = menu_kwargs.pop('screen', screen)
    with_none = menu_kwargs.pop('_with_none', with_none)
    mode = menu_kwargs.pop('_mode', type)
    if interact:
        renpy.exports.mode(mode)
        choice_for_skipping()
        if not predict_only:
            if renpy.config.choice_empty_window and (not renpy.game.context().scene_lists.shown_window):
                renpy.config.choice_empty_window('', interact=False)
    choices = []
    for (_, val) in items:
        if isinstance(val, renpy.ui.ChoiceReturn):
            val = val.value
        if val is None:
            continue
        choices.append(val)
    roll_forward = renpy.exports.roll_forward_info()
    if roll_forward not in choices:
        roll_forward = None
    if renpy.config.auto_choice_delay:
        renpy.ui.pausebehavior(renpy.config.auto_choice_delay, random.choice(choices))
    location = renpy.game.context().current
    if in_fixed_rollback() and renpy.config.fix_rollback_without_choice:
        renpy.ui.saybehavior()
    scope = dict(scope)
    scope.update(menu_kwargs)
    if has_screen(screen):
        item_actions = []
        if widget_properties is None:
            props = {}
        else:
            props = widget_properties
        for (label, value) in items:
            if not label:
                value = None
            if isinstance(value, renpy.ui.ChoiceReturn):
                action = value
                chosen = action.get_chosen()
                item_args = action.args
                item_kwargs = action.kwargs
            elif value is not None:
                action = renpy.ui.ChoiceReturn(label, value, location)
                chosen = action.get_chosen()
                item_args = ()
                item_kwargs = {}
            else:
                action = None
                chosen = False
                item_args = ()
                item_kwargs = {}
            if renpy.config.choice_screen_chosen:
                me = MenuEntry((label, action, chosen))
            else:
                me = MenuEntry((label, action))
            me.caption = label
            me.action = action
            me.chosen = chosen
            me.args = item_args
            me.kwargs = item_kwargs
            item_actions.append(me)
        show_screen(screen, *menu_args, items=item_actions, _widget_properties=props, _transient=True, _layer=renpy.config.choice_layer, **scope)
    else:
        renpy.exports.shown_window()
        renpy.ui.window(style=window_style, focus='menu')
        renpy.ui.menu(items, location=renpy.game.context().current, focus='choices', default=True, caption_style=caption_style, choice_style=choice_style, choice_chosen_style=choice_chosen_style, choice_button_style=choice_button_style, choice_chosen_button_style=choice_chosen_button_style, **kwargs)
    if renpy.config.menu_showed_window:
        renpy.exports.shown_window()
    for (label, val) in items:
        if val is not None:
            log('Choice: ' + label)
        else:
            log(label)
    log('')
    if interact:
        rv = renpy.ui.interact(mouse='menu', type=type, roll_forward=roll_forward)
        for (label, val) in items:
            if isinstance(val, renpy.ui.ChoiceReturn):
                val = val.value
            if rv == val:
                log('Player chose: ' + label)
                break
        else:
            log('No choice chosen.')
        log('')
        checkpoint(rv)
        if with_none is None:
            with_none = renpy.config.implicit_with_none
        if with_none:
            renpy.game.interface.do_with(None, None)
        return rv
    return None

class TagQuotingDict(object):

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        store = renpy.store.__dict__
        if key in store:
            rv = store[key]
            if isinstance(rv, basestring):
                rv = rv.replace('{', '{{')
            return rv
        else:
            if renpy.config.debug:
                raise Exception("During an interpolation, '%s' was not found as a variable." % key)
            return '<' + key + ' unbound>'
tag_quoting_dict = TagQuotingDict()

def predict_say(who, what):
    if False:
        return 10
    '\n    :undocumented:\n\n    This is called to predict the results of a say command.\n    '
    if who is None:
        who = renpy.store.narrator
    if isinstance(who, basestring):
        return renpy.store.predict_say(who, what)
    predict = getattr(who, 'predict', None)
    if predict:
        predict(what)

def scry_say(who, what, scry):
    if False:
        for i in range(10):
            print('nop')
    '\n    :undocumented:\n\n    Called when scry is called on a say statement. Needs to set\n    the interacts field.\n    '
    try:
        scry.interacts = who.will_interact()
    except Exception:
        scry.interacts = True
    try:
        scry.extend_text = who.get_extend_text(what)
    except Exception:
        scry.extend_text = renpy.ast.DoesNotExtend

def say(who, what, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    :doc: se_say\n\n    The equivalent of the say statement.\n\n    `who`\n        Either the character that will say something, None for the narrator,\n        or a string giving the character name. In the latter case, the\n        :var:`say` store function is called.\n\n    `what`\n        A string giving the line to say. Percent-substitutions are performed\n        in this string.\n\n    `interact`\n        If true, Ren\'Py waits for player input when displaying the dialogue. If\n        false, Ren\'Py shows the dialogue, but does not perform an interaction.\n        (This is passed in as a keyword argument.)\n\n    This function is rarely necessary, as the following three lines are\n    equivalent. ::\n\n        e "Hello, world."\n        $ renpy.say(e, "Hello, world.")\n        $ e("Hello, world.") # when e is not a string\n        $ say(e, "Hello, world.") # when e is a string\n    '
    if renpy.config.old_substitutions:
        what = what % tag_quoting_dict
    if who is None:
        who = renpy.store.narrator
    if renpy.config.say_arguments_callback:
        (args, kwargs) = renpy.config.say_arguments_callback(who, *args, **kwargs)
    if isinstance(who, basestring):
        renpy.store.say(who, what, *args, **kwargs)
    else:
        who(what, *args, **kwargs)

def imagemap(ground, selected, hotspots, unselected=None, overlays=False, style='imagemap', mouse='imagemap', with_none=None, **properties):
    if False:
        while True:
            i = 10
    "\n    :undocumented: Use screens already.\n\n    Displays an imagemap. An image map consists of two images and a\n    list of hotspots that are defined on that image. When the user\n    clicks on a hotspot, the value associated with that hotspot is\n    returned.\n\n    @param ground: The name of the file containing the ground\n    image. The ground image is displayed for areas that are not part\n    of any hotspots.\n\n    @param selected: The name of the file containing the selected\n    image. This image is displayed in hotspots when the mouse is over\n    them.\n\n    @param hotspots: A list of tuples defining the hotspots in this\n    image map. Each tuple has the format (x0, y0, x1, y1, result).\n    (x0, y0) gives the coordinates of the upper-left corner of the\n    hotspot, (x1, y1) gives the lower-right corner, and result gives\n    the value returned from this function if the mouse is clicked in\n    the hotspot.\n\n    @param unselected: If provided, then it is the name of a file\n    containing the image that's used to fill in hotspots that are not\n    selected as part of any image. If not provided, the ground image\n    is used instead.\n\n    @param overlays: If True, overlays are displayed when this imagemap\n    is active. If False, the overlays are suppressed.\n\n    @param with_none: If True, performs a with None after the input. If None,\n    takes the value from config.implicit_with_none.\n    "
    renpy.exports.mode('imagemap')
    renpy.ui.imagemap_compat(ground, selected, hotspots, unselected=unselected, style=style, **properties)
    roll_forward = renpy.exports.roll_forward_info()
    if roll_forward not in [result for (_x0, _y0, _x1, _y1, result) in hotspots]:
        roll_forward = None
    if in_fixed_rollback() and renpy.config.fix_rollback_without_choice:
        renpy.ui.saybehavior()
    rv = renpy.ui.interact(suppress_overlay=not overlays, type='imagemap', mouse=mouse, roll_forward=roll_forward)
    renpy.exports.checkpoint(rv)
    if with_none is None:
        with_none = renpy.config.implicit_with_none
    if with_none:
        renpy.game.interface.do_with(None, None)
    return rv

def pause(delay=None, music=None, with_none=None, hard=False, predict=False, checkpoint=None, modal=None):
    if False:
        print('Hello World!')
    "\n    :doc: se_pause\n    :args: (delay=None, *, hard=False, predict=False, modal=None)\n\n    Causes Ren'Py to pause. Returns true if the user clicked to end the pause,\n    or false if the pause timed out or was skipped.\n\n    `delay`\n        If given, the number of seconds Ren'Py should pause for.\n\n    The following should be given as keyword arguments:\n\n    `hard`\n        This must be given as a keyword argument. When True, Ren'Py may prevent\n        the user from clicking to interrupt the pause. If the player enables\n        skipping, the hard pause will be skipped. There may be other circumstances\n        where the hard pause ends early or prevents Ren'Py from operating properly,\n        these will not be treated as bugs.\n\n        In general, using hard pauses is rude. When the user clicks to advance\n        the game, it's an explicit request - the user wishes the game to advance.\n        To override that request is to assume you understand what the player\n        wants more than the player does.\n\n        Calling renpy.pause guarantees that whatever is on the screen will be\n        displayed for at least one frame, and hence has been shown to the\n        player.\n\n        tl;dr - Don't use renpy.pause with hard=True.\n\n    `predict`\n        If True, Ren'Py will end the pause when all prediction, including\n        prediction scheduled with :func:`renpy.start_predict` and\n        :func:`renpy.start_predict_screen`, has been finished.\n\n        This also causes Ren'Py to prioritize prediction over display smoothness\n        for the duration of the pause. Because of that, it's recommended to not\n        display animations during prediction.\n\n    `modal`\n        If True or None, the pause will not end when a modal screen is being displayed.\n        If False, the pause will end while a modal screen is being displayed.\n    "
    if renpy.config.skipping == 'fast':
        return False
    if checkpoint is None:
        if delay is not None:
            checkpoint = False
        else:
            checkpoint = True
    roll_forward = renpy.exports.roll_forward_info()
    if type(roll_forward) not in (bool, renpy.game.CallException, renpy.game.JumpException):
        roll_forward = None
    if delay is not None and renpy.game.after_rollback and (not renpy.config.pause_after_rollback):
        rv = roll_forward
        if rv is None:
            rv = False
        if checkpoint:
            renpy.exports.checkpoint(rv, keep_rollback=True, hard=False)
        return rv
    renpy.exports.mode('pause')
    if music is not None:
        newdelay = renpy.audio.music.get_delay(music)
        if newdelay is not None:
            delay = newdelay
    if delay is not None and renpy.game.after_rollback and (roll_forward is None):
        delay = 0
    if delay is None:
        afm = ' '
    else:
        afm = None
    if hard or not renpy.store._dismiss_pause:
        renpy.ui.saybehavior(afm=afm, dismiss='dismiss_hard_pause', dismiss_unfocused=[])
    else:
        renpy.ui.saybehavior(afm=afm)
    if predict:
        renpy.display.interface.force_prediction = True
        renpy.ui.add(renpy.display.behavior.PredictPauseBehavior())
    try:
        rv = renpy.ui.interact(mouse='pause', type='pause', roll_forward=roll_forward, pause=delay, pause_modal=modal)
    except (renpy.game.JumpException, renpy.game.CallException) as e:
        rv = e
    if checkpoint:
        renpy.exports.checkpoint(rv, keep_rollback=True, hard=renpy.config.pause_after_rollback or delay is None)
    if with_none is None:
        with_none = renpy.config.implicit_with_none
    if with_none:
        renpy.game.interface.do_with(None, None)
    if isinstance(rv, (renpy.game.JumpException, renpy.game.CallException)):
        raise rv
    return rv

def movie_cutscene(filename, delay=None, loops=0, stop_music=True):
    if False:
        print('Hello World!')
    "\n    :doc: movie_cutscene\n\n    This displays a movie cutscene for the specified number of\n    seconds. The user can click to interrupt the cutscene.\n    Overlays and Underlays are disabled for the duration of the cutscene.\n\n    `filename`\n        The name of a file containing any movie playable by Ren'Py.\n\n    `delay`\n        The number of seconds to wait before ending the cutscene.\n        Normally the length of the movie, in seconds. If None, then the\n        delay is computed from the number of loops (that is, loops + 1) *\n        the length of the movie. If -1, we wait until the user clicks.\n\n    `loops`\n        The number of extra loops to show, -1 to loop forever.\n\n    Returns True if the movie was terminated by the user, or False if the\n    given delay elapsed uninterrupted.\n    "
    renpy.exports.mode('movie')
    if stop_music:
        renpy.audio.audio.set_force_stop('music', True)
    movie_start_fullscreen(filename, loops=loops)
    renpy.ui.saybehavior()
    if delay is None or delay < 0:
        renpy.ui.soundstopbehavior('movie')
    else:
        renpy.ui.pausebehavior(delay, False)
    if renpy.game.log.forward:
        roll_forward = True
    else:
        roll_forward = None
    rv = renpy.ui.interact(suppress_overlay=True, roll_forward=roll_forward)
    movie_stop()
    if stop_music:
        renpy.audio.audio.set_force_stop('music', False)
    return rv

def with_statement(trans, always=False, paired=None, clear=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: se_with\n    :name: renpy.with_statement\n    :args: (trans, always=False)\n\n    Causes a transition to occur. This is the Python equivalent of the\n    with statement.\n\n    `trans`\n        The transition.\n\n    `always`\n        If True, the transition will always occur, even if the user has\n        disabled transitions.\n\n    This function returns true if the user chose to interrupt the transition,\n    and false otherwise.\n    '
    if renpy.game.context().init_phase:
        raise Exception('With statements may not run while in init phase.')
    if renpy.config.skipping:
        trans = None
    if not (renpy.game.preferences.transitions or always):
        trans = None
    renpy.exports.mode('with')
    if isinstance(trans, dict):
        for (k, v) in trans.items():
            if k is None:
                continue
            renpy.exports.transition(v, layer=k)
        if None not in trans:
            return
        trans = trans[None]
    return renpy.game.interface.do_with(trans, paired, clear=clear)
globals()['with'] = with_statement

def rollback(force=False, checkpoints=1, defer=False, greedy=True, label=None, abnormal=True, current_label=None):
    if False:
        print('Hello World!')
    "\n    :doc: rollback\n    :args: (force=False, checkpoints=1, defer=False, greedy=True, label=None, abnormal=True)\n\n    Rolls the state of the game back to the last checkpoint.\n\n    `force`\n        If true, the rollback will occur in all circumstances. Otherwise,\n        the rollback will only occur if rollback is enabled in the store,\n        context, and config.\n\n    `checkpoints`\n        Ren'Py will roll back through this many calls to renpy.checkpoint. It\n        will roll back as far as it can, subject to this condition.\n\n    `defer`\n        If true, the call will be deferred until control returns to the main\n        context.\n\n    `greedy`\n        If true, rollback will finish just after the previous checkpoint.\n        If false, rollback finish just before the current checkpoint.\n\n    `label`\n        If not None, a label that is called when rollback completes.\n\n    `abnormal`\n        If true, the default, script executed after the transition is run in\n        an abnormal mode that skips transitions that would have otherwise\n        occured. Abnormal mode ends when an interaction begins.\n    "
    if defer and (not renpy.game.log.log):
        return
    if defer and len(renpy.game.contexts) > 1:
        renpy.game.contexts[0].defer_rollback = (force, checkpoints)
        return
    if not force:
        if not renpy.store._rollback:
            return
        if not renpy.game.context().rollback:
            return
        if not renpy.config.rollback_enabled:
            return
    renpy.config.skipping = None
    renpy.game.log.complete()
    renpy.game.log.rollback(checkpoints, greedy=greedy, label=label, force=force is True, abnormal=abnormal, current_label=current_label)

def toggle_fullscreen():
    if False:
        i = 10
        return i + 15
    '\n    :undocumented:\n    Toggles the fullscreen mode.\n    '
    renpy.game.preferences.fullscreen = not renpy.game.preferences.fullscreen

def toggle_music():
    if False:
        for i in range(10):
            print('nop')
    '\n    :undocumented:\n    Does nothing.\n    '

@renpy_pure
def has_label(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: label\n\n    Returns true if `name` is a valid label in the program, or false\n    otherwise.\n\n    `name`\n        Should be a string to check for the existence of a label. It can\n        also be an opaque tuple giving the name of a non-label statement.\n    '
    return renpy.game.script.has_label(name)

@renpy_pure
def get_all_labels():
    if False:
        while True:
            i = 10
    '\n    :doc: label\n\n    Returns the set of all labels defined in the program, including labels\n    defined for internal use in the libraries.\n    '
    rv = []
    for i in renpy.game.script.namemap:
        if isinstance(i, basestring):
            rv.append(i)
    return renpy.revertable.RevertableSet(rv)

def take_screenshot(scale=None, background=False):
    if False:
        i = 10
        return i + 15
    '\n    :doc: loadsave\n    :args: ()\n\n    Causes a screenshot to be taken. This screenshot will be saved as part of\n    a saved game.\n    '
    if scale is None:
        scale = (renpy.config.thumbnail_width, renpy.config.thumbnail_height)
    renpy.game.interface.take_screenshot(scale, background=background)

def full_restart(transition=False, label='_invoke_main_menu', target='_main_menu', save=False):
    if False:
        while True:
            i = 10
    "\n    :doc: other\n    :args: (transition=False, *, save=False)\n\n    Causes Ren'Py to restart, returning the user to the main menu.\n\n    `transition`\n        If given, the transition to run, or None to not run a transition.\n        False uses :var:`config.end_game_transition`.\n\n    `save`\n        If true, the game is saved in :var:`_quit_slot` before Ren'Py\n        restarts and returns the user to the main menu.\n    "
    if save and renpy.store._quit_slot is not None:
        renpy.loadsave.save(renpy.store._quit_slot, getattr(renpy.store, 'save_name', ''))
    if transition is False:
        transition = renpy.config.end_game_transition
    raise renpy.game.FullRestartException((transition, label, target))

def utter_restart(keep_renderer=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    :undocumented: Used in the implementation of shift+R.\n\n    Causes an utter restart of Ren'Py. This reloads the script and\n    re-runs initialization.\n    "
    renpy.session['_keep_renderer'] = keep_renderer
    raise renpy.game.UtterRestartException()

def reload_script():
    if False:
        while True:
            i = 10
    "\n    :doc: reload\n\n    Causes Ren'Py to save the game, reload the script, and then load the\n    save.\n\n    This should only be called during development. It works on Windows, macOS,\n    and Linux, but may not work on other platforms.\n    "
    if renpy.store._in_replay:
        return
    s = get_screen('menu')
    session = renpy.session
    session['_reload'] = True
    if '_reload_screen' in session or '_main_menu_screen' in session:
        utter_restart()
    if not renpy.store.main_menu:
        if s is not None:
            session['_reload_screen'] = s.screen_name[0]
            session['_reload_screen_args'] = s.scope.get('_args', ())
            session['_reload_screen_kwargs'] = s.scope.get('_kwargs', {})
        renpy.game.call_in_new_context('_save_reload_game')
    else:
        if s is not None:
            session['_main_menu_screen'] = s.screen_name[0]
            session['_main_menu_screen_args'] = s.scope.get('_args', ())
            session['_main_menu_screen_kwargs'] = s.scope.get('_kwargs', {})
        utter_restart()

def quit(relaunch=False, status=0, save=False):
    if False:
        while True:
            i = 10
    "\n    :doc: other\n\n    This causes Ren'Py to exit entirely.\n\n    `relaunch`\n        If true, Ren'Py will run a second copy of itself before quitting.\n\n    `status`\n        The status code Ren'Py will return to the operating system.\n        Generally, 0 is success, and positive integers are failure.\n\n    `save`\n        If true, the game is saved in :var:`_quit_slot` before Ren'Py\n        terminates.\n    "
    if save and renpy.store._quit_slot is not None:
        renpy.loadsave.save(renpy.store._quit_slot, getattr(renpy.store, 'save_name', ''))
    if has_label('quit'):
        call_in_new_context('quit')
    raise renpy.game.QuitException(relaunch=relaunch, status=status)

def jump(label):
    if False:
        i = 10
        return i + 15
    '\n    :doc: se_jump\n\n    Causes the current statement to end, and control to jump to the given\n    label.\n    '
    raise renpy.game.JumpException(label)

def jump_out_of_context(label):
    if False:
        i = 10
        return i + 15
    '\n    :doc: context\n\n    Causes control to leave the current context, and then to be\n    transferred in the parent context to the given label.\n    '
    raise renpy.game.JumpOutException(label)

def call(label, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    :doc: se_call\n    :args: (label, *args, from_current=False, **kwargs)\n\n    Causes the current Ren\'Py statement to terminate, and a jump to a\n    `label` to occur. When the jump returns, control will be passed\n    to the statement following the current statement.\n\n    The label must be either of the form "global_name" or "global_name.local_name".\n    The form ".local_name" is not allowed.\n\n    `from_current`\n        If true, control will return to the current statement, rather than\n        the statement following the current statement. (This will lead to\n        the current statement being run twice. This must be passed as a\n        keyword argument.)\n    '
    from_current = kwargs.pop('from_current', False)
    raise renpy.game.CallException(label, args, kwargs, from_current=from_current)

def return_statement(value=None):
    if False:
        print('Hello World!')
    "\n    :doc: se_call\n\n    Causes Ren'Py to return from the current Ren'Py-level call.\n    "
    renpy.store._return = value
    jump('_renpy_return')

def warp_to_line(warp_spec):
    if False:
        i = 10
        return i + 15
    '\n    :doc: debug\n\n    This takes as an argument a filename:linenumber pair, and tries to warp to\n    the statement before that line number.\n\n    This works samely as the `--warp` command.\n    '
    renpy.warp.warp_spec = warp_spec
    renpy.warp.warp()

def screenshot(filename):
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    Saves a screenshot in `filename`.\n\n    Returns True if the screenshot was saved successfully, False if saving\n    failed for some reason.\n\n    The :var:`config.screenshot_pattern` and :var:`_screenshot_pattern`\n    variables control the file the screenshot is saved in.\n    '
    return renpy.game.interface.save_screenshot(filename)

def screenshot_to_bytes(size):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: other\n\n    Returns a screenshot as a bytes object, that can be passed to im.Data().\n    The bytes will be a png-format image, such that::\n\n        $ data = renpy.screenshot_to_bytes((640, 360))\n        show expression im.Data(data, "screenshot.png"):\n            align (0, 0)\n\n    Will show the image. The bytes objects returned can be stored in save\n    files and persistent data. However, these may be large, and care should\n    be taken to not include too many.\n\n    `size`\n        The size the screenshot will be resized to. If None, the screenshot\n        will be resized, and hence will be the size of the player\'s window,\n        without any letterbars.\n\n    This function may be slow, and so it\'s intended for save-like screenshots,\n    and not realtime effects.\n    '
    return renpy.game.interface.screenshot_to_bytes(size)

@renpy_pure
def version(tuple=False):
    if False:
        i = 10
        return i + 15
    '\n    :doc: renpy_version\n\n    If `tuple` is false, returns a string containing "Ren\'Py ", followed by\n    the current version of Ren\'Py.\n\n    If `tuple` is true, returns a tuple giving each component of the\n    version as an integer.\n    '
    if tuple:
        return renpy.version_tuple
    return renpy.version
version_string = renpy.version
version_only = renpy.version_only
version_name = renpy.version_name
version_tuple = renpy.version_tuple
license = ''
try:
    import platform as _platform
    platform = '-'.join(_platform.platform().split('-')[:2])
except Exception:
    if renpy.android:
        platform = 'Android'
    elif renpy.ios:
        platform = 'iOS'
    else:
        platform = 'Unknown'

def transition(trans, layer=None, always=False, force=False):
    if False:
        while True:
            i = 10
    '\n    :doc: other\n    :args: (trans, layer=None, always=False)\n\n    Sets the transition that will be used during the next interaction.\n\n    `layer`\n        The layer the transition applies to. If None, the transition\n        applies to the entire scene.\n\n    `always`\n        If false, this respects the transition preference. If true, the\n        transition is always run.\n    '
    if isinstance(trans, dict):
        for (ly, t) in trans.items():
            transition(t, layer=ly, always=always, force=force)
        return
    if not always and (not renpy.game.preferences.transitions):
        trans = None
    if renpy.config.skipping:
        trans = None
    renpy.game.interface.set_transition(trans, layer, force=force)

def get_transition(layer=None):
    if False:
        i = 10
        return i + 15
    '\n    :doc: other\n\n    Gets the transition for `layer`, or the entire scene if\n    `layer` is None. This returns the transition that is queued up\n    to run during the next interaction, or None if no such\n    transition exists.\n    '
    return renpy.game.interface.transition.get(layer, None)

def clear_game_runtime():
    if False:
        print('Hello World!')
    '\n    :doc: other\n\n    Resets the game runtime counter.\n    '
    renpy.game.contexts[0].runtime = 0

def get_game_runtime():
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: other\n\n    Returns the game runtime counter.\n\n    The game runtime counter counts the number of seconds that have\n    elapsed while waiting for user input in the top-level context.\n    (It does not count time spent in the main or game menus.)\n    '
    return renpy.game.contexts[0].runtime

@renpy_pure
def loadable(filename, directory=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: file\n\n    Returns True if the given filename is loadable, meaning that it\n    can be loaded from the disk or from inside an archive. Returns\n    False if this is not the case.\n\n    `directory`\n        If not None, a directory to search in if the file is not found\n        in the game directory. This will be prepended to filename, and\n        the search tried again.\n    '
    return renpy.loader.loadable(filename, directory=directory)

@renpy_pure
def exists(filename):
    if False:
        for i in range(10):
            print('nop')
    "\n    :doc: file_rare\n\n    Returns true if the given filename can be found in the\n    searchpath. This only works if a physical file exists on disk. It\n    won't find the file if it's inside of an archive.\n\n    You almost certainly want to use :func:`renpy.loadable` in preference\n    to this function.\n    "
    try:
        renpy.loader.transfn(filename)
        return True
    except Exception:
        return False

def restart_interaction():
    if False:
        i = 10
        return i + 15
    '\n    :doc: other\n\n    Restarts the current interaction. Among other things, this displays\n    images added to the scene, re-evaluates screens, and starts any\n    queued transitions.\n\n    This only does anything when called from within an interaction (for\n    example, from an action). Outside an interaction, this function has\n    no effect.\n    '
    try:
        renpy.game.interface.restart_interaction = True
    except Exception:
        pass

def context():
    if False:
        return 10
    '\n    :doc: context\n\n    Returns an object that is unique to the current context. The object\n    is copied when entering a new context, but changes to the copy do\n    not change the original.\n\n    The object is saved and participates in rollback.\n    '
    return renpy.game.context().info

def context_nesting_level():
    if False:
        i = 10
        return i + 15
    '\n    :doc: context\n\n    Returns the nesting level of the current context. This is 0 for the\n    outermost context (the context that is saved, loaded, and rolled-back),\n    and is non-zero in other contexts, such as menu and replay contexts.\n    '
    return len(renpy.game.contexts) - 1

def music_start(filename, loops=True, fadeout=None, fadein=0):
    if False:
        i = 10
        return i + 15
    '\n    Deprecated music start function, retained for compatibility. Use\n    renpy.music.play() or .queue() instead.\n    '
    renpy.audio.music.play(filename, loop=loops, fadeout=fadeout, fadein=fadein)

def music_stop(fadeout=None):
    if False:
        return 10
    '\n    Deprecated music stop function, retained for compatibility. Use\n    renpy.music.stop() instead.\n    '
    renpy.audio.music.stop(fadeout=fadeout)

def get_filename_line():
    if False:
        i = 10
        return i + 15
    '\n    :doc: debug\n\n    Returns a pair giving the filename and line number of the current\n    statement.\n    '
    n = renpy.game.script.namemap.get(renpy.game.context().current, None)
    if n is None:
        return ('unknown', 0)
    else:
        return (n.filename, n.linenumber)
logfile = None

def log(msg):
    if False:
        i = 10
        return i + 15
    '\n    :doc: debug\n\n    If :var:`config.log` is not set, this does nothing. Otherwise, it opens\n    the logfile (if not already open), formats the message to :var:`config.log_width`\n    columns, and prints it to the logfile.\n    '
    global logfile
    if not renpy.config.log:
        return
    if msg is None:
        return
    try:
        msg = unicode(msg)
    except Exception:
        pass
    try:
        if not logfile:
            import os
            logfile = open(os.path.join(renpy.config.basedir, renpy.config.log), 'a')
            if not logfile.tell():
                logfile.write('\ufeff')
        import textwrap
        wrapped = textwrap.fill(msg, renpy.config.log_width)
        wrapped = unicode(wrapped)
        logfile.write(wrapped + '\n')
        logfile.flush()
    except Exception:
        renpy.config.log = None

def force_full_redraw():
    if False:
        while True:
            i = 10
    '\n    :undocumented:\n\n    Forces the screen to be redrawn in full. Call this after using pygame\n    to redraw the screen directly.\n    '
    return

def do_reshow_say(who, what, interact=False, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    if who is not None:
        who = renpy.python.py_eval(who)
    say(who, what, *args, interact=interact, **kwargs)
curried_do_reshow_say = curry(do_reshow_say)

def get_reshow_say(**kwargs):
    if False:
        print('Hello World!')
    kw = dict(renpy.store._last_say_kwargs)
    kw.update(kwargs)
    return curried_do_reshow_say(renpy.store._last_say_who, renpy.store._last_say_what, renpy.store._last_say_args, **kw)

def reshow_say(**kwargs):
    if False:
        i = 10
        return i + 15
    get_reshow_say()(**kwargs)

def current_interact_type():
    if False:
        return 10
    return getattr(renpy.game.context().info, '_current_interact_type', None)

def last_interact_type():
    if False:
        print('Hello World!')
    return getattr(renpy.game.context().info, '_last_interact_type', None)

def dynamic(*variables, **kwargs):
    if False:
        return 10
    '\n    :doc: label\n\n    This can be given one or more variable names as arguments. This makes the\n    variables dynamically scoped to the current call. When the call returns, the\n    variables will be reset to the value they had when this function was called.\n\n    If the variables are given as keyword arguments, the value of the argument\n    is assigned to the variable name.\n\n    Example calls are::\n\n        $ renpy.dynamic("x", "y", "z")\n        $ renpy.dynamic(players=2, score=0)\n    '
    variables = variables + tuple(kwargs)
    renpy.game.context().make_dynamic(variables)
    for (k, v) in kwargs.items():
        setattr(renpy.store, k, v)

def context_dynamic(*variables):
    if False:
        print('Hello World!')
    '\n    :doc: context\n\n    This can be given one or more variable names as arguments. This makes\n    the variables dynamically scoped to the current context. The variables will\n    be reset to their original value when returning to the prior context.\n\n    An example call is::\n\n        $ renpy.context_dynamic("x", "y", "z")\n    '
    renpy.game.context().make_dynamic(variables, context=True)

def seen_label(label):
    if False:
        return 10
    "\n    :doc: label\n\n    Returns true if the named label has executed at least once on the current user's\n    system, and false otherwise. This can be used to unlock scene galleries, for\n    example.\n    "
    return label in renpy.game.persistent._seen_ever

def mark_label_seen(label):
    if False:
        i = 10
        return i + 15
    "\n    :doc: label\n\n    Marks the named label as if it has been already executed on the current user's\n    system.\n    "
    renpy.game.persistent._seen_ever[str(label)] = True

def mark_label_unseen(label):
    if False:
        i = 10
        return i + 15
    "\n    :doc: label\n\n    Marks the named label as if it has not been executed on the current user's\n    system yet.\n    "
    if label in renpy.game.persistent._seen_ever:
        del renpy.game.persistent._seen_ever[label]

def seen_audio(filename):
    if False:
        return 10
    "\n    :doc: audio\n\n    Returns True if the given filename has been played at least once on the current\n    user's system.\n    "
    filename = re.sub('^<.*?>', '', filename)
    return filename in renpy.game.persistent._seen_audio

def mark_audio_seen(filename):
    if False:
        print('Hello World!')
    "\n    :doc: audio\n\n    Marks the given filename as if it has been already played on the current user's\n    system.\n    "
    filename = re.sub('^<.*?>', '', filename)
    renpy.game.persistent._seen_audio[filename] = True

def mark_audio_unseen(filename):
    if False:
        print('Hello World!')
    "\n    :doc: audio\n\n    Marks the given filename as if it has not been played on the current user's\n    system yet.\n    "
    filename = re.sub('^<.*?>', '', filename)
    if filename in renpy.game.persistent._seen_audio:
        del renpy.game.persistent._seen_audio[filename]

def seen_image(name):
    if False:
        print('Hello World!')
    "\n    :doc: image_func\n\n    Returns True if the named image has been seen at least once on the user's\n    system. An image has been seen if it's been displayed using the show statement,\n    scene statement, or :func:`renpy.show` function. (Note that there are cases\n    where the user won't actually see the image, like a show immediately followed by\n    a hide.)\n    "
    if not isinstance(name, tuple):
        name = tuple(name.split())
    return name in renpy.game.persistent._seen_images

def mark_image_seen(name):
    if False:
        return 10
    "\n    :doc: image_func\n\n    Marks the named image as if it has been already displayed on the current user's\n    system.\n    "
    if not isinstance(name, tuple):
        name = tuple(name.split())
    renpy.game.persistent._seen_images[tuple((str(i) for i in name))] = True

def mark_image_unseen(name):
    if False:
        print('Hello World!')
    "\n    :doc: image_func\n\n    Marks the named image as if it has not been displayed on the current user's\n    system yet.\n    "
    if not isinstance(name, tuple):
        name = tuple(name.split())
    if name in renpy.game.persistent._seen_images:
        del renpy.game.persistent._seen_images[name]

def open_file(fn, encoding=None, directory=None):
    if False:
        print('Hello World!')
    "\n    :doc: file\n\n    Returns a read-only file-like object that accesses the file named `fn`. The file is\n    accessed using Ren'Py's standard search method, and may reside in the game directory,\n    in an RPA archive, or as an Android asset.\n\n    The object supports a wide subset of the fields and methods found on Python's\n    standard file object, opened in binary mode. (Basically, all of the methods that\n    are sensible for a read-only file.)\n\n    `encoding`\n        If given, the file is open in text mode with the given encoding.\n        If None, the default, the encoding is taken from :var:`config.open_file_encoding`.\n        If False, the file is opened in binary mode.\n\n    `directory`\n        If not None, a directory to search in if the file is not found\n        in the game directory. This will be prepended to filename, and\n        the search tried again.\n    "
    rv = renpy.loader.load(fn, directory=directory)
    if encoding is None:
        encoding = renpy.config.open_file_encoding
    if encoding:
        rv = io.TextIOWrapper(rv, encoding=encoding, errors='surrogateescape')
    return rv

def file(fn, encoding=None):
    if False:
        return 10
    "\n    :doc: file\n\n    An alias for :func:`renpy.open_file`, for compatibility with older\n    versions of Ren'Py.\n    "
    return open_file(fn, encoding=encoding)

def notl_file(fn):
    if False:
        i = 10
        return i + 15
    "\n    :undocumented:\n\n    Like file, but doesn't search the translation prefix.\n    "
    return renpy.loader.load(fn, tl=False)

def image_size(im):
    if False:
        return 10
    '\n    :doc: file_rare\n\n    Given an image manipulator, loads it and returns a (``width``,\n    ``height``) tuple giving its size.\n\n    This reads the image in from disk and decompresses it, without\n    using the image cache. This can be slow.\n    '
    renpy.loader.index_archives()
    im = renpy.easy.displayable(im)
    if not isinstance(im, renpy.display.im.Image):
        raise Exception("renpy.image_size expects it's argument to be an image.")
    surf = im.load()
    return surf.get_size()

def get_at_list(name, layer=None):
    if False:
        while True:
            i = 10
    '\n    :doc: se_images\n\n    Returns the list of transforms being applied to the image with tag `name`\n    on `layer`. Returns an empty list if no transforms are being applied, or\n    None if the image is not shown.\n\n    If `layer` is None, uses the default layer for the given tag.\n    '
    if isinstance(name, basestring):
        name = tuple(name.split())
    tag = name[0]
    layer = default_layer(layer, tag)
    transforms = renpy.game.context().scene_lists.at_list[layer].get(tag, None)
    if transforms is None:
        return None
    return list(transforms)

def show_layer_at(at_list, layer='master', reset=True, camera=False):
    if False:
        print('Hello World!')
    '\n    :doc: se_images\n    :name: renpy.show_layer_at\n\n    The Python equivalent of the ``show layer`` `layer` ``at`` `at_list`\n    statement. If `camera` is True, the equivalent of the ``camera`` statement.\n\n    `reset`\n        If true, the transform state is reset to the start when it is shown.\n        If false, the transform state is persisted, allowing the new transform\n        to update that state.\n    '
    at_list = renpy.easy.to_list(at_list)
    renpy.game.context().scene_lists.set_layer_at_list(layer, at_list, reset=reset, camera=camera)
layer_at_list = show_layer_at

def free_memory():
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    Attempts to free some memory. Useful before running a renpygame-based\n    minigame.\n    '
    force_full_redraw()
    renpy.display.interface.kill_textures()
    renpy.display.interface.kill_surfaces()
    renpy.text.font.free_memory()
    gc.collect(2)
    if gc.garbage:
        del gc.garbage[:]

def flush_cache_file(fn):
    if False:
        print('Hello World!')
    "\n    :doc: image_func\n\n    This flushes all image cache entries that refer to the file `fn`.  This\n    may be called when an image file changes on disk to force Ren'Py to\n    use the new version.\n    "
    renpy.display.im.cache.flush_file(fn)

@renpy_pure
def easy_displayable(d, none=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    :undocumented:\n    '
    if none:
        return renpy.easy.displayable(d)
    else:
        return renpy.easy.displayable_or_none(d)

def quit_event():
    if False:
        i = 10
        return i + 15
    '\n    :doc: other\n\n    Triggers a quit event, as if the player clicked the quit button in the\n    window chrome.\n    '
    renpy.game.interface.quit_event()

def iconify():
    if False:
        print('Hello World!')
    '\n    :doc: other\n\n    Iconifies the game.\n    '
    renpy.game.interface.iconify()
call_in_new_context = renpy.game.call_in_new_context
curried_call_in_new_context = curry(call_in_new_context)
invoke_in_new_context = renpy.game.invoke_in_new_context
curried_invoke_in_new_context = curry(invoke_in_new_context)
call_replay = renpy.game.call_replay
renpy_pure('curried_call_in_new_context')
renpy_pure('curried_invoke_in_new_context')

def _error(msg):
    if False:
        print('Hello World!')
    raise Exception(msg)
_error_handlers = [_error]

def push_error_handler(eh):
    if False:
        i = 10
        return i + 15
    _error_handlers.append(eh)

def pop_error_handler():
    if False:
        i = 10
        return i + 15
    _error_handlers.pop()

def error(msg):
    if False:
        return 10
    '\n    :doc: lint\n\n    Reports `msg`, a string, as as error for the user. This is logged as a\n    parse or lint error when approprate, and otherwise it is raised as an\n    exception.\n    '
    _error_handlers[-1](msg)

def timeout(seconds):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: udd_utility\n\n    Causes an event to be generated before `seconds` seconds have elapsed.\n    This ensures that the event method of a user-defined displayable will be\n    called.\n    '
    renpy.game.interface.timeout(seconds)

def end_interaction(value):
    if False:
        while True:
            i = 10
    '\n    :doc: udd_utility\n\n    If `value` is not None, immediately ends the current interaction, causing\n    the interaction to return `value`. If `value` is None, does nothing.\n\n    This can be called from inside the render and event methods of a\n    creator-defined displayable.\n    '
    if value is None:
        return
    raise renpy.display.core.EndInteraction(value)

def scry():
    if False:
        return 10
    "\n    :doc: other\n\n    Returns the scry object for the current statement. Returns None if\n    there are no statements executing.\n\n    The scry object tells Ren'Py about things that must be true in the\n    future of the current statement. Right now, the scry object has the\n    following fields:\n\n    `nvl_clear`\n        Is true if an ``nvl clear`` statement will execute before the\n        next interaction.\n\n    `say`\n        Is true if an ``say`` statement will execute before the\n        next interaction.\n\n    `menu_with_caption`\n        Is true if a ``menu`` statement with a caption will execute\n        before the next interaction.\n\n    `who`\n        If a ``say`` or ``menu-with-caption`` statement will execute\n        before the next interaction, this is the character object it will use.\n\n    The scry object has a next() method, which returns the scry object of\n    the statement after the current one, if only one statement will execute\n    after the this one. Otherwise, it returns None.\n\n    .. warning::\n\n        Like other similar functions, the object this returns is meant to be used\n        in the short term after the function is called. Including it in save data\n        or making it participate in rollback is not advised.\n    "
    name = renpy.game.context().current
    if name is None:
        return None
    node = renpy.game.script.lookup(name)
    return node.scry()

@renpy_pure
def munged_filename():
    if False:
        return 10
    return renpy.lexer.munge_filename(get_filename_line()[0])
loaded_modules = set()

def load_module(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    :doc: other\n    :args: (name)\n\n    This loads the Ren'Py module named name. A Ren'Py module consists of Ren'Py script\n    that is loaded into the usual (store) namespace, contained in a file named\n    name.rpym or name.rpymc. If a .rpym file exists, and is newer than the\n    corresponding .rpymc file, it is loaded and a new .rpymc file is created.\n\n    All of the init blocks (and other init-phase code) in the module are run\n    before this function returns. An error is raised if the module name cannot\n    be found, or is ambiguous.\n\n    Module loading may only occur from inside an init block.\n    "
    if not renpy.game.context().init_phase:
        raise Exception('Module loading is only allowed in init code.')
    if name in loaded_modules:
        return
    loaded_modules.add(name)
    old_locked = renpy.config.locked
    renpy.config.locked = False
    initcode = renpy.game.script.load_module(name)
    context = renpy.execution.Context(False)
    context.init_phase = True
    renpy.game.contexts.append(context)
    context.make_dynamic(kwargs)
    renpy.store.__dict__.update(kwargs)
    for (_prio, node) in initcode:
        if isinstance(node, renpy.ast.Node):
            renpy.game.context().run(node)
        else:
            node()
    context.pop_all_dynamic()
    renpy.game.contexts.pop()
    renpy.config.locked = old_locked

def load_string(s, filename='<string>'):
    if False:
        for i in range(10):
            print('nop')
    "\n    :doc: other\n\n    Loads `s` as Ren'Py script that can be called.\n\n    Returns the name of the first statement in s.\n\n    `filename` is the name of the filename that statements in the string will\n    appear to be from.\n    "
    old_exception_info = renpy.game.exception_info
    try:
        old_locked = renpy.config.locked
        renpy.config.locked = False
        (stmts, initcode) = renpy.game.script.load_string(filename, str(s))
        if stmts is None:
            return None
        context = renpy.execution.Context(False)
        context.init_phase = True
        renpy.game.contexts.append(context)
        for (_prio, node) in initcode:
            if isinstance(node, renpy.ast.Node):
                renpy.game.context().run(node)
            else:
                node()
        context.pop_all_dynamic()
        renpy.game.contexts.pop()
        renpy.config.locked = old_locked
        renpy.game.script.analyze()
        return stmts[0].name
    finally:
        renpy.game.exception_info = old_exception_info

def include_module(name):
    if False:
        return 10
    '\n    :doc: other\n\n    Similar to :func:`renpy.load_module`, but instead of loading the module right away,\n    inserts it into the init queue somewhere after the current AST node.\n\n    The module may not contain init blocks lower than the block that includes the module.\n    For example, if your module contains an init 10 block, the latest you can load it is\n    init 10.\n\n    Module loading may only occur from inside an init block.\n    '
    if not renpy.game.context().init_phase:
        raise Exception('Module loading is only allowed in init code.')
    renpy.game.script.include_module(name)

def pop_call():
    if False:
        print('Hello World!')
    "\n    :doc: label\n    :name: renpy.pop_call\n\n    Pops the current call from the call stack, without returning to the\n    location. Also reverts the values of :func:`dynamic <renpy.dynamic>`\n    variables, the same way the Ren'Py return statement would.\n\n    This can be used if a label that is called decides not to return\n    to its caller.\n    "
    renpy.game.context().pop_call()
pop_return = pop_call

def call_stack_depth():
    if False:
        return 10
    '\n    :doc: label\n\n    Returns the depth of the call stack of the current context - the number\n    of calls that have run without being returned from or popped from the\n    call stack.\n    '
    return len(renpy.game.context().return_stack)

def game_menu(screen=None):
    if False:
        i = 10
        return i + 15
    '\n    :undocumented: Probably not what we want in the presence of\n    screens.\n    '
    if screen is None:
        call_in_new_context('_game_menu')
    else:
        call_in_new_context('_game_menu', _game_menu_screen=screen)

def shown_window():
    if False:
        print('Hello World!')
    '\n    :doc: other\n\n    Call this to indicate that the window has been shown. This interacts\n    with the "window show" statement, which shows an empty window whenever\n    this functions has not been called during an interaction.\n    '
    renpy.game.context().scene_lists.shown_window = True

class placement(renpy.revertable.RevertableObject):

    def __init__(self, p):
        if False:
            print('Hello World!')
        super(placement, self).__init__()
        self.xpos = p[0]
        self.ypos = p[1]
        self.xanchor = p[2]
        self.yanchor = p[3]
        self.xoffset = p[4]
        self.yoffset = p[5]
        self.subpixel = p[6]

    @property
    def pos(self):
        if False:
            i = 10
            return i + 15
        return (self.xpos, self.ypos)

    @property
    def anchor(self):
        if False:
            while True:
                i = 10
        return (self.xanchor, self.yanchor)

    @property
    def offset(self):
        if False:
            i = 10
            return i + 15
        return (self.xoffset, self.yoffset)

def get_placement(d):
    if False:
        i = 10
        return i + 15
    "\n    :doc: image_func\n\n    This gets the placement of displayable d. There's very little warranty on this\n    information, as it might change when the displayable is rendered, and might not\n    exist until the displayable is first rendered.\n\n    This returns an object with the following fields, each corresponding to a style\n    property:\n\n    * pos\n    * xpos\n    * ypos\n    * anchor\n    * xanchor\n    * yanchor\n    * offset\n    * xoffset\n    * yoffset\n    * subpixel\n    "
    p = d.get_placement()
    return placement(p)

def get_image_bounds(tag, width=None, height=None, layer=None):
    if False:
        i = 10
        return i + 15
    '\n    :doc: image_func\n\n    If an image with `tag` exists on `layer`, returns the bounding box of\n    that image. Returns None if the image is not found.\n\n    The bounding box is an (x, y, width, height) tuple. The components of\n    the tuples are expressed in pixels, and may be floating point numbers.\n\n    `width`, `height`\n        The width and height of the area that contains the image. If None,\n        defaults the width and height of the screen, respectively.\n\n    `layer`\n        If None, uses the default layer for `tag`.\n    '
    tag = tag.split()[0]
    layer = default_layer(layer, tag)
    if width is None:
        width = renpy.config.screen_width
    if height is None:
        height = renpy.config.screen_height
    return scene_lists().get_image_bounds(layer, tag, width, height)
Render = renpy.display.render.Render
render = renpy.display.render.render
IgnoreEvent = renpy.display.core.IgnoreEvent
redraw = renpy.display.render.redraw

def is_pixel_opaque(d, width, height, st, at, x, y):
    if False:
        i = 10
        return i + 15
    '\n    :doc: udd_utility\n\n    Returns whether the pixel at (x, y) is opaque when this displayable\n    is rendered by ``renpy.render(d, width, height, st, at)``.\n    '
    return bool(render(renpy.easy.displayable(d), width, height, st, at).is_pixel_opaque(x, y))

class Displayable(renpy.display.displayable.Displayable, renpy.revertable.RevertableObject):
    pass

class Container(renpy.display.layout.Container, renpy.revertable.RevertableObject):
    _list_type = renpy.revertable.RevertableList

def get_roll_forward():
    if False:
        print('Hello World!')
    return renpy.game.interface.shown_window

def cache_pin(*args):
    if False:
        while True:
            i = 10
    '\n    :undocumented: Cache pin is deprecated.\n    '
    new_pins = renpy.revertable.RevertableSet()
    for i in args:
        im = renpy.easy.displayable(i)
        if not isinstance(im, renpy.display.im.ImageBase):
            raise Exception('Cannot pin non-image-manipulator %r' % im)
        new_pins.add(im)
    renpy.store._cache_pin_set = new_pins | renpy.store._cache_pin_set

def cache_unpin(*args):
    if False:
        while True:
            i = 10
    '\n    :undocumented: Cache pin is deprecated.\n    '
    new_pins = renpy.revertable.RevertableSet()
    for i in args:
        im = renpy.easy.displayable(i)
        if not isinstance(im, renpy.display.im.ImageBase):
            raise Exception('Cannot unpin non-image-manipulator %r' % im)
        new_pins.add(im)
    renpy.store._cache_pin_set = renpy.store._cache_pin_set - new_pins

def expand_predict(d):
    if False:
        i = 10
        return i + 15
    '\n    :undocumented:\n\n    Use the fnmatch function to expland `d` for the purposes of prediction.\n    '
    if not isinstance(d, basestring):
        return [d]
    if not '*' in d:
        return [d]
    if '.' in d:
        l = list_files(False)
    else:
        l = list_images()
    return fnmatch.filter(l, d)

def start_predict(*args):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: image_func\n\n    This function takes one or more displayables as arguments. It causes\n    Ren\'Py to predict those displayables during every interaction until\n    the displayables are removed by :func:`renpy.stop_predict`.\n\n    If a displayable name is a string containing one or more \\*\n    characters, the asterisks are used as a wildcard pattern. If there\n    is at least one . in the string, the pattern is matched against\n    filenames, otherwise it is matched against image names.\n\n    For example::\n\n        $ renpy.start_predict("eileen *")\n\n    starts predicting all images with the name eileen, while::\n\n        $ renpy.start_predict("images/concert*.*")\n\n    matches all files starting with concert in the images directory.\n\n    Prediction will occur during normal gameplay. To wait for prediction\n    to complete, use the `predict` argument to :func:`renpy.pause`.\n    '
    new_predict = renpy.revertable.RevertableSet(renpy.store._predict_set)
    for i in args:
        for d in expand_predict(i):
            d = renpy.easy.displayable(d)
            new_predict.add(d)
    renpy.store._predict_set = new_predict

def stop_predict(*args):
    if False:
        while True:
            i = 10
    "\n    :doc: image_func\n\n    This function takes one or more displayables as arguments. It causes\n    Ren'Py to stop predicting those displayables during every interaction.\n\n    Wildcard patterns can be used as described in :func:`renpy.start_predict`.\n    "
    new_predict = renpy.revertable.RevertableSet(renpy.store._predict_set)
    for i in args:
        for d in expand_predict(i):
            d = renpy.easy.displayable(d)
            new_predict.discard(d)
    renpy.store._predict_set = new_predict

def start_predict_screen(_screen_name, *args, **kwargs):
    if False:
        print('Hello World!')
    "\n    :doc: screens\n\n    Causes Ren'Py to start predicting the screen named `_screen_name`\n    with the given arguments. This replaces any previous prediction\n    of `_screen_name`. To stop predicting a screen, call :func:`renpy.stop_predict_screen`.\n\n    Prediction will occur during normal gameplay. To wait for prediction\n    to complete, use the `predict` argument to :func:`renpy.pause`.\n    "
    new_predict = renpy.revertable.RevertableDict(renpy.store._predict_screen)
    new_predict[_screen_name] = (args, kwargs)
    renpy.store._predict_screen = new_predict

def stop_predict_screen(name):
    if False:
        print('Hello World!')
    "\n    :doc: screens\n\n    Causes Ren'Py to stop predicting the screen named `name`.\n    "
    new_predict = renpy.revertable.RevertableDict(renpy.store._predict_screen)
    new_predict.pop(name, None)
    renpy.store._predict_screen = new_predict

def call_screen(_screen_name, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    :doc: screens\n    :args: (_screen_name, *args, _with_none=True, _mode="screen", **kwargs)\n\n    The programmatic equivalent of the call screen statement.\n\n    This shows `_screen_name` as a screen, then causes an interaction\n    to occur. The screen is hidden at the end of the interaction, and\n    the result of the interaction is returned.\n\n    Positional arguments, and keyword arguments that do not begin with\n    _ are passed to the screen.\n\n    If `_with_none` is false, "with None" is not run at the end of end\n    of the interaction.\n\n    If `_mode` is passed, it will be the mode of this interaction,\n    otherwise the mode will be "screen".\n    '
    mode = 'screen'
    if '_mode' in kwargs:
        mode = kwargs.pop('_mode')
    renpy.exports.mode(mode)
    with_none = renpy.config.implicit_with_none
    if '_with_none' in kwargs:
        with_none = kwargs.pop('_with_none')
    show_screen(_screen_name, *args, _transient=True, **kwargs)
    roll_forward = renpy.exports.roll_forward_info()
    can_roll_forward = renpy.display.screen.get_screen_roll_forward(_screen_name)
    if can_roll_forward is None:
        can_roll_forward = renpy.config.call_screen_roll_forward
    if not can_roll_forward:
        roll_forward = None
    try:
        rv = renpy.ui.interact(mouse='screen', type='screen', roll_forward=roll_forward)
    except (renpy.game.JumpException, renpy.game.CallException) as e:
        rv = e
    renpy.exports.checkpoint(rv)
    if with_none:
        renpy.game.interface.do_with(None, None)
    if isinstance(rv, (renpy.game.JumpException, renpy.game.CallException)):
        raise rv
    return rv

@renpy_pure
def list_files(common=False):
    if False:
        return 10
    '\n    :doc: file\n\n    Lists the files in the game directory and archive files. Returns\n    a list of files, with / as the directory separator.\n\n    `common`\n        If true, files in the common directory are included in the\n        listing.\n    '
    rv = []
    for (_dir, fn) in renpy.loader.listdirfiles(common):
        if fn.startswith('saves/'):
            continue
        rv.append(fn)
    rv.sort()
    return rv

def get_renderer_info():
    if False:
        i = 10
        return i + 15
    '\n    :doc: other\n\n    Returns a dictionary, giving information about the renderer Ren\'Py is\n    currently using. Defined keys are:\n\n    ``"renderer"``\n        A string giving the name of the renderer that is in use.\n\n    ``"resizable"``\n        True if and only if the window is resizable.\n\n    ``"additive"``\n        True if and only if the renderer supports additive blending.\n\n    ``"model"``\n        Present and true if model-based rendering is supported.\n\n    Other, renderer-specific, keys may also exist. The dictionary should\n    be treated as immutable. This should only be called once the display\n    has been started (that is, after the init phase has finished).\n    '
    return renpy.display.draw.info

def display_reset():
    if False:
        return 10
    '\n    :undocumented: Used internally.\n\n    Causes the display to be restarted at the start of the next interaction.\n    '
    renpy.display.interface.display_reset = True

def mode(mode):
    if False:
        print('Hello World!')
    "\n    :doc: modes\n\n    Causes Ren'Py to enter the named mode, or stay in that mode if it's\n    already in it.\n    "
    ctx = renpy.game.context()
    if not ctx.use_modes:
        return
    modes = ctx.modes
    try:
        ctx.use_modes = False
        if mode != modes[0]:
            for c in renpy.config.mode_callbacks:
                c(mode, modes)
    finally:
        ctx.use_modes = True
    if mode in modes:
        modes.remove(mode)
    modes.insert(0, mode)

def get_mode():
    if False:
        return 10
    '\n    :doc: modes\n\n    Returns the current mode, or None if it is not defined.\n    '
    ctx = renpy.game.context()
    if not ctx.use_modes:
        return None
    modes = ctx.modes
    return modes[0]

def notify(message):
    if False:
        while True:
            i = 10
    "\n    :doc: other\n\n    Causes Ren'Py to display the `message` using the notify screen. By\n    default, this will cause the message to be dissolved in, displayed\n    for two seconds, and dissolved out again.\n\n    This is useful for actions that otherwise wouldn't produce feedback,\n    like screenshots or quicksaves.\n\n    Only one notification is displayed at a time. If a second notification\n    is displayed, the first notification is replaced.\n\n    This function just calls :var:`config.notify`, allowing its implementation\n    to be replaced by assigning a new function to that variable.\n    "
    renpy.config.notify(message)

def display_notify(message):
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    The default implementation of :func:`renpy.notify`.\n    '
    hide_screen('notify')
    show_screen('notify', message=message)
    renpy.display.tts.notify_text = renpy.text.extras.filter_alt_text(message)
    restart_interaction()

@renpy_pure
def variant(name):
    if False:
        print('Hello World!')
    "\n    :doc: screens\n\n    Returns true if `name` is a screen variant that corresponds to the\n    context in which Ren'Py is currently executing. See :ref:`screen-variants`\n    for more details. This function can be used as the condition in an\n    if statement to switch behavior based on the selected screen variant.\n\n    `name` can also be a list of variants, in which case this function\n    returns True if any of the variants would.\n    "
    if isinstance(name, basestring):
        return name in renpy.config.variants
    else:
        for n in name:
            if n in renpy.config.variants:
                return True
        return False

def vibrate(duration):
    if False:
        i = 10
        return i + 15
    '\n    :doc: other\n\n    Causes the device to vibrate for `duration` seconds. Currently, this\n    is only supported on Android.\n    '
    if renpy.android:
        import android
        android.vibrate(duration)

def get_say_attributes():
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    Gets the attributes associated with the current say statement, or\n    None if no attributes are associated with this statement.\n\n    This is only valid when executing or predicting a say statement.\n    '
    return renpy.game.context().say_attributes

def get_side_image(prefix_tag, image_tag=None, not_showing=None, layer=None):
    if False:
        print('Hello World!')
    "\n    :doc: side\n\n    This attempts to find an image to show as the side image.\n\n    It begins by determining a set of image attributes. If `image_tag` is\n    given, it gets the image attributes from the tag. Otherwise, it gets\n    them from the currently showing character. If no attributes are available\n    for the tag, this returns None.\n\n    It then looks up an image with the tag `prefix_tag`, and the image tage (either\n    from `image_tag` or the currently showing character) and the set of image\n    attributes as attributes. If such an image exists, it's returned.\n\n    If not_showing is True, this only returns a side image if the image the\n    attributes are taken from is not on the screen. If Nome, the value\n    is taken from :var:`config.side_image_only_not_showing`.\n\n    If `layer` is None, uses the default layer for the currently showing\n    tag.\n    "
    if not_showing is None:
        not_showing = renpy.config.side_image_only_not_showing
    images = renpy.game.context().images
    if image_tag is not None:
        image_layer = default_layer(layer, image_tag)
        attrs = (image_tag,) + images.get_attributes(image_layer, image_tag)
        if renpy.config.side_image_requires_attributes and len(attrs) < 2:
            return None
    else:
        attrs = renpy.store._side_image_attributes
    if not attrs:
        return None
    attr_layer = default_layer(layer, attrs)
    if not_showing and images.showing(attr_layer, (attrs[0],)):
        return None
    required = [attrs[0]]
    optional = list(attrs[1:])
    return images.choose_image(prefix_tag, required, optional, None)

def get_physical_size():
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: other\n\n    Returns the size of the physical window.\n    '
    return renpy.display.draw.get_physical_size()

def set_physical_size(size):
    if False:
        print('Hello World!')
    '\n    :doc: other\n\n    Attempts to set the size of the physical window to `size`. This has the\n    side effect of taking the screen out of fullscreen mode.\n    '
    width = int(size[0])
    height = int(size[1])
    renpy.game.preferences.fullscreen = False
    if get_renderer_info()['resizable']:
        renpy.game.preferences.physical_size = (width, height)
        if renpy.display.draw is not None:
            renpy.display.draw.resize()

def reset_physical_size():
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: other\n\n    Attempts to set the size of the physical window to the size specified\n    using :var:`renpy.config.physical_height` and :var:`renpy.config.physical_width`,\n    or the size set using :var:`renpy.config.screen_width` and :var:`renpy.config.screen_height`\n    if not set.\n    '
    set_physical_size((renpy.config.physical_width or renpy.config.screen_width, renpy.config.physical_height or renpy.config.screen_height))

@renpy_pure
def fsencode(s, force=False):
    if False:
        print('Hello World!')
    '\n    :doc: file_rare\n    :name: renpy.fsencode\n\n    Converts s from unicode to the filesystem encoding.\n    '
    if not PY2 and (not force):
        return s
    if not isinstance(s, str):
        return s
    fsencoding = sys.getfilesystemencoding() or 'utf-8'
    return s.encode(fsencoding)

@renpy_pure
def fsdecode(s):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: file_rare\n    :name: renpy.fsdecode\n\n    Converts s from filesystem encoding to unicode.\n    '
    if isinstance(s, str):
        return s
    fsencoding = sys.getfilesystemencoding() or 'utf-8'
    return s.decode(fsencoding)
from renpy.editor import launch_editor

def get_image_load_log(age=None):
    if False:
        print('Hello World!')
    '\n    :doc: other\n\n    A generator that yields a log of image loading activity. For the last 100\n    image loads, this returns:\n\n    * The time the image was loaded (in seconds since the epoch).\n    * The filename of the image that was loaded.\n    * A boolean that is true if the image was preloaded, and false if the\n      game stalled to load it.\n\n    The entries are ordered from newest to oldest.\n\n    `age`\n        If not None, only images that have been loaded in the past `age`\n        seconds are included.\n\n    The image load log is only kept if config.developer = True.\n    '
    if age is not None:
        deadline = time.time() - age
    else:
        deadline = 0
    for i in renpy.display.im.cache.load_log:
        if i[0] < deadline:
            break
        yield i

def end_replay():
    if False:
        for i in range(10):
            print('nop')
    "\n    :doc: replay\n\n    If we're in a replay, ends the replay immediately. Otherwise, does\n    nothing.\n    "
    if renpy.store._in_replay:
        raise renpy.game.EndReplay()

def save_persistent():
    if False:
        while True:
            i = 10
    '\n    :doc: persistent\n\n    Saves the persistent data to disk.\n    '
    renpy.persistent.update(True)

def is_seen(ever=True):
    if False:
        return 10
    '\n    :doc: other\n\n    Returns true if the current line has been seen by the player.\n\n    If `ever` is true, we check to see if the line has ever been seen by the\n    player. If false, we check if the line has been seen in the current\n    play-through.\n    '
    return renpy.game.context().seen_current(ever)

def get_mouse_pos():
    if False:
        print('Hello World!')
    '\n    :doc: other\n\n    Returns an (x, y) tuple giving the location of the mouse pointer or the\n    current touch location. If the device does not support a mouse and is not\n    currently being touched, x and y are numbers, but not meaningful.\n    '
    return renpy.display.draw.get_mouse_pos()

def set_mouse_pos(x, y, duration=0):
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    Jump the mouse pointer to the location given by arguments x and y.\n    If the device does not have a mouse pointer, this does nothing.\n\n    `duration`\n        The time it will take to perform the move, in seconds.\n        During this time, the mouse may be unresponsive.\n    '
    renpy.display.interface.set_mouse_pos(x, y, duration)

def set_autoreload(autoreload):
    if False:
        i = 10
        return i + 15
    '\n    :doc: reload\n\n    Sets the autoreload flag, which determines if the game will be\n    automatically reloaded after file changes. Autoreload will not be\n    fully enabled until the game is reloaded with :func:`renpy.reload_script`.\n    '
    renpy.autoreload = autoreload

def get_autoreload():
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: reload\n\n    Gets the autoreload flag.\n    '
    return renpy.autoreload

def count_dialogue_blocks():
    if False:
        while True:
            i = 10
    "\n    :doc: other\n\n    Returns the number of dialogue blocks in the game's original language.\n    "
    return renpy.game.script.translator.count_translates()

def count_seen_dialogue_blocks():
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    Returns the number of dialogue blocks the user has seen in any play-through\n    of the current game.\n    '
    return renpy.game.seen_translates_count

def count_newly_seen_dialogue_blocks():
    if False:
        i = 10
        return i + 15
    '\n    :doc: other\n\n    Returns the number of dialogue blocks the user has seen for the first time\n    during this session.\n    '
    return renpy.game.new_translates_count

def substitute(s, scope=None, translate=True):
    if False:
        while True:
            i = 10
    '\n    :doc: text_utility\n\n    Applies translation and new-style formatting to the string `s`.\n\n    `scope`\n        If not None, a scope which is used in formatting, in addition to the\n        default store.\n\n    `translate`\n        Determines if translation occurs.\n\n    Returns the translated and formatted string.\n    '
    return renpy.substitutions.substitute(s, scope=scope, translate=translate)[0]

def munge(name, filename=None):
    if False:
        return 10
    '\n    :doc: other\n\n    Munges `name`, which must begin with __.\n\n    `filename`\n        The filename the name is munged into. If None, the name is munged\n        into the filename containing the call to this function.\n    '
    if filename is None:
        filename = sys._getframe(1).f_code.co_filename
    if not name.startswith('__'):
        return name
    if name.endswith('__'):
        return name
    return renpy.lexer.munge_filename(filename) + name[2:]

def get_return_stack():
    if False:
        while True:
            i = 10
    '\n    :doc: label\n\n    Returns a list giving the current return stack. The return stack is a\n    list of statement names.\n\n    The statement names will be strings (for labels), or opaque tuples (for\n    non-label statements).\n    '
    return renpy.game.context().get_return_stack()

def set_return_stack(stack):
    if False:
        return 10
    '\n    :doc: label\n\n    Sets the current return stack. The return stack is a list of statement\n    names.\n\n    Statement names may be strings (for labels) or opaque tuples (for\n    non-label statements).\n\n    The most common use of this is to use::\n\n        renpy.set_return_stack([])\n\n    to clear the return stack.\n    '
    renpy.game.context().set_return_stack(stack)

def invoke_in_thread(fn, *args, **kwargs):
    if False:
        return 10
    "\n    :doc: other\n\n    Invokes the function `fn` in a background thread, passing it the\n    provided arguments and keyword arguments. Restarts the interaction\n    once the thread returns.\n\n    This function creates a daemon thread, which will be automatically\n    stopped when Ren'Py is shutting down.\n\n    This thread is very limited in what it can do with the Ren'Py API.\n    Changing store variables is allowed, as are calling calling the following\n    functions:\n\n    * :func:`renpy.restart_interaction`\n    * :func:`renpy.invoke_in_main_thread`\n    * :func:`renpy.queue_event`\n\n    Most other portions of the Ren'Py API are expected to be called from\n    the main thread.\n\n    This does not work on the web platform, except for immediately returning\n    without an error.\n    "
    if renpy.emscripten:
        return

    def run():
        if False:
            for i in range(10):
                print('nop')
        try:
            fn(*args, **kwargs)
        except Exception:
            import traceback
            traceback.print_exc()
        restart_interaction()
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()

def invoke_in_main_thread(fn, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    :doc: other\n\n    This runs the given function with the given arguments in the main\n    thread, if it is not already running in the main thread. The function\n    runs in an interaction context similar to an event handler.\n\n    This may not be called during the init phase.\n    '
    if renpy.game.context().init_phase:
        raise Exception('invoke_in_main_thread may not be called during the init phase.')
    renpy.display.interface.invoke_queue.append((fn, args, kwargs))
    restart_interaction()

def cancel_gesture():
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: gesture\n\n    Cancels the current gesture, preventing the gesture from being recognized.\n    This should be called by displayables that have gesture-like behavior.\n    '
    renpy.display.gesture.recognizer.cancel()

def execute_default_statement(start=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    :undocumented:\n\n    Executes the default statement.\n\n    `start`\n        This is true at the start of the game, and false at other\n        times.\n    '
    for i in renpy.ast.default_statements:
        i.execute_default(start)
    for i in renpy.config.after_default_callbacks:
        i()

def write_log(s, *args):
    if False:
        print('Hello World!')
    '\n    :undocumented:\n\n    Writes to log.txt.\n    '
    renpy.display.log.write(s, *args)

def predicting():
    if False:
        while True:
            i = 10
    "\n    :doc: other\n\n    Returns true if Ren'Py is currently in a predicting phase.\n    "
    return renpy.display.predict.predicting

def get_line_log():
    if False:
        print('Hello World!')
    '\n    :undocumented:\n\n    Returns the list of lines that have been shown since the last time\n    :func:`renpy.clear_line_log` was called.\n    '
    return renpy.game.context().line_log[:]

def clear_line_log():
    if False:
        for i in range(10):
            print('nop')
    '\n    :undocumented:\n\n    Clears the line log.\n    '
    renpy.game.context().line_log = []

def add_layer(layer, above=None, below=None, menu_clear=True, sticky=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: image_func\n\n    Adds a new layer to the screen. If the layer already exists, this\n    function does nothing.\n\n    One of `behind` or `above` must be given.\n\n    `layer`\n        A string giving the name of the new layer to add.\n\n    `above`\n        If not None, a string giving the name of a layer the new layer will\n        be placed above.\n\n    `below`\n        If not None, a string giving the name of a layer the new layer will\n        be placed below.\n\n    `menu_clear`\n        If true, this layer will be cleared when entering the game menu\n        context, and restored when leaving it.\n\n    `sticky`\n        If true, any tags added to this layer will have it become their\n        default layer until they are hidden. If None, this layer will be\n        sticky only if other sticky layers already exist.\n    '
    layers = renpy.config.layers
    if layer in renpy.config.layers:
        return
    if above is not None and below is not None:
        raise Exception('The above and below arguments to renpy.add_layer are mutually exclusive.')
    elif above is not None:
        try:
            index = layers.index(above) + 1
        except ValueError:
            raise Exception("Layer '%s' does not exist." % above)
    elif below is not None:
        try:
            index = layers.index(below)
        except ValueError:
            raise Exception("Layer '%s' does not exist." % below)
    else:
        raise Exception('The renpy.add_layer function requires either the above or below argument.')
    layers.insert(index, layer)
    if menu_clear:
        renpy.config.menu_clear_layers.append(layer)
    if sticky or (sticky is None and renpy.config.sticky_layers):
        renpy.config.sticky_layers.append(layer)

def maximum_framerate(t):
    if False:
        i = 10
        return i + 15
    "\n    :doc: other\n\n    Forces Ren'Py to draw the screen at the maximum framerate for `t` seconds.\n    If `t` is None, cancels the maximum framerate request.\n    "
    if renpy.display.interface is not None:
        renpy.display.interface.maximum_framerate(t)
    elif t is None:
        renpy.display.core.initial_maximum_framerate = 0
    else:
        renpy.display.core.initial_maximum_framerate = max(renpy.display.core.initial_maximum_framerate, t)

def is_start_interact():
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    Returns true if restart_interaction has not been called during the current\n    interaction. This can be used to determine if the interaction is just being\n    started, or has been restarted.\n    '
    return renpy.display.interface.start_interact

def play(filename, channel=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    :doc: audio\n\n    Plays a sound effect. If `channel` is None, it defaults to\n    :var:`config.play_channel`. This is used to play sounds defined in\n    styles, :propref:`hover_sound` and :propref:`activate_sound`.\n    '
    if filename is None:
        return
    if channel is None:
        channel = renpy.config.play_channel
    renpy.audio.music.play(filename, channel=channel, loop=False, **kwargs)

def get_editable_input_value():
    if False:
        print('Hello World!')
    '\n    :undocumented:\n\n    Returns the current input value, and a flag that is true if it is editable.\n    and false otherwise.\n    '
    return (renpy.display.behavior.current_input_value, renpy.display.behavior.input_value_active)

def set_editable_input_value(input_value, editable):
    if False:
        i = 10
        return i + 15
    '\n    :undocumented:\n\n    Sets the currently active input value, and if it should be marked as\n    editable.\n    '
    renpy.display.behavior.current_input_value = input_value
    renpy.display.behavior.input_value_active = editable

def get_refresh_rate(precision=5):
    if False:
        print('Hello World!')
    "\n    :doc: other\n\n    Returns the refresh rate of the current screen, as a floating-point\n    number of frames per second.\n\n    `precision`\n        The raw data Ren'Py gets is number of frames per second, rounded down.\n        This means that a monitor that runs at 59.95 frames per second will\n        be reported at 59 fps. The precision argument reduces the precision\n        of this reading, such that the only valid readings are multiples of\n        the precision.\n\n        Since all monitor framerates tend to be multiples of 5 (25, 30, 60,\n        75, and 120), this likely will improve accuracy. Setting precision\n        to 1 disables this.\n    "
    precision *= 1.0
    info = renpy.display.get_info()
    rv = info.refresh_rate
    rv = round(rv / precision) * precision
    return rv

def get_identifier_checkpoints(identifier):
    if False:
        return 10
    '\n    :doc: rollback\n\n    Given a rollback_identifier from a HistoryEntry object, returns the number\n    of checkpoints that need to be passed to :func:`renpy.rollback` to reach\n    that identifier. Returns None of the identifier is not in the rollback\n    history.\n    '
    return renpy.game.log.get_identifier_checkpoints(identifier)

def get_adjustment(bar_value):
    if False:
        while True:
            i = 10
    '\n    :doc: screens\n\n    Given `bar_value`, a  :class:`BarValue`, returns the :func:`ui.adjustment`\n    if uses. The adjustment has the following to attributes defined:\n\n    .. attribute:: value\n\n        The current value of the bar.\n\n    .. attribute:: range\n\n        The current range of the bar.\n    '
    return bar_value.get_adjustment()

def get_skipping():
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    Returns "slow" if the Ren\'Py is skipping, "fast" if Ren\'Py is fast skipping,\n    and None if it is not skipping.\n    '
    return renpy.config.skipping

def get_texture_size():
    if False:
        i = 10
        return i + 15
    '\n    :undocumented:\n\n    Returns the number of bytes of memory locked up in OpenGL textures and the\n    number of textures that are defined.\n    '
    return renpy.display.draw.get_texture_size()
old_battery = False

def get_on_battery():
    if False:
        while True:
            i = 10
    "\n    :doc: other\n\n    Returns True if Ren'Py is running on a device that is powered by an internal\n    battery, or False if the device is being charged by some external source.\n    "
    global old_battery
    pi = pygame_sdl2.power.get_power_info()
    if pi.state == pygame_sdl2.POWERSTATE_UNKNOWN:
        return old_battery
    elif pi.state == pygame_sdl2.POWERSTATE_ON_BATTERY:
        old_battery = True
        return True
    else:
        old_battery = False
        return False

def get_say_image_tag():
    if False:
        return 10
    '\n    :doc: image_func\n\n    Returns the tag corresponding to the currently speaking character (the\n    `image` argument given to that character). Returns None if no character\n    is speaking or the current speaking character does not have a corresponding\n    image tag.\n    '
    if renpy.store._side_image_attributes is None:
        return None
    return renpy.store._side_image_attributes[0]

class LastSay:
    """
    :undocumented:
    Object containing info about the last dialogue line.
    Returned by the last_say function.
    """

    def __init__(self, who, what, args, kwargs):
        if False:
            return 10
        self._who = who
        self.what = what
        self.args = args
        self.kwargs = kwargs

    @property
    def who(self):
        if False:
            print('Hello World!')
        return eval_who(self._who)

def last_say():
    if False:
        return 10
    "\n    :doc: other\n\n    Returns an object containing information about the last say statement.\n\n    While this can be called during a say statement, if the say statement is using\n    a normal Character, the information will be about the *current* say statement,\n    instead of the preceding one.\n\n    `who`\n        The speaker. This is usually a :func:`Character` object, but this\n        is not required.\n\n    `what`\n        A string with the dialogue spoken. This may be None if dialogue\n        hasn't been shown yet, for example at the start of the game.\n\n    `args`\n        A tuple of arguments passed to the last say statement.\n\n    `kwargs`\n        A dictionary of keyword arguments passed to the last say statement.\n\n    .. warning::\n\n        Like other similar functions, the object this returns is meant to be used\n        in the short term after the function is called. Including it in save data\n        or making it participate in rollback is not advised.\n    "
    return LastSay(who=renpy.store._last_say_who, what=renpy.store._last_say_what, args=renpy.store._last_say_args, kwargs=renpy.store._last_say_kwargs)

def is_skipping():
    if False:
        print('Hello World!')
    "\n    :doc: other\n\n    Returns True if Ren'Py is currently skipping (in fast or slow skip mode),\n    or False otherwise.\n    "
    return not not renpy.config.skipping

def is_init_phase():
    if False:
        return 10
    "\n    :doc: other\n\n    Returns True if Ren'Py is currently executing init code, or False otherwise.\n    "
    return renpy.game.context().init_phase

def add_to_all_stores(name, value):
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    Adds the `value` by the `name` to all creator defined namespaces. If the name\n    already exist in that namespace - do nothing for it.\n\n    This function may only be run from inside an init block. It is an\n    error to run this function once the game has started.\n    '
    if not is_init_phase():
        raise Exception('add_to_all_stores is only allowed in init code.')
    for (_k, ns) in renpy.python.store_dicts.items():
        if name not in ns:
            ns[name] = value

def get_zorder_list(layer):
    if False:
        print('Hello World!')
    '\n    :doc: image_func\n\n    Returns a list of (tag, zorder) pairs for `layer`.\n    '
    return renpy.display.core.scene_lists().get_zorder_list(layer)

def change_zorder(layer, tag, zorder):
    if False:
        i = 10
        return i + 15
    '\n    :doc: image_func\n\n    Changes the zorder of `tag` on `layer` to `zorder`.\n    '
    return renpy.display.core.scene_lists().change_zorder(layer, tag, zorder)
sdl_dll = False

def get_sdl_dll():
    if False:
        print('Hello World!')
    "\n    :doc: sdl\n\n    This returns a ctypes.cdll object that refers to the library that contains\n    the instance of SDL2 that Ren'Py is using.\n\n    If this can not be done, None is returned.\n    "
    global sdl_dll
    if sdl_dll is not False:
        return sdl_dll
    try:
        DLLS = [None, 'librenpython.dll', 'librenpython.dylib', 'librenpython.so', 'SDL2.dll', 'libSDL2.dylib', 'libSDL2-2.0.so.0']
        import ctypes
        for i in DLLS:
            try:
                dll = ctypes.cdll[i]
                dll.SDL_GetError
            except Exception:
                continue
            sdl_dll = dll
            return dll
    except Exception:
        pass
    sdl_dll = None
    return None

def get_sdl_window_pointer():
    if False:
        i = 10
        return i + 15
    '\n    :doc: sdl\n\n    Returns a pointer (of type ctypes.c_void_p) to the main window, or None\n    if the main window is not displayed, or some other problem occurs.\n    '
    try:
        window = pygame_sdl2.display.get_window()
        if window is None:
            return
        return window.get_sdl_window_pointer()
    except Exception:
        return None

def is_mouse_visible():
    if False:
        print('Hello World!')
    '\n    :doc: other\n\n    Returns True if the mouse cursor is visible, False otherwise.\n    '
    if not renpy.display.interface:
        return True
    if not renpy.display.interface.mouse_focused:
        return False
    return renpy.display.interface.is_mouse_visible()

def get_mouse_name(interaction=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    :doc: other\n\n    Returns the name of the mouse that should be shown.\n\n\n    `interaction`\n        If true, get a mouse name that is based on the type of interaction\n        occuring. (This is rarely useful.)\n    '
    if not renpy.display.interface:
        return 'default'
    return renpy.display.interface.get_mouse_name(interaction=interaction)

def set_focus(screen, id, layer='screens'):
    if False:
        i = 10
        return i + 15
    "\n    :doc: screens\n\n    This attempts to focus the displayable with `id` in the screen `screen`.\n    Focusing will fail if the displayable isn't found, the window isn't\n    focused, or something else is grabbing focus.\n\n    The focus may change if the mouse moves, even slightly, after this call\n    is processed.\n    "
    renpy.display.focus.override = (screen, id, layer)
    renpy.display.interface.last_event = None
    restart_interaction()

def check_permission(permission):
    if False:
        print('Hello World!')
    '\n    :doc: android_permission\n\n    Checks to see if an Android permission has been granted to this application.\n\n    `permission`\n        A string giving the name of the permission, for example, "android.permission.WRITE_EXTERNAL_STORAGE".\n\n    Returns true if the permission has been granted, false if it has not or if called on\n    a non-Android platform.\n    '
    if not renpy.android:
        return False
    from jnius import autoclass
    PythonSDLActivity = autoclass('org.renpy.android.PythonSDLActivity')
    activity = PythonSDLActivity.mActivity
    try:
        return activity.checkSelfPermission(permission) == 0
    except Exception:
        return False

def request_permission(permission):
    if False:
        while True:
            i = 10
    '\n    :doc: android_permission\n\n    Asks Android to grant a permission to this application. The user may be\n    prompted to grant the permission.\n\n    `permission`\n        A string giving the name of the permission, for example, "android.permission.WRITE_EXTERNAL_STORAGE".\n\n    Returns true if the permission has been granted, false if not or if called on a\n    non-Android platform.\n    '
    if not renpy.android:
        return False
    return get_sdl_dll().SDL_AndroidRequestPermission(permission.encode('utf-8'))

def clear_retain(layer='screens', prefix='_retain'):
    if False:
        return 10
    '\n    :doc: other\n\n    Clears all retained screens\n    '
    for i in get_showing_tags(layer):
        if i.startswith(prefix):
            hide_screen(i)

def confirm(message):
    if False:
        while True:
            i = 10
    '\n    :doc: other\n\n    This causes the a yes/no prompt screen with the given message\n    to be displayed, and dismissed when the player hits yes or no.\n\n    Returns True if the player hits yes, and False if the player hits no.\n\n    `message`\n        The message that will be displayed.\n\n    See :func:`Confirm` for a similar Action.\n    '
    Return = renpy.store.Return
    renpy.store.layout.yesno_screen(message, yes=Return(True), no=Return(False))
    return renpy.ui.interact()

class FetchError(Exception):
    """
    :undocumented:

    The type of errors raised by :func:`renpy.fetch`.
    """
    pass

def fetch_requests(url, method, data, content_type, timeout):
    if False:
        while True:
            i = 10
    '\n    :undocumented:\n\n    Used by fetch on non-emscripten systems.\n\n    Returns either a bytes object, or a FetchError.\n    '
    import threading
    import requests
    resp = [None]

    def make_request():
        if False:
            print('Hello World!')
        try:
            r = requests.request(method, url, data=data, timeout=timeout, headers={'Content-Type': content_type} if data is not None else {})
            r.raise_for_status()
            resp[0] = r.content
        except Exception as e:
            resp[0] = FetchError(str(e))
    t = threading.Thread(target=make_request)
    t.start()
    while resp[0] is None:
        renpy.exports.pause(0)
    t.join()
    return resp[0]

def fetch_emscripten(url, method, data, content_type, timeout):
    if False:
        print('Hello World!')
    '\n    :undocumented:\n\n    Used by fetch on emscripten systems.\n\n    Returns either a bytes object, or a FetchError.\n    '
    import emscripten
    import time
    import os
    fn = '/req-' + str(time.time()) + '.data'
    with open(fn, 'wb') as f:
        if data is not None:
            f.write(data)
    url = url.replace('"', '\\"')
    if method == 'GET' or method == 'HEAD':
        command = 'fetchFile("{method}", "{url}", null, "{fn}", null)'.format(method=method, url=url, fn=fn, content_type=content_type)
    else:
        command = 'fetchFile("{method}", "{url}", "{fn}", "{fn}", "{content_type}")'.format(method=method, url=url, fn=fn, content_type=content_type)
    fetch_id = emscripten.run_script_int(command)
    status = 'PENDING'
    message = 'Pending.'
    start = time.time()
    while time.time() - start < timeout:
        renpy.exports.pause(0)
        result = emscripten.run_script_string('fetchFileResult({})'.format(fetch_id))
        (status, _ignored, message) = result.partition(' ')
        if status != 'PENDING':
            break
    try:
        if status == 'OK':
            with open(fn, 'rb') as f:
                return f.read()
        else:
            return FetchError(message)
    finally:
        os.unlink(fn)

def fetch(url, method=None, data=None, json=None, content_type=None, timeout=5, result='bytes'):
    if False:
        while True:
            i = 10
    '\n    :doc: fetch\n\n    This performs an HTTP (or HTTPS) request to the given URL, and returns\n    the content of that request. If it fails, raises a FetchError exception,\n    with text that describes the failure. (But may not be suitable for\n    presentation to the user.)\n\n    `url`\n        The URL to fetch.\n\n    `method`\n        The method to use. Generally one of "GET", "POST", or "PUT", but other\n        HTTP methods are possible. If `data` or `json` are not None, defaults to\n        "POST", otherwise defaults to GET.\n\n    `data`\n        If not None, a byte string of data to send with the request.\n\n    `json`\n        If not None, a JSON object to send with the request. This takes precendence\n        over `data`.\n\n    `content_type`\n        The content type of the data. If not given, defaults to "application/json"\n        if `json` is not None, or "application/octet-stream" otherwise. Only\n        used on a POST or PUT request.\n\n    `timeout`\n        The number of seconds to wait for the request to complete.\n\n    `result`\n        How to process the result. If "bytes", returns the raw bytes of the result.\n        If "text", decodes the result using UTF-8 and returns a unicode string. If "json",\n        decodes the result as JSON. (Other exceptions may be generated by the decoding\n        process.)\n\n    While waiting for `timeout` to pass, this will repeatedly call :func:`renpy.pause`\\ (0),\n    so Ren\'Py doesn\'t lock up. It may make sense to display a screen to the user\n    to let them know what is going on.\n\n    This function should work on all platforms. However, on the web platform,\n    requests going to a different origin than the game will fail unless allowed\n    by CORS.\n    '
    import json as _json
    if data is not None and json is not None:
        raise FetchError('data and json arguments are mutually exclusive.')
    if result not in ('bytes', 'text', 'json'):
        raise FetchError("result must be one of 'bytes', 'text', or 'json'.")
    if renpy.game.context().init_phase:
        raise FetchError('renpy.fetch may not be called during init.')
    if method is None:
        if data is not None or json is not None:
            method = 'POST'
        else:
            method = 'GET'
    if content_type is None:
        if json is not None:
            content_type = 'application/json'
        else:
            content_type = 'application/octet-stream'
    if json is not None:
        data = _json.dumps(json).encode('utf-8')
    if renpy.emscripten:
        content = fetch_emscripten(url, method, data, content_type, timeout)
    else:
        content = fetch_requests(url, method, data, content_type, timeout)
    if isinstance(content, Exception):
        raise content
    if result == 'bytes':
        return content
    elif result == 'text':
        return content.decode('utf-8')
    elif result == 'json':
        return _json.loads(content)