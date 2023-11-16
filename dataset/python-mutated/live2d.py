from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
from typing import Any
import renpy
import renpy.gl2.live2dmotion
from renpy.gl2.gl2shadercache import register_shader
from renpy.display.core import absolute
try:
    import renpy.gl2.live2dmodel as live2dmodel
except ImportError:
    live2dmodel = None
import sys
import os
import json
import collections
did_onetime_init = False

def onetime_init():
    if False:
        i = 10
        return i + 15
    global did_onetime_init
    if did_onetime_init:
        return
    if renpy.windows:
        dll = 'Live2DCubismCore.dll'
    elif renpy.macintosh:
        dll = 'libLive2DCubismCore.dylib'
    else:
        dll = 'libLive2DCubismCore.so'
    fn = os.path.join(os.path.dirname(sys.executable), dll)
    if os.path.exists(fn):
        dll = fn
    if not PY2:
        dll = dll.encode('utf-8')
    if not renpy.gl2.live2dmodel.load(dll):
        raise Exception('Could not load Live2D. {} was not found.'.format(dll))
    did_onetime_init = True
did_init = False

def init():
    if False:
        i = 10
        return i + 15
    '\n    Called to initialize Live2D, if needed.\n    '
    global did_init
    if did_init:
        return
    if live2dmodel is None:
        raise Exception('Live2D has not been built.')
    if not renpy.config.gl2:
        raise Exception('Live2D requires that config.gl2 be True.')
    if renpy.emscripten:
        raise Exception('Live2D is not supported the web platform.')
    onetime_init()
    register_shader('live2d.mask', variables='\n        uniform sampler2D tex0;\n        uniform sampler2D tex1;\n        attribute vec4 a_position;\n        attribute vec2 a_tex_coord;\n        varying vec2 v_tex_coord;\n        varying vec2 v_mask_coord;\n    ', vertex_200='\n        v_tex_coord = a_tex_coord;\n        v_mask_coord = vec2(a_position.x / 2.0 + .5, -a_position.y / 2.0 + .5);\n    ', fragment_200='\n        vec4 color = texture2D(tex0, v_tex_coord);\n        vec4 mask = texture2D(tex1, v_mask_coord);\n        gl_FragColor = color * mask.a;\n    ')
    register_shader('live2d.inverted_mask', variables='\n        uniform sampler2D tex0;\n        uniform sampler2D tex1;\n        attribute vec4 a_position;\n        attribute vec2 a_tex_coord;\n        varying vec2 v_tex_coord;\n        varying vec2 v_mask_coord;\n    ', vertex_200='\n        v_tex_coord = a_tex_coord;\n        v_mask_coord = vec2(a_position.x / 2.0 + .5, -a_position.y / 2.0 + .5);\n    ', fragment_200='\n        vec4 color = texture2D(tex0, v_tex_coord);\n        vec4 mask = texture2D(tex1, v_mask_coord);\n        gl_FragColor = color * (1.0 - mask.a);\n    ')
    register_shader('live2d.colors', variables='\n        uniform vec4 u_multiply;\n        uniform vec4 u_screen;\n    ', fragment_250='\n        gl_FragColor.rgb = gl_FragColor.rgb * u_multiply.rgb;\n        gl_FragColor.rgb = (gl_FragColor.rgb + u_screen.rgb * gl_FragColor.a) - (gl_FragColor.rgb * u_screen.rgb);\n    ')
    register_shader('live2d.flip_texture', variables='\n        varying vec2 v_tex_coord;\n    ', vertex_250='\n        v_tex_coord.y = 1.0 - v_tex_coord.y;\n    ')
    renpy.config.interact_callbacks.append(update_states)
    did_init = True

def reset():
    if False:
        for i in range(10):
            print('nop')
    "\n    Resets this module when Ren'Py reloads the script.\n    "
    global did_init
    did_init = False
    common_cache.clear()

def reset_states():
    if False:
        while True:
            i = 10
    "\n    Resets the Live2D states when Ren'Py restarts the game.\n    "
    states.clear()

class Live2DExpression(object):
    """
    The data corresponding to an expression.
    """

    def __init__(self, parameters, fadein, fadeout):
        if False:
            return 10
        self.parameters = parameters
        self.fadein = fadein
        self.fadeout = fadeout

class Live2DCommon(object):
    """
    This object stores information that is common to all of the Live2D
    displayables that use the same .model3.json file, so this information
    only needs to be loaded once. This should not leak into the save games,
    but is loaded at init time.
    """

    def __init__(self, filename, default_fade):
        if False:
            print('Hello World!')
        init()
        if not filename.endswith('.json'):
            suffix = filename.rpartition('/')[2]
            filename = filename + '/' + suffix + '.model3.json'
        if renpy.config.log_live2d_loading:
            renpy.display.log.write('Loading Live2D from %r.', filename)
        if not renpy.loader.loadable(filename, directory='images'):
            raise Exception('Live2D model {} does not exist.'.format(filename))
        model_name = filename.rpartition('/')[2].partition('.')[0].lower()
        self.base = filename.rpartition('/')[0]
        if self.base:
            self.base += '/'
        with renpy.loader.load(filename, directory='images') as f:
            self.model_json = json.load(f)
        self.model = renpy.gl2.live2dmodel.Live2DModel(self.base + self.model_json['FileReferences']['Moc'])
        self.textures = []
        for i in self.model_json['FileReferences']['Textures']:
            self.textures.append(renpy.easy.displayable(self.base + i))
        motion_files = {}
        expression_files = {}
        for i in renpy.exports.list_files():
            if not i.startswith(self.base):
                continue
            if i.endswith('motion3.json'):
                i = i[len(self.base):]
                motion_files[i] = {'File': i}
            elif i.endswith('.exp3.json'):
                i = i[len(self.base):]
                expression_files[i] = {'File': i}

        def walk_json_files(o, d):
            if False:
                print('Hello World!')
            if isinstance(o, list):
                for i in o:
                    walk_json_files(i, d)
                return
            if 'File' in o:
                d[o['File']] = o
                return
            for i in o.values():
                walk_json_files(i, d)
        walk_json_files(self.model_json['FileReferences'].get('Motions', {}), motion_files)
        walk_json_files(self.model_json['FileReferences'].get('Expressions', {}), expression_files)
        self.attributes = set(['still', 'null'])
        self.motions = {'still': renpy.gl2.live2dmotion.NullMotion()}
        for i in motion_files.values():
            name = i['File'].lower().rpartition('/')[2].partition('.')[0]
            (prefix, _, suffix) = name.partition('_')
            if prefix == model_name:
                name = suffix
            if renpy.loader.loadable(self.base + i['File'], directory='images'):
                if renpy.config.log_live2d_loading:
                    renpy.display.log.write(' - motion %s -> %s', name, i['File'])
                self.motions[name] = renpy.gl2.live2dmotion.Motion(self.base + i['File'], i.get('FadeInTime', default_fade), i.get('FadeOutTime', default_fade))
                self.attributes.add(name)
        self.expressions = {'null': Live2DExpression([], 0.0, 0.0)}
        for i in expression_files.values():
            name = i['File'].lower().rpartition('/')[2].partition('.')[0]
            (prefix, _, suffix) = name.partition('_')
            if prefix == model_name:
                name = suffix
            if renpy.loader.loadable(self.base + i['File'], directory='images'):
                if renpy.config.log_live2d_loading:
                    renpy.display.log.write(' - expression %s -> %s', name, i['File'])
                if name in self.attributes:
                    raise Exception('Name {!r} is already specified as a motion.'.format(name))
                with renpy.loader.load(self.base + i['File'], directory='images') as f:
                    expression_json = json.load(f)
                self.expressions[name] = Live2DExpression(expression_json.get('Parameters', []), expression_json.get('FadeInTime', default_fade), expression_json.get('FadeOutTime', default_fade))
                self.attributes.add(name)
        for i in self.model_json.get('Groups', []):
            name = i['Name']
            ids = i['Ids']
            if i['Target'] == 'Parameter':
                self.model.parameter_groups[name] = ids
            elif i['Target'] == 'Opacity':
                self.model.opacity_groups[name] = ids
        self.all_expressions = dict(self.expressions)
        self.nonexclusive = {}
        self.seamless = False
        self.attribute_function = None
        self.attribute_filter = None
        self.update_function = None

    def apply_aliases(self, aliases):
        if False:
            i = 10
            return i + 15
        for (k, v) in aliases.items():
            target = None
            expression = False
            if v in self.motions:
                target = self.motions
            elif v in self.expressions:
                target = self.expressions
                expression = True
            elif v in self.nonexclusive:
                target = self.nonexclusive
                expression = True
            else:
                raise Exception('Name {!r} is not a known motion or expression.'.format(v))
            if k in target:
                raise Exception('Name {!r} is already specified as a motion or expression.'.format(k))
            target[k] = target[v]
            if expression:
                self.all_expressions[k] = target[v]

    def apply_nonexclusive(self, nonexclusive):
        if False:
            while True:
                i = 10
        for i in nonexclusive:
            if i not in self.expressions:
                raise Exception('Name {!r} is not a known expression.'.format(i))
            self.nonexclusive[i] = self.expressions.pop(i)

    def apply_seamless(self, value):
        if False:
            print('Hello World!')
        self.seamless = value

    def is_seamless(self, motion):
        if False:
            for i in range(10):
                print('nop')
        if self.seamless is True:
            return True
        elif self.seamless is False:
            return False
        else:
            return motion in self.seamless
common_cache = {}

class Live2DState(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.mark = False
        self.cycle_new = False
        self.old = None
        self.new = None
        self.old_base_time = 0
        self.new_base_time = 0
        self.expressions = []
        self.old_expressions = []

    def update_expressions(self, expressions, now):
        if False:
            i = 10
            return i + 15
        '\n        Updates the lists of new and old expressions.\n\n        `expressions`\n            A list of strings giving expression names.\n\n        `now`\n            The time the current displayable started showing.\n        '
        current = set((name for (name, _) in self.expressions))
        self.old_expressions = [(name, shown, hidden) for (name, shown, hidden) in self.old_expressions if name not in expressions] + [(name, shown, now) for (name, shown) in self.expressions if name not in expressions]
        self.expressions = [(name, shown) for (name, shown) in self.expressions if name in expressions]
        self.expressions += [(name, now) for name in expressions if name not in current]
states = collections.defaultdict(Live2DState)

def update_states():
    if False:
        for i in range(10):
            print('nop')
    '\n    Called once per interact to walk the tree of displayables and find\n    the old and new live2d states.\n    '

    def visit(d):
        if False:
            i = 10
            return i + 15
        if not isinstance(d, Live2D):
            return
        if d.name is None:
            return
        state = states[d.name]
        if state.mark:
            return
        state.mark = True
        if state.new is d:
            return
        if state.old is d:
            return
        if state.cycle_new:
            state.old = state.new
            state.old_base_time = state.new_base_time
        else:
            state.old = None
            state.old_base_time = None
            state.expressions = []
            state.old_expressions = []
        state.new = d
        if d.sustain:
            state.new_base_time = state.old_base_time
        else:
            state.new_base_time = None
        state.cycle_new = True
    sls = renpy.display.core.scene_lists()
    for d in sls.get_all_displayables(current=True):
        if d is not None:
            d.visit_all(visit)
    for s in states.values():
        if not s.mark:
            s.cycle_new = False
        s.mark = False

class Live2D(renpy.display.displayable.Displayable):
    nosave = ['common_cache']
    common_cache = None
    _duplicatable = True
    used_nonexclusive = None

    def create_common(self, default_fade=1.0):
        if False:
            i = 10
            return i + 15
        rv = common_cache.get(self.filename, None)
        if rv is None:
            rv = Live2DCommon(self.filename, default_fade)
            common_cache[self.filename] = rv
        self.common_cache = rv
        return rv

    @property
    def common(self):
        if False:
            for i in range(10):
                print('nop')
        if self.common_cache is not None:
            return self.common_cache
        return self.create_common(self.filename)

    def __init__(self, filename, zoom=None, top=0.0, base=1.0, height=1.0, loop=False, aliases={}, fade=None, motions=None, expression=None, nonexclusive=None, used_nonexclusive=None, seamless=None, sustain=False, attribute_function=None, attribute_filter=None, update_function=None, default_fade=1.0, **properties):
        if False:
            print('Hello World!')
        super(Live2D, self).__init__(**properties)
        self.filename = filename
        self.motions = motions
        self.expression = expression
        self.used_nonexclusive = used_nonexclusive
        self.zoom = zoom
        self.top = top
        self.base = base
        self.height = height
        self.loop = loop
        self.fade = fade
        self.sustain = sustain
        self.name = None
        common = self.create_common(default_fade)
        if nonexclusive:
            common.apply_nonexclusive(nonexclusive)
        if aliases:
            common.apply_aliases(aliases)
        if seamless is not None:
            common.apply_seamless(seamless)
        if attribute_function is not None:
            common.attribute_function = attribute_function
        if attribute_filter is not None:
            common.attribute_filter = attribute_filter
        if update_function is not None:
            common.update_function = update_function

    def _duplicate(self, args):
        if False:
            i = 10
            return i + 15
        if not self._duplicatable:
            return self
        if not args:
            return self
        common = self.common
        motions = []
        used_nonexclusive = []
        expression = None
        sustain = False
        if '_sustain' in args.args:
            attributes = tuple((i for i in args.args if i != '_sustain'))
            sustain = True
        else:
            attributes = args.args
        if common.attribute_function is not None:
            attributes = common.attribute_function(attributes)
        for i in attributes:
            if i in common.motions:
                motions.append(i)
                continue
            if i in common.nonexclusive:
                used_nonexclusive.append(i)
                continue
            if i in common.expressions:
                if expression is not None:
                    raise Exception('When showing {}, {} and {} are both live2d expressions.'.format(' '.join(args.name), i, expression))
                expression = i
                continue
            raise Exception('When showing {}, {} is not a known attribute.'.format(' '.join(args.name), i))
        rv = Live2D(self.filename, motions=motions, zoom=self.zoom, top=self.top, base=self.base, height=self.height, loop=self.loop, fade=self.fade, expression=expression, used_nonexclusive=used_nonexclusive, sustain=sustain)
        rv.name = args.name
        rv._duplicatable = False
        return rv

    def _list_attributes(self, tag, attributes):
        if False:
            print('Hello World!')
        common = self.common
        available = set(common.attributes)
        for i in attributes:
            if i in common.expressions:
                available -= set(common.expressions)
        available |= set(attributes)
        return [i for i in common.attributes if i in available]

    def _choose_attributes(self, tag, attributes, optional):
        if False:
            return 10
        attributes = [i for i in attributes if i != '_sustain']
        common = self.common
        rv = [i for i in attributes if i in common.motions]
        for i in list(attributes) + list(optional):
            if i in common.expressions:
                rv.insert(0, i)
                break
        for i in sorted(list(attributes)):
            if i in common.nonexclusive:
                rv.append(i)
        for i in sorted(list(optional)):
            if i in common.nonexclusive:
                rv.append(i)
        if set(attributes) - set(rv):
            return None
        rv = tuple(rv)
        if common.attribute_filter:
            rv = common.attribute_filter(rv)
            if not isinstance(rv, tuple):
                rv = tuple(rv)
        if not any((i in common.motions for i in rv)):
            rv = ('_sustain',) + tuple((i for i in optional if i in common.motions)) + rv
        return rv

    def update(self, common, st, st_fade):
        if False:
            for i in range(10):
                print('nop')
        "\n        This updates the common model with the information taken from the\n        motions associated with this object. It returns the delay until\n        Ren'Py needs to cause a redraw to occur, or None if no delay\n        should occur.\n        "
        if not self.motions:
            return
        do_fade_in = True
        do_fade_out = True
        last_frame = False
        current_index = 0
        motion = None
        motion_st = st
        if st_fade is not None:
            motion_st = st - st_fade
        for m in self.motions:
            motion = common.motions.get(m, None)
            if motion is None:
                continue
            if motion.duration > st:
                break
            elif motion.duration > motion_st and (not common.is_seamless(m)):
                break
            motion_st -= motion.duration
            st -= motion.duration
            current_index += 1
        else:
            if motion is None:
                return None
            m = self.motions[-1]
            if not self.loop or not motion.duration:
                st = motion.duration
                last_frame = True
            elif st_fade is not None and (not common.is_seamless(m)):
                motion_start = motion_st - motion_st % motion.duration
                if st - motion_start > motion.duration:
                    st = motion.duration
                    last_frame = True
        if motion is None:
            return None
        if current_index < len(self.motions):
            current_name = self.motions[current_index]
        else:
            current_name = self.motions[-1]
        if current_index > 0:
            last_name = self.motions[current_index - 1]
        else:
            last_name = None
        if current_index < len(self.motions) - 1:
            next_name = self.motions[current_index + 1]
        elif self.loop:
            next_name = self.motions[-1]
        else:
            next_name = None
        if last_name == current_name and common.is_seamless(current_name):
            do_fade_in = False
        if next_name == current_name and common.is_seamless(current_name) and (st_fade is None):
            do_fade_out = False
        motion_data = motion.get(st, st_fade, do_fade_in, do_fade_out)
        for (k, v) in motion_data.items():
            (kind, key) = k
            (factor, value) = v
            if kind == 'PartOpacity':
                common.model.set_part_opacity(key, value)
            elif kind == 'Parameter':
                common.model.set_parameter(key, value, factor)
            elif kind == 'Model':
                common.model.set_parameter(key, value, factor)
        if last_frame:
            return None
        else:
            return motion.wait(st, st_fade, do_fade_in, do_fade_out)

    def update_expressions(self, st):
        if False:
            i = 10
            return i + 15
        common = self.common
        model = common.model
        state = states[self.name]
        now = renpy.display.interface.frame_time
        state.old_expressions = [(name, shown, hidden) for (name, shown, hidden) in state.old_expressions if now - hidden < common.all_expressions[name].fadeout]
        expressions = list(self.used_nonexclusive)
        if self.expression:
            expressions.append(self.expression)
        state.update_expressions(expressions, now - st)
        redraw = None
        for (name, shown, hidden) in state.old_expressions:
            weight = 1.0
            e = common.all_expressions[name]
            if e.fadein > 0 and now - shown < e.fadein:
                weight = min(weight, (now - shown) / e.fadein)
                redraw = 0
            if e.fadeout > 0 and now - hidden < e.fadeout:
                weight = min(weight, 1.0 - (now - hidden) / e.fadeout)
                redraw = 0
            for i in e.parameters:
                model.blend_parameter(i['Id'], i['Blend'], i['Value'], weight=weight)
        for (name, shown) in state.expressions:
            weight = 1.0
            e = common.all_expressions[name]
            if e.fadein > 0 and now - shown < e.fadein:
                weight = min(weight, (now - shown) / e.fadein)
                redraw = 0
            for i in e.parameters:
                model.blend_parameter(i['Id'], i['Blend'], i['Value'], weight=weight)
        return redraw

    def blend_parameter(self, name, blend, value, weight=1.0):
        if False:
            for i in range(10):
                print('nop')
        if blend not in ('Add', 'Multiply', 'Overwrite'):
            raise Exception('Unknown blend mode {!r}'.format(blend))
        self.common.model.blend_parameter(name, blend, value, weight)

    def render(self, width, height, st, at):
        if False:
            while True:
                i = 10
        common = self.common
        model = common.model
        fade = self.fade if self.fade is not None else renpy.store._live2d_fade
        if not self.name:
            fade = False
        if fade:
            state = states[self.name]
            if state.new is not self:
                fade = False
            if state.new_base_time is None:
                state.new_base_time = renpy.display.interface.frame_time - st
            if state.old is None:
                fade = False
            elif state.old_base_time is None:
                fade = False
            elif state.old.common is not self.common:
                fade = False
        model.reset_parameters()
        if fade:
            t = renpy.display.interface.frame_time - state.new_base_time
        else:
            t = st
        new_redraw = self.update(common, t, None)
        if fade:
            old_redraw = state.old.update(common, renpy.display.interface.frame_time - state.old_base_time, st)
        else:
            old_redraw = None
        model.finish_parameters()
        expression_redraw = self.update_expressions(st)
        if common.update_function is None:
            user_redraw = None
        else:
            user_redraw = common.update_function(self, st)
        redraws = [new_redraw, old_redraw, expression_redraw, user_redraw]
        redraws = [i for i in redraws if i is not None]
        if redraws:
            renpy.display.render.redraw(self, min(redraws))
        textures = [renpy.display.render.render(d, width, height, st, at) for d in common.textures]
        (sw, sh) = model.get_size()
        zoom = self.zoom
        if zoom is None:
            top = absolute.compute_raw(self.top, sh)
            base = absolute.compute_raw(self.base, sh)
            size = max(base - top, 1.0)
            zoom = 1.0 * self.height * renpy.config.screen_height / size
        else:
            size = sh
            top = 0
        rend = model.render(textures, zoom)
        rv = renpy.exports.Render(sw * zoom, size * zoom)
        rv.blit(rend, (0, -top * zoom))
        return rv

    def visit(self):
        if False:
            return 10
        return self.common.textures
_has_live2d = None

def has_live2d():
    if False:
        print('Hello World!')
    '\n    :doc: live2d\n\n    Returns True if Live2d is supported on the current platform, and\n    False otherwise.\n    '
    global _has_live2d
    if _has_live2d is None:
        try:
            init()
            _has_live2d = True
        except Exception:
            _has_live2d = False
    return _has_live2d