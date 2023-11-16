from collections import defaultdict
import json
import time
from visidata import colors, vd, clipdraw, ColorAttr
__all__ = ['Animation', 'AnimationMgr']

class AttrDict(dict):

    def __getattr__(self, k):
        if False:
            for i in range(10):
                print('nop')
        try:
            v = self[k]
            if isinstance(v, dict):
                v = AttrDict(v)
            return v
        except KeyError as e:
            if k.startswith('__'):
                raise AttributeError from e
            return ''

    def __setattr__(self, k, v):
        if False:
            for i in range(10):
                print('nop')
        self[k] = v

    def __dir__(self):
        if False:
            while True:
                i = 10
        return self.keys()

class Animation:

    def __init__(self, fp):
        if False:
            for i in range(10):
                print('nop')
        self.frames = defaultdict(AttrDict)
        self.groups = defaultdict(AttrDict)
        self.height = 0
        self.width = 0
        self.load_from(fp)

    def iterdeep(self, rows, x=0, y=0, parents=None):
        if False:
            return 10
        'Walk rows deeply and generate (row, x, y, [ancestors]) for each row.'
        for r in rows:
            newparents = (parents or []) + [r]
            if r.type == 'frame':
                continue
            if r.ref:
                assert r.type == 'ref'
                g = self.groups[r.ref]
                yield from self.iterdeep(map(AttrDict, g.rows or []), x + r.x, y + r.y, newparents)
            else:
                yield (r, x + r.x, y + r.y, newparents)
                yield from self.iterdeep(map(AttrDict, r.rows or []), x + r.x, x + r.y, newparents)

    def load_from(self, fp):
        if False:
            return 10
        for line in fp.readlines():
            r = AttrDict(json.loads(line))
            if r.type == 'frame':
                self.frames[r.id].update(r)
            elif r.type == 'group':
                self.groups[r.id].update(r)
            f = self.frames[r.frame or '']
            if not f.rows:
                f.rows = [r]
            else:
                f.rows.append(r)
        self.total_ms = 0
        if self.frames:
            self.total_ms = sum((f.duration_ms or 0 for f in self.frames.values()))
            for f in self.frames.values():
                for (r, x, y, _) in self.iterdeep(f.rows):
                    self.width = max(self.width, x + len(r.text))
                    self.height = max(self.height, y)

    def draw(self, scr, *, t=0, x=0, y=0, loop=False, attr=ColorAttr(), **kwargs):
        if False:
            i = 10
            return i + 15
        for (r, dx, dy, _) in self.iterdeep(self.frames[''].rows):
            clipdraw(scr, y + dy, x + dx, r.text, attr.update(colors[r.color], 2))
        if not self.total_ms:
            return None
        ms = int(t * 1000) % self.total_ms
        for f in self.frames.values():
            ms -= int(f.duration_ms or 0)
            if ms < 0:
                for (r, dx, dy, _) in self.iterdeep(f.rows):
                    clipdraw(scr, y + dy, x + dx, r.text, colors[r.color])
                return -ms / 1000
        if loop:
            return -ms / 1000

class AnimationMgr:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.library = {}
        self.active = []

    def trigger(self, name, **kwargs):
        if False:
            return 10
        if name in self.library:
            self.active.append((time.time(), self.library[name], kwargs))
        else:
            vd.debug('unknown drawing "%s"' % name)

    def load(self, name, fp):
        if False:
            print('Hello World!')
        self.library[name] = Animation(fp)

    @property
    def maxHeight(self):
        if False:
            i = 10
            return i + 15
        return max((anim.height for (_, anim, _) in self.active)) if self.active else 0

    @property
    def maxWidth(self):
        if False:
            print('Hello World!')
        return max((anim.width for (_, anim, _) in self.active)) if self.active else 0

    def draw(self, scr, t=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Draw all active animations on *scr* at time *t*.  Return next t to be called at.'
        if t is None:
            t = time.time()
        times = []
        done = []
        for row in self.active:
            (startt, anim, akwargs) = row
            kwargs.update(akwargs)
            nextt = anim.draw(scr, t=t - startt, **kwargs)
            if nextt is None:
                if not akwargs.get('loop'):
                    done.append(row)
            else:
                times.append(t + nextt)
        for row in done:
            self.active.remove(row)
        return min(times) if times else None
vd.addGlobals({'Animation': Animation, 'AnimationMgr': AnimationMgr})