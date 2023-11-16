from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
import operator
warp_spec = None

def warp():
    if False:
        print('Hello World!')
    '\n    Given a filename and line number, this attempts to warp the user\n    to that filename and line number.\n    '
    global warp_spec
    spec = warp_spec
    warp_spec = None
    if spec is None:
        return None
    if ':' not in spec:
        raise Exception('No : found in warp location.')
    (filename, line) = spec.split(':', 1)
    line = int(line)
    if not renpy.config.developer:
        raise Exception("Can't warp, developer mode disabled.")
    if not filename.startswith('game/'):
        filename = 'game/' + filename
    prev = {}
    seenset = set(renpy.game.script.namemap.values())

    def add(node, next):
        if False:
            print('Hello World!')
        if next not in prev:
            prev[next] = node
            return
        old = prev[next]

        def prefer(fn):
            if False:
                return 10
            if fn(node, old):
                return node
            if fn(old, node):
                return old
            return None
        n = None
        n = n or prefer(lambda a, b: a.filename == next.filename and b.filename != next.filename)
        n = n or prefer(lambda a, b: a.linenumber <= next.linenumber and b.linenumber > next.linenumber)
        n = n or prefer(lambda a, b: a.linenumber >= b.linenumber)
        n = n or node
        prev[next] = n
    for n in seenset:
        if isinstance(n, renpy.ast.Translate) and n.language:
            continue
        if isinstance(n, renpy.ast.Menu):
            for i in n.items:
                if i[2] is not None:
                    add(n, i[2][0])
        if isinstance(n, renpy.ast.Jump):
            if not n.expression and n.target in renpy.game.script.namemap:
                add(n, renpy.game.script.namemap[n.target])
                continue
        if isinstance(n, renpy.ast.While):
            add(n, n.block[0])
        if isinstance(n, renpy.ast.If):
            seen_true = False
            for (condition, block) in n.entries:
                add(n, block[0])
                if condition == 'True':
                    seen_true = True
            if seen_true:
                continue
        if isinstance(n, renpy.ast.UserStatement):
            add(n, n.get_next())
        elif getattr(n, 'next', None) is not None:
            add(n, n.next)
    candidates = [(n.linenumber, n) for n in seenset if n.filename == filename and n.linenumber <= line]
    if not candidates:
        raise Exception('Could not find a statement to warp to. ({})'.format(spec))
    candidates.sort(key=operator.itemgetter(0))
    node = candidates[-1][1]
    run = []
    n = node
    while True:
        n = prev.pop(n, None)
        if n:
            run.append(n)
        else:
            break
    run.reverse()
    run = run[-renpy.config.warp_limit:]
    renpy.config.skipping = 'fast'
    for n in run:
        if n.can_warp():
            try:
                n.execute()
            except Exception:
                pass
    renpy.config.skipping = None
    renpy.game.after_rollback = True
    renpy.exports.block_rollback()
    renpy.game.context().goto_label(node.name)
    renpy.game.context().come_from(node.name, '_after_warp')
    raise renpy.game.RestartContext()