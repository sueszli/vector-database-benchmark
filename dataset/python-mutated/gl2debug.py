from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy

def geometry(mesh, transform, width, height):
    if False:
        print('Hello World!')
    '\n    Debugs the drawing of geometry by applying `transform` to `mesh`, and\n    then projecting it onto the screen.\n    '
    points = mesh.get_points()
    triangles = mesh.get_triangles()
    l = ['Mesh:']
    for (a, b, c) in triangles:
        l.append('{}-{}-{}'.format(a, b, c))
    print(' '.join(l))
    sxlist = []
    sylist = []
    for (i, p) in enumerate(points):
        (px, py, pz, pw) = p
        (tx, ty, tz, tw) = transform.transform(px, py, pz, pw, components=4)
        dtx = tx / tw
        dty = ty / tw
        sx = width * (dtx + 1.0) / 2.0
        sy = height * (1.0 - dty) / 2.0
        print('{:3g}: {: >9.4f} {: >9.4f} {: >3.1f} {: >3.1f} | {: >9.6f} {: >9.6f} {:>3.1f} {:>3.1f} | {:> 9.4f} {:< 9.4f}'.format(i, px, py, pz, pw, tx, ty, tz, tw, sx, sy))
        sxlist.append(sx)
        sylist.append(sy)
    if sxlist:
        minsx = min(sxlist)
        minsy = min(sylist)
        maxsx = max(sxlist)
        maxsy = max(sylist)
        print('     ({:> 9.4f}, {:< 9.4f}) - ({:> 9.4f}, {:< 9.4f})'.format(minsx, minsy, maxsx, maxsy))