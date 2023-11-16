"""Pygame Drawing algorithms written in Python. (Work in Progress)

Implement Pygame's Drawing Algorithms in a Python version for testing
and debugging.
"""
from collections import namedtuple
from math import floor, ceil

def frac(value):
    if False:
        return 10
    'return fractional part of x'
    return value - floor(value)

def inv_frac(value):
    if False:
        for i in range(10):
            print('nop')
    'return inverse fractional part of x'
    return 1 - (value - floor(value))
BoundingBox = namedtuple('BoundingBox', ['left', 'top', 'right', 'bottom'])
Point = namedtuple('Point', ['x', 'y'])

def set_at(surf, in_x, in_y, color):
    if False:
        print('Hello World!')
    'Set the color of a pixel in a surface'
    surf.set_at((in_x, in_y), color)

def draw_pixel(surf, pos, color, bright, blend=True):
    if False:
        for i in range(10):
            print('nop')
    'draw one blended pixel with given brightness.'
    try:
        other_col = surf.get_at(pos) if blend else (0, 0, 0, 0)
    except IndexError:
        return
    new_color = tuple((bright * col + (1 - bright) * pix for (col, pix) in zip(color, other_col)))
    surf.set_at(pos, new_color)

def _drawhorzline(surf, color, x_from, in_y, x_to):
    if False:
        for i in range(10):
            print('nop')
    if x_from == x_to:
        surf.set_at((x_from, in_y), color)
        return
    (start, end) = (x_from, x_to) if x_from <= x_to else (x_to, x_from)
    for line_x in range(start, end + 1):
        surf.set_at((line_x, in_y), color)

def _drawvertline(surf, color, in_x, y_from, y_to):
    if False:
        print('Hello World!')
    if y_from == y_to:
        surf.set_at((in_x, y_from), color)
        return
    (start, end) = (y_from, y_to) if y_from <= y_to else (y_to, y_from)
    for line_y in range(start, end + 1):
        surf.set_at((in_x, line_y), color)

def _clip_and_draw_horizline(surf, color, x_from, in_y, x_to):
    if False:
        i = 10
        return i + 15
    'draw clipped horizontal line.'
    clip = surf.get_clip()
    if in_y < clip.y or in_y >= clip.y + clip.h:
        return
    x_from = max(x_from, clip.x)
    x_to = min(x_to, clip.x + clip.w - 1)
    if x_to < clip.x or x_from >= clip.x + clip.w:
        return
    _drawhorzline(surf, color, x_from, in_y, x_to)

def _clip_and_draw_vertline(surf, color, in_x, y_from, y_to):
    if False:
        i = 10
        return i + 15
    'draw clipped vertical line.'
    clip = surf.get_clip()
    if in_x < clip.x or in_x >= clip.x + clip.w:
        return
    y_from = max(y_from, clip.y)
    y_to = min(y_to, clip.y + clip.h - 1)
    if y_to < clip.y or y_from >= clip.y + clip.h:
        return
    _drawvertline(surf, color, in_x, y_from, y_to)
LEFT_EDGE = 1
RIGHT_EDGE = 2
BOTTOM_EDGE = 4
TOP_EDGE = 8

def encode(pos, b_box):
    if False:
        print('Hello World!')
    'returns a code that defines position with respect to a bounding box'
    return (pos[0] < b_box.left) * LEFT_EDGE + (pos[0] > b_box.right) * RIGHT_EDGE + (pos[1] < b_box.top) * TOP_EDGE + (pos[1] > b_box.bottom) * BOTTOM_EDGE

def clip_line(line, b_box, use_float=False):
    if False:
        print('Hello World!')
    'Algorithm to calculate the clipped line.\n\n    We calculate the coordinates of the part of the line segment within the\n    bounding box (defined by left, top, right, bottom). The we write\n    the coordinates of the line segment into "line", much like the C-algorithm.\n    With `use_float` True, clip_line is usable for float-clipping.\n\n    Returns: true if the line segment cuts the bounding box (false otherwise)\n    '

    def inside(code):
        if False:
            while True:
                i = 10
        return not code

    def accept(code_a, code_b):
        if False:
            return 10
        return not (code_a or code_b)

    def reject(code_a, code_b):
        if False:
            print('Hello World!')
        return code_a and code_b
    assert isinstance(line, list)
    (x_1, y_1, x_2, y_2) = line
    dtype = float if use_float else int
    while True:
        code1 = encode((x_1, y_1), b_box)
        code2 = encode((x_2, y_2), b_box)
        if accept(code1, code2):
            line[:] = (x_1, y_1, x_2, y_2)
            return True
        if reject(code1, code2):
            return False
        if inside(code1):
            (x_1, x_2) = (x_2, x_1)
            (y_1, y_2) = (y_2, y_1)
            (code1, code2) = (code2, code1)
        slope = (y_2 - y_1) / float(x_2 - x_1) if x_2 != x_1 else 1.0
        if code1 & LEFT_EDGE:
            y_1 += dtype((b_box.left - x_1) * slope)
            x_1 = b_box.left
        elif code1 & RIGHT_EDGE:
            y_1 += dtype((b_box.right - x_1) * slope)
            x_1 = b_box.right
        elif code1 & BOTTOM_EDGE:
            if x_2 != x_1:
                x_1 += dtype((b_box.bottom - y_1) / slope)
            y_1 = b_box.bottom
        elif code1 & TOP_EDGE:
            if x_2 != x_1:
                x_1 += dtype((b_box.top - y_1) / slope)
            y_1 = b_box.top

def _draw_line(surf, color, start, end):
    if False:
        for i in range(10):
            print('nop')
    'draw a non-horizontal line (without anti-aliasing).'
    if start.x == end.x:
        raise ValueError
    slope = abs((end.y - start.y) / (end.x - start.x))
    error = 0.0
    if slope < 1:
        if end.x < start.x:
            (start.x, end.x) = (end.x, start.x)
            (start.y, end.y) = (end.y, start.y)
        line_y = start.y
        dy_sign = 1 if start.y < end.y else -1
        for line_x in range(start.x, end.x + 1):
            set_at(surf, line_x, line_y, color)
            error += slope
            if error >= 0.5:
                line_y += dy_sign
                error -= 1
    else:
        if start.y > end.y:
            (start.x, end.x) = (end.x, start.x)
            (start.y, end.y) = (end.y, start.y)
        line_x = start.x
        slope = 1 / slope
        dx_sign = 1 if start.x < end.x else -1
        for line_y in range(start.y, end.y + 1):
            set_at(surf, line_x, line_y, color)
            error += slope
            if error >= 0.5:
                line_x += dx_sign
                error -= 1

def _draw_aaline(surf, color, start, end, blend):
    if False:
        i = 10
        return i + 15
    'draw an anti-aliased line.\n\n    The algorithm yields identical results with _draw_line for horizontal,\n    vertical or diagonal lines, and results changes smoothly when changing\n    any of the endpoint coordinates.\n\n    Note that this yields strange results for very short lines, eg\n    a line from (0, 0) to (0, 1) will draw 2 pixels, and a line from\n    (0, 0) to (0, 1.1) will blend 10 % on the pixel (0, 2).\n    '
    d_x = end.x - start.x
    d_y = end.y - start.y
    if d_x == 0 and d_y == 0:
        set_at(surf, int(start.x), int(start.y), color)
        return
    if start.x > end.x or start.y > end.y:
        (start.x, end.x) = (end.x, start.x)
        (start.y, end.y) = (end.y, start.y)
        d_x = -d_x
        d_y = -d_y
    if abs(d_x) >= abs(d_y):
        slope = d_y / d_x

        def draw_two_pixel(in_x, float_y, factor):
            if False:
                return 10
            flr_y = floor(float_y)
            draw_pixel(surf, (in_x, flr_y), color, factor * inv_frac(float_y), blend)
            draw_pixel(surf, (in_x, flr_y + 1), color, factor * frac(float_y), blend)
        _draw_aaline_dx(d_x, slope, end, start, draw_two_pixel)
    else:
        slope = d_x / d_y

        def draw_two_pixel(float_x, in_y, factor):
            if False:
                for i in range(10):
                    print('nop')
            fl_x = floor(float_x)
            draw_pixel(surf, (fl_x, in_y), color, factor * inv_frac(float_x), blend)
            draw_pixel(surf, (fl_x + 1, in_y), color, factor * frac(float_x), blend)
        _draw_aaline_dy(d_y, slope, end, start, draw_two_pixel)

def _draw_aaline_dy(d_y, slope, end, start, draw_two_pixel):
    if False:
        return 10
    g_y = ceil(start.y)
    g_x = start.x + (g_y - start.y) * slope
    if start.y < g_y:
        draw_two_pixel(g_x - slope, floor(start.y), inv_frac(start.y))
    rest = frac(end.y)
    s_y = ceil(end.y)
    if rest > 0:
        s_x = start.x + slope * (d_y + 1 - rest)
        draw_two_pixel(s_x, s_y, rest)
    else:
        s_y += 1
    for line_y in range(g_y, s_y):
        line_x = g_x + slope * (line_y - g_y)
        draw_two_pixel(line_x, line_y, 1)

def _draw_aaline_dx(d_x, slope, end, start, draw_two_pixel):
    if False:
        for i in range(10):
            print('nop')
    g_x = ceil(start.x)
    g_y = start.y + (g_x - start.x) * slope
    if start.x < g_x:
        draw_two_pixel(floor(start.x), g_y - slope, inv_frac(start.x))
    rest = frac(end.x)
    s_x = ceil(end.x)
    if rest > 0:
        s_y = start.y + slope * (d_x + 1 - rest)
        draw_two_pixel(s_x, s_y, rest)
    else:
        s_x += 1
    for line_x in range(g_x, s_x):
        line_y = g_y + slope * (line_x - g_x)
        draw_two_pixel(line_x, line_y, 1)

def _clip_and_draw_line(surf, rect, color, pts):
    if False:
        print('Hello World!')
    'clip the line into the rectangle and draw if needed.\n\n    Returns true if anything has been drawn, else false.'
    if not clip_line(pts, BoundingBox(rect.x, rect.y, rect.x + rect.w - 1, rect.y + rect.h - 1)):
        return 0
    if pts[1] == pts[3]:
        _drawhorzline(surf, color, pts[0], pts[1], pts[2])
    elif pts[0] == pts[2]:
        _drawvertline(surf, color, pts[0], pts[1], pts[3])
    else:
        _draw_line(surf, color, Point(pts[0], pts[1]), Point(pts[2], pts[3]))
    return 1

def _clip_and_draw_line_width(surf, rect, color, line, width):
    if False:
        return 10
    yinc = xinc = 0
    if abs(line[0] - line[2]) > abs(line[1] - line[3]):
        yinc = 1
    else:
        xinc = 1
    newpts = line[:]
    if _clip_and_draw_line(surf, rect, color, newpts):
        anydrawn = 1
        frame = newpts[:]
    else:
        anydrawn = 0
        frame = [10000, 10000, -10000, -10000]
    for loop in range(1, width // 2 + 1):
        newpts[0] = line[0] + xinc * loop
        newpts[1] = line[1] + yinc * loop
        newpts[2] = line[2] + xinc * loop
        newpts[3] = line[3] + yinc * loop
        if _clip_and_draw_line(surf, rect, color, newpts):
            anydrawn = 1
            frame[0] = min(newpts[0], frame[0])
            frame[1] = min(newpts[1], frame[1])
            frame[2] = max(newpts[2], frame[2])
            frame[3] = max(newpts[3], frame[3])
        if loop * 2 < width:
            newpts[0] = line[0] - xinc * loop
            newpts[1] = line[1] - yinc * loop
            newpts[2] = line[2] - xinc * loop
            newpts[3] = line[3] - yinc * loop
            if _clip_and_draw_line(surf, rect, color, newpts):
                anydrawn = 1
                frame[0] = min(newpts[0], frame[0])
                frame[1] = min(newpts[1], frame[1])
                frame[2] = max(newpts[2], frame[2])
                frame[3] = max(newpts[3], frame[3])
    return anydrawn

def _clip_and_draw_aaline(surf, rect, color, line, blend):
    if False:
        while True:
            i = 10
    'draw anti-aliased line between two endpoints.'
    if not clip_line(line, BoundingBox(rect.x - 1, rect.y - 1, rect.x + rect.w, rect.y + rect.h), use_float=True):
        return
    _draw_aaline(surf, color, Point(line[0], line[1]), Point(line[2], line[3]), blend)
    return

def draw_aaline(surf, color, from_point, to_point, blend=True):
    if False:
        return 10
    'draw anti-aliased line between two endpoints.'
    line = [from_point[0], from_point[1], to_point[0], to_point[1]]
    return _clip_and_draw_aaline(surf, surf.get_clip(), color, line, blend)

def draw_line(surf, color, from_point, to_point, width=1):
    if False:
        return 10
    'draw anti-aliased line between two endpoints.'
    line = [from_point[0], from_point[1], to_point[0], to_point[1]]
    return _clip_and_draw_line_width(surf, surf.get_clip(), color, line, width)

def _multi_lines(surf, color, closed, points, width=1, blend=False, aaline=False):
    if False:
        i = 10
        return i + 15
    'draw several lines, either anti-aliased or not.'
    if len(points) <= 2:
        raise TypeError
    line = [0] * 4
    xlist = [pt[0] for pt in points]
    ylist = [pt[1] for pt in points]
    line[0] = xlist[0]
    line[1] = ylist[0]
    b_box = BoundingBox(left=xlist[0], right=xlist[0], top=ylist[0], bottom=ylist[0])
    for (line_x, line_y) in points[1:]:
        b_box.left = min(b_box.left, line_x)
        b_box.right = max(b_box.right, line_x)
        b_box.top = min(b_box.top, line_y)
        b_box.bottom = max(b_box.bottom, line_y)
    rect = surf.get_clip()
    for loop in range(1, len(points)):
        line[0] = xlist[loop - 1]
        line[1] = ylist[loop - 1]
        line[2] = xlist[loop]
        line[3] = ylist[loop]
        if aaline:
            _clip_and_draw_aaline(surf, rect, color, line, blend)
        else:
            _clip_and_draw_line_width(surf, rect, color, line, width)
    if closed:
        line[0] = xlist[len(points) - 1]
        line[1] = ylist[len(points) - 1]
        line[2] = xlist[0]
        line[3] = ylist[0]
        if aaline:
            _clip_and_draw_aaline(surf, rect, color, line, blend)
        else:
            _clip_and_draw_line_width(surf, rect, color, line, width)

def draw_lines(surf, color, closed, points, width=1):
    if False:
        print('Hello World!')
    'draw several lines connected through the points.'
    return _multi_lines(surf, color, closed, points, width, aaline=False)

def draw_aalines(surf, color, closed, points, blend=True):
    if False:
        i = 10
        return i + 15
    'draw several anti-aliased lines connected through the points.'
    return _multi_lines(surf, color, closed, points, blend=blend, aaline=True)

def draw_polygon(surface, color, points, width):
    if False:
        return 10
    'Draw a polygon'
    if width:
        draw_lines(surface, color, 1, points, width)
        return
    num_points = len(points)
    point_x = [x for (x, y) in points]
    point_y = [y for (x, y) in points]
    miny = min(point_y)
    maxy = max(point_y)
    if miny == maxy:
        minx = min(point_x)
        maxx = max(point_x)
        _clip_and_draw_horizline(surface, color, minx, miny, maxx)
        return
    for y_coord in range(miny, maxy + 1):
        x_intersect = []
        for i in range(num_points):
            _draw_polygon_inner_loop(i, point_x, point_y, y_coord, x_intersect)
        x_intersect.sort()
        for i in range(0, len(x_intersect), 2):
            _clip_and_draw_horizline(surface, color, x_intersect[i], y_coord, x_intersect[i + 1])
    for i in range(num_points):
        i_prev = i - 1 if i else num_points - 1
        if miny < point_y[i] == point_y[i_prev] < maxy:
            _clip_and_draw_horizline(surface, color, point_x[i], point_y[i], point_x[i_prev])
    return

def _draw_polygon_inner_loop(index, point_x, point_y, y_coord, x_intersect):
    if False:
        while True:
            i = 10
    i_prev = index - 1 if index else len(point_x) - 1
    y_1 = point_y[i_prev]
    y_2 = point_y[index]
    if y_1 < y_2:
        x_1 = point_x[i_prev]
        x_2 = point_x[index]
    elif y_1 > y_2:
        y_2 = point_y[i_prev]
        y_1 = point_y[index]
        x_2 = point_x[i_prev]
        x_1 = point_x[index]
    else:
        return
    if y_2 > y_coord >= y_1 or (y_coord == max(point_y) and y_coord <= y_2):
        x_intersect.append((y_coord - y_1) * (x_2 - x_1) // (y_2 - y_1) + x_1)