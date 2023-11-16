"""
Check if Point is Inside Polygon

Given a polygon (created by counterclockwise ordered points, more than 2 points) and a point "p", find if "p" lies inside the polygon or not.
The points lying on the border are considered inside.

Input: [(0, 0), (3, 0), (3, 2), (0, 2)], (1, 1)
Output: True
Output explanation: The polygon is a 3x2 rectangle parallel with the X axis.

=========================================
To check if a point is inside a polygon you'll need to draw a straight line (in any of the 4 directions: up, right, down, left),
and count the number of times the line intersects with polygon edges. If the number of intersections is odd then the point
is inside or lies on an edge of the polygon, otherwise the point is outside.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def check_if_point_inside_polygon(polygon, p):
    if False:
        return 10
    n = len(polygon)
    prev = polygon[-1]
    is_inside = False
    for curr in polygon:
        if intersect(prev, curr, p):
            is_inside = not is_inside
        prev = curr
    return is_inside

def intersect(a, b, p):
    if False:
        while True:
            i = 10
    if (a[1] > p[1]) != (b[1] > p[1]):
        '\n        Equation of line:\n        y = (x - x0) * ((y1 - y0) / (x1 - x0)) + y0\n        This formula is computed using the gradients (slopes, changes in the coordinates).\n        The following formula differs from the previous in that it finds X instead of Y (because Y is known).\n        '
        x_intersect = (p[1] - a[1]) * ((b[0] - a[0]) / (b[1] - a[1])) + a[0]
        return x_intersect <= p[1]
        "\n        There exists a more complicated solution. (just in case if you're trying to compare X coordinates and find an intersection)\n        Compare X coordinates, if both line X coordinates are bigger than point X then there is an intersection.\n        If both line X coordinates are bigger than point X then there is no intersection.\n        Else compute the angle between point-lineA and point-lineB (using math.atan2),\n        if the angle is smaller or equal than 180 (Pi) there is an interesection else there is no intersection.\n        "
    return False
print(check_if_point_inside_polygon([(0, 0), (3, 0), (3, 2), (0, 2)], (1, 1)))
print(check_if_point_inside_polygon([(0, 0), (3, 0), (3, 2), (0, 2)], (1, 0)))
print(check_if_point_inside_polygon([(0, 0), (3, 0), (3, 2), (0, 2)], (3, 1)))
print(check_if_point_inside_polygon([(0, 0), (3, 0), (3, 2), (0, 2)], (3, 0)))
print(check_if_point_inside_polygon([(0, 0), (3, 0), (3, 2), (0, 2)], (3, 3)))