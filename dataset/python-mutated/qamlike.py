"""
This file contains constellations that are similar to QAM, but are not perfect squares.
"""
from . import digital_python
from .qam import large_ampls_to_corners_mapping

def qam32_holeinside_constellation(large_ampls_to_corners=False):
    if False:
        print('Hello World!')
    indices_and_numbers = (((0, 0), 0), ((0, 1), 3), ((0, 2), 2), ((1, 0), 1), ((1, 1), 5), ((1, 2), 7), ((2, 1), 4), ((2, 2), 6))
    points = [None] * 32
    for (indices, number) in indices_and_numbers:
        p_in_quadrant = 0.5 + indices[0] + 1j * (0.5 + indices[1])
        for quadrant in range(4):
            index = number + 8 * quadrant
            rotation = pow(1j, quadrant)
            p = p_in_quadrant * rotation
            points[index] = p
    side = 6
    width = 1
    side = 12
    width = 0.5
    pre_diff_code = []
    if not large_ampls_to_corners:
        constellation = digital_python.constellation_rect(points, pre_diff_code, 4, side, side, width, width)
    else:
        sector_values = large_ampls_to_corners_mapping(side, points, width)
        constellation = digital_python.constellation_expl_rect(points, pre_diff_code, 4, side, side, width, width, sector_values)
    return constellation