"""
QAM modulation and demodulation.
"""
from math import pi, sqrt, log
from gnuradio import gr
from .generic_mod_demod import generic_mod, generic_demod
from .generic_mod_demod import shared_mod_args, shared_demod_args
from .utils.gray_code import gray_code
from .utils import mod_codes
from . import modulation_utils
from . import digital_python as digital
_def_constellation_points = 16
_def_differential = True
_def_mod_code = mod_codes.NO_CODE

def is_power_of_four(x):
    if False:
        print('Hello World!')
    v = log(x) / log(4)
    return int(v) == v

def get_bit(x, n):
    if False:
        i = 10
        return i + 15
    " Get the n'th bit of integer x (from little end)."
    return (x & 1 << n) >> n

def get_bits(x, n, k):
    if False:
        for i in range(10):
            print('nop')
    ' Get the k bits of integer x starting at bit n(from little end).'
    v = x >> n
    return v % pow(2, k)

def make_differential_constellation(m, gray_coded):
    if False:
        i = 10
        return i + 15
    '\n    Create a constellation with m possible symbols where m must be\n    a power of 4.\n\n    Points are laid out in a square grid.\n\n    Bits referring to the quadrant are differentilly encoded,\n    remaining bits are gray coded.\n\n    '
    sqrtm = pow(m, 0.5)
    if not isinstance(m, int) or m < 4 or (not is_power_of_four(m)):
        raise ValueError('m must be a power of 4 integer.')
    k = int(log(m) / log(2.0))
    side = int(sqrtm / 2)
    if gray_coded:
        gcs = gray_code(side)
        i_gcs = dict([(v, key) for (key, v) in enumerate(gcs)])
    else:
        i_gcs = dict([(i, i) for i in range(0, side)])
    step = 1 / (side - 0.5)
    gc_to_x = [(i_gcs[gc] + 0.5) * step for gc in range(0, side)]

    def get_c(gc_x, gc_y, quad):
        if False:
            return 10
        if quad == 0:
            return complex(gc_to_x[gc_x], gc_to_x[gc_y])
        if quad == 1:
            return complex(-gc_to_x[gc_y], gc_to_x[gc_x])
        if quad == 2:
            return complex(-gc_to_x[gc_x], -gc_to_x[gc_y])
        if quad == 3:
            return complex(gc_to_x[gc_y], -gc_to_x[gc_x])
        raise Exception('Impossible!')
    const_map = []
    for i in range(m):
        y = get_bits(i, 0, (k - 2) // 2)
        x = get_bits(i, (k - 2) // 2, (k - 2) // 2)
        quad = get_bits(i, k - 2, 2)
        const_map.append(get_c(x, y, quad))
    return const_map

def make_non_differential_constellation(m, gray_coded):
    if False:
        for i in range(10):
            print('nop')
    side = int(pow(m, 0.5))
    if not isinstance(m, int) or m < 4 or (not is_power_of_four(m)):
        raise ValueError('m must be a power of 4 integer.')
    k = int(log(m) / log(2.0))
    if gray_coded:
        gcs = gray_code(side)
        i_gcs = mod_codes.invert_code(gcs)
    else:
        i_gcs = list(range(0, side))
    step = 2.0 / (side - 1)
    gc_to_x = [-1 + i_gcs[gc] * step for gc in range(0, side)]
    const_map = []
    for i in range(m):
        y = gc_to_x[get_bits(i, 0, k // 2)]
        x = gc_to_x[get_bits(i, k // 2, k // 2)]
        const_map.append(complex(x, y))
    return const_map

def qam_constellation(constellation_points=_def_constellation_points, differential=_def_differential, mod_code=_def_mod_code, large_ampls_to_corners=False):
    if False:
        return 10
    "\n    Creates a QAM constellation object.\n\n    If large_ampls_to_corners=True then sectors that are probably\n    occupied due to a phase offset, are not mapped to the closest\n    constellation point.  Rather we take into account the fact that a\n    phase offset is probably the problem and map them to the closest\n    corner point.  It's a bit hackish but it seems to improve\n    frequency locking.\n    "
    if mod_code == mod_codes.GRAY_CODE:
        gray_coded = True
    elif mod_code == mod_codes.NO_CODE:
        gray_coded = False
    else:
        raise ValueError('Mod code is not implemented for QAM')
    if differential:
        points = make_differential_constellation(constellation_points, gray_coded=False)
    else:
        points = make_non_differential_constellation(constellation_points, gray_coded)
    side = int(sqrt(constellation_points))
    width = 2.0 / (side - 1)
    pre_diff_code = []
    if not large_ampls_to_corners:
        constellation = digital.constellation_rect(points, pre_diff_code, 4, side, side, width, width)
    else:
        sector_values = large_ampls_to_corners_mapping(side, points, width)
        constellation = digital.constellation_expl_rect(points, pre_diff_code, 4, side, side, width, width, sector_values)
    return constellation

def find_closest_point(p, qs):
    if False:
        while True:
            i = 10
    "\n    Return in index of the closest point in 'qs' to 'p'.\n    "
    min_dist = None
    min_i = None
    for (i, q) in enumerate(qs):
        dist = abs(q - p)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_i = i
    return min_i

def large_ampls_to_corners_mapping(side, points, width):
    if False:
        return 10
    '\n    We have a grid that we use for decision making.  One additional row/column\n    is placed on each side of the grid.  Points in these additional rows/columns\n    are mapped to the corners rather than the closest constellation points.\n\n    Args:\n        side: The number of rows/columns in the grid that we use to do\n              decision making.\n        points: The list of constellation points.\n        width: The width of the rows/columns.\n\n    Returns:\n        sector_values maps the sector index to the constellation\n        point index.\n    '
    corner_indices = []
    corner_points = []
    max_mag = 0
    for (i, p) in enumerate(points):
        if abs(p) > max_mag:
            corner_indices = [i]
            corner_points = [p]
            max_mag = abs(p)
        elif abs(p) == max_mag:
            corner_indices.append(i)
            corner_points.append(p)
    if len(corner_indices) != 4:
        raise ValueError('Found {0} corner indices.  Expected 4.'.format(len(corner_indices)))
    extra_layers = 1
    side = side + extra_layers * 2
    sector_values = []
    for real_x in range(side):
        for imag_x in range(side):
            sector = real_x * side + imag_x
            c = (real_x - side / 2.0 + 0.5) * width + (imag_x - side / 2.0 + 0.5) * width * 1j
            if real_x >= extra_layers and real_x < side - extra_layers and (imag_x >= extra_layers) and (imag_x < side - extra_layers):
                index = find_closest_point(c, points)
            else:
                index = corner_indices[find_closest_point(c, corner_points)]
            sector_values.append(index)
    return sector_values
modulation_utils.add_type_1_constellation('qam', qam_constellation)