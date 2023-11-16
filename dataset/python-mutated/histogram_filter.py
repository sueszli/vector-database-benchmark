"""

Histogram Filter 2D localization example


In this simulation, x,y are unknown, yaw is known.

Initial position is not needed.

author: Atsushi Sakai (@Atsushi_twi)

"""
import copy
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
EXTEND_AREA = 10.0
SIM_TIME = 50.0
DT = 0.1
MAX_RANGE = 10.0
MOTION_STD = 1.0
RANGE_STD = 3.0
XY_RESOLUTION = 0.5
MIN_X = -15.0
MIN_Y = -5.0
MAX_X = 15.0
MAX_Y = 25.0
NOISE_RANGE = 2.0
NOISE_SPEED = 0.5
show_animation = True

class GridMap:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.data = None
        self.xy_resolution = None
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_w = None
        self.y_w = None
        self.dx = 0.0
        self.dy = 0.0

def histogram_filter_localization(grid_map, u, z, yaw):
    if False:
        print('Hello World!')
    grid_map = motion_update(grid_map, u, yaw)
    grid_map = observation_update(grid_map, z, RANGE_STD)
    return grid_map

def calc_gaussian_observation_pdf(grid_map, z, iz, ix, iy, std):
    if False:
        i = 10
        return i + 15
    x = ix * grid_map.xy_resolution + grid_map.min_x
    y = iy * grid_map.xy_resolution + grid_map.min_y
    d = math.hypot(x - z[iz, 1], y - z[iz, 2])
    pdf = norm.pdf(d - z[iz, 0], 0.0, std)
    return pdf

def observation_update(grid_map, z, std):
    if False:
        print('Hello World!')
    for iz in range(z.shape[0]):
        for ix in range(grid_map.x_w):
            for iy in range(grid_map.y_w):
                grid_map.data[ix][iy] *= calc_gaussian_observation_pdf(grid_map, z, iz, ix, iy, std)
    grid_map = normalize_probability(grid_map)
    return grid_map

def calc_control_input():
    if False:
        return 10
    v = 1.0
    yaw_rate = 0.1
    u = np.array([v, yaw_rate]).reshape(2, 1)
    return u

def motion_model(x, u):
    if False:
        for i in range(10):
            print('nop')
    F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])
    B = np.array([[DT * math.cos(x[2, 0]), 0], [DT * math.sin(x[2, 0]), 0], [0.0, DT], [1.0, 0.0]])
    x = F @ x + B @ u
    return x

def draw_heat_map(data, mx, my):
    if False:
        for i in range(10):
            print('nop')
    max_value = max([max(i_data) for i_data in data])
    plt.grid(False)
    plt.pcolor(mx, my, data, vmax=max_value, cmap=mpl.colormaps['Blues'])
    plt.axis('equal')

def observation(xTrue, u, RFID):
    if False:
        print('Hello World!')
    xTrue = motion_model(xTrue, u)
    z = np.zeros((0, 3))
    for i in range(len(RFID[:, 0])):
        dx = xTrue[0, 0] - RFID[i, 0]
        dy = xTrue[1, 0] - RFID[i, 1]
        d = math.hypot(dx, dy)
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * NOISE_RANGE
            zi = np.array([dn, RFID[i, 0], RFID[i, 1]])
            z = np.vstack((z, zi))
    ud = u[:, :]
    ud[0] += np.random.randn() * NOISE_SPEED
    return (xTrue, z, ud)

def normalize_probability(grid_map):
    if False:
        i = 10
        return i + 15
    sump = sum([sum(i_data) for i_data in grid_map.data])
    for ix in range(grid_map.x_w):
        for iy in range(grid_map.y_w):
            grid_map.data[ix][iy] /= sump
    return grid_map

def init_grid_map(xy_resolution, min_x, min_y, max_x, max_y):
    if False:
        while True:
            i = 10
    grid_map = GridMap()
    grid_map.xy_resolution = xy_resolution
    grid_map.min_x = min_x
    grid_map.min_y = min_y
    grid_map.max_x = max_x
    grid_map.max_y = max_y
    grid_map.x_w = int(round((grid_map.max_x - grid_map.min_x) / grid_map.xy_resolution))
    grid_map.y_w = int(round((grid_map.max_y - grid_map.min_y) / grid_map.xy_resolution))
    grid_map.data = [[1.0 for _ in range(grid_map.y_w)] for _ in range(grid_map.x_w)]
    grid_map = normalize_probability(grid_map)
    return grid_map

def map_shift(grid_map, x_shift, y_shift):
    if False:
        i = 10
        return i + 15
    tmp_grid_map = copy.deepcopy(grid_map.data)
    for ix in range(grid_map.x_w):
        for iy in range(grid_map.y_w):
            nix = ix + x_shift
            niy = iy + y_shift
            if 0 <= nix < grid_map.x_w and 0 <= niy < grid_map.y_w:
                grid_map.data[ix + x_shift][iy + y_shift] = tmp_grid_map[ix][iy]
    return grid_map

def motion_update(grid_map, u, yaw):
    if False:
        return 10
    grid_map.dx += DT * math.cos(yaw) * u[0]
    grid_map.dy += DT * math.sin(yaw) * u[0]
    x_shift = grid_map.dx // grid_map.xy_resolution
    y_shift = grid_map.dy // grid_map.xy_resolution
    if abs(x_shift) >= 1.0 or abs(y_shift) >= 1.0:
        grid_map = map_shift(grid_map, int(x_shift[0]), int(y_shift[0]))
        grid_map.dx -= x_shift * grid_map.xy_resolution
        grid_map.dy -= y_shift * grid_map.xy_resolution
    grid_map.data = gaussian_filter(grid_map.data, sigma=MOTION_STD)
    return grid_map

def calc_grid_index(grid_map):
    if False:
        while True:
            i = 10
    (mx, my) = np.mgrid[slice(grid_map.min_x - grid_map.xy_resolution / 2.0, grid_map.max_x + grid_map.xy_resolution / 2.0, grid_map.xy_resolution), slice(grid_map.min_y - grid_map.xy_resolution / 2.0, grid_map.max_y + grid_map.xy_resolution / 2.0, grid_map.xy_resolution)]
    return (mx, my)

def main():
    if False:
        while True:
            i = 10
    print(__file__ + ' start!!')
    RF_ID = np.array([[10.0, 0.0], [10.0, 10.0], [0.0, 15.0], [-5.0, 20.0]])
    time = 0.0
    xTrue = np.zeros((4, 1))
    grid_map = init_grid_map(XY_RESOLUTION, MIN_X, MIN_Y, MAX_X, MAX_Y)
    (mx, my) = calc_grid_index(grid_map)
    while SIM_TIME >= time:
        time += DT
        print(f'time={time:.1f}')
        u = calc_control_input()
        yaw = xTrue[2, 0]
        (xTrue, z, ud) = observation(xTrue, u, RF_ID)
        grid_map = histogram_filter_localization(grid_map, u, z, yaw)
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            draw_heat_map(grid_map.data, mx, my)
            plt.plot(xTrue[0, :], xTrue[1, :], 'xr')
            plt.plot(RF_ID[:, 0], RF_ID[:, 1], '.k')
            for i in range(z.shape[0]):
                plt.plot([xTrue[0, 0], z[i, 1]], [xTrue[1, 0], z[i, 2]], '-k')
            plt.title('Time[s]:' + str(time)[0:4])
            plt.pause(0.1)
    print('Done')
if __name__ == '__main__':
    main()