"""

Path tracking simulation with pure pursuit steering and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)
        Guillaume Jacquenot (@Gjacquenot)

"""
import numpy as np
import math
import matplotlib.pyplot as plt
k = 0.1
Lfc = 2.0
Kp = 1.0
dt = 0.1
WB = 2.9
show_animation = True

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        if False:
            return 10
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - WB / 2 * math.cos(self.yaw)
        self.rear_y = self.y - WB / 2 * math.sin(self.yaw)

    def update(self, a, delta):
        if False:
            i = 10
            return i + 15
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - WB / 2 * math.cos(self.yaw)
        self.rear_y = self.y - WB / 2 * math.sin(self.yaw)

    def calc_distance(self, point_x, point_y):
        if False:
            for i in range(10):
                print('nop')
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)

class States:

    def __init__(self):
        if False:
            print('Hello World!')
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        if False:
            while True:
                i = 10
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)

def proportional_control(target, current):
    if False:
        while True:
            i = 10
    a = Kp * (target - current)
    return a

class TargetCourse:

    def __init__(self, cx, cy):
        if False:
            return 10
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        if False:
            for i in range(10):
                print('nop')
        if self.old_nearest_point_index is None:
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind], self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1], self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if ind + 1 < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind
        Lf = k * state.v + Lfc
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if ind + 1 >= len(self.cx):
                break
            ind += 1
        return (ind, Lf)

def pure_pursuit_steer_control(state, trajectory, pind):
    if False:
        for i in range(10):
            print('nop')
    (ind, Lf) = trajectory.search_target_index(state)
    if pind >= ind:
        ind = pind
    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1
    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw
    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)
    return (delta, ind)

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc='r', ec='k'):
    if False:
        return 10
    '\n    Plot arrow\n    '
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def main():
    if False:
        while True:
            i = 10
    cx = np.arange(0, 50, 0.5)
    cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    target_speed = 10.0 / 3.6
    T = 100.0
    state = State(x=-0.0, y=-3.0, yaw=0.0, v=0.0)
    lastIndex = len(cx) - 1
    time = 0.0
    states = States()
    states.append(time, state)
    target_course = TargetCourse(cx, cy)
    (target_ind, _) = target_course.search_target_index(state)
    while T >= time and lastIndex > target_ind:
        ai = proportional_control(target_speed, state.v)
        (di, target_ind) = pure_pursuit_steer_control(state, target_course, target_ind)
        state.update(ai, di)
        time += dt
        states.append(time, state)
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, '-r', label='course')
            plt.plot(states.x, states.y, '-b', label='trajectory')
            plt.plot(cx[target_ind], cy[target_ind], 'xg', label='target')
            plt.axis('equal')
            plt.grid(True)
            plt.title('Speed[km/h]:' + str(state.v * 3.6)[:4])
            plt.pause(0.001)
    assert lastIndex >= target_ind, 'Cannot goal'
    if show_animation:
        plt.cla()
        plt.plot(cx, cy, '.r', label='course')
        plt.plot(states.x, states.y, '-b', label='trajectory')
        plt.legend()
        plt.xlabel('x[m]')
        plt.ylabel('y[m]')
        plt.axis('equal')
        plt.grid(True)
        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v], '-r')
        plt.xlabel('Time[s]')
        plt.ylabel('Speed[km/h]')
        plt.grid(True)
        plt.show()
if __name__ == '__main__':
    print('Pure pursuit path tracking simulation start')
    main()