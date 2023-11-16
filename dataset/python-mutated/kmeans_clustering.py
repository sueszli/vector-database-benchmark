"""

Object clustering with k-means algorithm

author: Atsushi Sakai (@Atsushi_twi)

"""
import math
import matplotlib.pyplot as plt
import random
MAX_LOOP = 10
DCOST_TH = 0.1
show_animation = True

def kmeans_clustering(rx, ry, nc):
    if False:
        return 10
    clusters = Clusters(rx, ry, nc)
    clusters.calc_centroid()
    pre_cost = float('inf')
    for loop in range(MAX_LOOP):
        print('loop:', loop)
        cost = clusters.update_clusters()
        clusters.calc_centroid()
        d_cost = abs(cost - pre_cost)
        if d_cost < DCOST_TH:
            break
        pre_cost = cost
    return clusters

class Clusters:

    def __init__(self, x, y, n_label):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.y = y
        self.n_data = len(self.x)
        self.n_label = n_label
        self.labels = [random.randint(0, n_label - 1) for _ in range(self.n_data)]
        self.center_x = [0.0 for _ in range(n_label)]
        self.center_y = [0.0 for _ in range(n_label)]

    def plot_cluster(self):
        if False:
            for i in range(10):
                print('nop')
        for label in set(self.labels):
            (x, y) = self._get_labeled_x_y(label)
            plt.plot(x, y, '.')

    def calc_centroid(self):
        if False:
            return 10
        for label in set(self.labels):
            (x, y) = self._get_labeled_x_y(label)
            n_data = len(x)
            self.center_x[label] = sum(x) / n_data
            self.center_y[label] = sum(y) / n_data

    def update_clusters(self):
        if False:
            return 10
        cost = 0.0
        for ip in range(self.n_data):
            px = self.x[ip]
            py = self.y[ip]
            dx = [icx - px for icx in self.center_x]
            dy = [icy - py for icy in self.center_y]
            dist_list = [math.hypot(idx, idy) for (idx, idy) in zip(dx, dy)]
            min_dist = min(dist_list)
            min_id = dist_list.index(min_dist)
            self.labels[ip] = min_id
            cost += min_dist
        return cost

    def _get_labeled_x_y(self, target_label):
        if False:
            while True:
                i = 10
        x = [self.x[i] for (i, label) in enumerate(self.labels) if label == target_label]
        y = [self.y[i] for (i, label) in enumerate(self.labels) if label == target_label]
        return (x, y)

def calc_raw_data(cx, cy, n_points, rand_d):
    if False:
        for i in range(10):
            print('nop')
    (rx, ry) = ([], [])
    for (icx, icy) in zip(cx, cy):
        for _ in range(n_points):
            rx.append(icx + rand_d * (random.random() - 0.5))
            ry.append(icy + rand_d * (random.random() - 0.5))
    return (rx, ry)

def update_positions(cx, cy):
    if False:
        i = 10
        return i + 15
    DX1 = 0.4
    DY1 = 0.5
    DX2 = -0.3
    DY2 = -0.5
    cx[0] += DX1
    cy[0] += DY1
    cx[1] += DX2
    cy[1] += DY2
    return (cx, cy)

def main():
    if False:
        return 10
    print(__file__ + ' start!!')
    cx = [0.0, 8.0]
    cy = [0.0, 8.0]
    n_points = 10
    rand_d = 3.0
    n_cluster = 2
    sim_time = 15.0
    dt = 1.0
    time = 0.0
    while time <= sim_time:
        print('Time:', time)
        time += dt
        (cx, cy) = update_positions(cx, cy)
        (raw_x, raw_y) = calc_raw_data(cx, cy, n_points, rand_d)
        clusters = kmeans_clustering(raw_x, raw_y, n_cluster)
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            clusters.plot_cluster()
            plt.plot(cx, cy, 'or')
            plt.xlim(-2.0, 10.0)
            plt.ylim(-2.0, 10.0)
            plt.pause(dt)
    print('Done')
if __name__ == '__main__':
    main()