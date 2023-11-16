import numpy as np
import os
import time
import copy
from scipy.optimize import curve_fit
import consts as c

class TrackletBox(object):

    def __init__(self):
        if False:
            return 10
        self.tracklets = []

    def add_tracklet(self, tracklet):
        if False:
            while True:
                i = 10
        self.tracklets.append(tracklet)

    def validate_tracklets(self):
        if False:
            i = 10
            return i + 15
        for tracklet in self.tracklets:
            score_tok = tracklet.score / tracklet.length
            if score_tok < c.SCORE_TOK_THRESH:
                tracklet.is_valid = False

    def split_tracklets(self):
        if False:
            i = 10
            return i + 15
        new_tracklets = []
        for t in self.tracklets:
            acc = []
            vel = []
            if t.is_valid:
                for (i, tok) in enumerate(t.tokens):
                    if i == 0:
                        vel.append(0 * tok.coords)
                    else:
                        vel.append(t.tokens[i].coords - t.tokens[i - 1].coords)
                for (j, v) in enumerate(vel):
                    if j < 3:
                        acc.append(0)
                    elif vel[j][c.Z_3D] > 0 and vel[j - 1][c.Z_3D] < 0 and (vel[j - 2][c.Z_3D] < 0) and (vel[j - 3][c.Z_3D] < 0):
                        acc.append(1)
                    else:
                        acc.append(-1)
                split_start_f = 0
                for (k, a) in enumerate(acc):
                    if k < 2 or k >= len(acc) - 1:
                        pass
                    elif acc[k] > 0 and acc[k - 1] <= 0 and (acc[k + 1] <= 0):
                        new_track = Tracklet(split_start_f, tokens=t.tokens[split_start_f:k], score=self.tok_score_sum(t.tokens[split_start_f:k]), length=len(t.tokens[split_start_f:k]))
                        t.is_valid = False
                        self.tracklets.append(new_track)
                        split_start_f = k

    def merge_tracklets(self):
        if False:
            for i in range(10):
                print('nop')
        for t1 in self.tracklets:
            hiscore = t1.score
            for t2 in self.tracklets:
                if t1 is not t2:
                    if t1.start_frame == t2.start_frame:
                        if t2.score > hiscore:
                            hiscore = t2.score
                            t1.is_valid = False
                        else:
                            t2.is_valid = False
        for t1 in self.tracklets:
            if not t1.is_valid:
                continue
            for t2 in self.tracklets:
                if t1 is not t2 and t1.is_valid and t2.is_valid:
                    first = None
                    second = None
                    if t2.start_frame > t1.start_frame and t2.start_frame <= t1.start_frame + t1.length:
                        first = t1
                        second = t2
                    elif t1.start_frame > t2.start_frame and t1.start_frame <= t2.start_frame + t2.length:
                        first = t2
                        second = t1
                    if first is not None and second is not None:
                        contained = None
                        if second.start_frame + second.length < first.start_frame + first.length:
                            contained = True
                        else:
                            contained = False
                        if contained:
                            pass
                        else:
                            shared_tracklets = []
                            for token1 in reversed(first.tokens):
                                cons = False
                                cons_count = 0
                                for token2 in second.tokens:
                                    sim = token1.calc_similarity(token2)
                                    if sim < c.TOKEN_SIM_THRESH:
                                        cons = True
                                        cons_count += 1
                                        first_index = first.tokens.index(token1)
                                        second_index = second.tokens.index(token2)
                                    elif cons is True:
                                        shared_tracklets.append([first_index, second_index, cons_count])
                                        break
                            if shared_tracklets != []:
                                shared_track = sorted(shared_tracklets, key=lambda x: x[2], reverse=True)[0]
                                first.tokens = first.tokens[0:shared_track[0] + 1]
                                first.length = len(first.tokens)
                                for tok in second.tokens[shared_track[1]:]:
                                    first.add_token(tok)
                                second.is_valid = False
        for t1 in self.tracklets:
            if not t1.is_valid:
                continue
            for t2 in self.tracklets:
                if not t2.is_valid:
                    continue
                first = None
                second = None
                if t1.start_frame + t1.length < t2.start_frame:
                    first = t1
                    second = t2
                elif t2.start_frame + t2.length < t1.start_frame:
                    first = t2
                    second = t1
                if first is not None and second is not None:
                    if first.length > 3 and second.length > 3:
                        first_extrapolation_points = []
                        second_extrapolation_points = []
                        for i in range(3):
                            first_extrapolation_points.append(first.tokens[i - 3].coords)
                            second_extrapolation_points.append(second.tokens[2 - i].coords)
                        for i in range(c.EXTRAPOLATE_N):
                            first_extrapolation_points.append(make_est(first_extrapolation_points[-3], first_extrapolation_points[-2], first_extrapolation_points[-1]))
                            second_extrapolation_points.append(make_est(second_extrapolation_points[-3], second_extrapolation_points[-2], second_extrapolation_points[-1]))
                        first_extrapolation_points = first_extrapolation_points[-c.EXTRAPOLATE_N:]
                        second_extrapolation_points = second_extrapolation_points[-c.EXTRAPOLATE_N:]
                        best_match = c.TOKEN_SIM_THRESH
                        best_f_p = None
                        best_s_p = None
                        for (i, f_p) in enumerate(first_extrapolation_points):
                            for (j, s_p) in enumerate(second_extrapolation_points):
                                sim = calc_dist(f_p - s_p)
                                if sim < c.TOKEN_SIM_THRESH:
                                    best_match = sim
                                    best_f_p = i
                                    best_s_p = j
                                    break
                            if best_f_p is not None:
                                break
                        if best_f_p is not None and best_s_p is not None:
                            new_first_points = first_extrapolation_points[:i]
                            new_second_points = second_extrapolation_points[:j]
                            for first_point in new_first_points:
                                first.add_token(Token(first.tokens[-1].f + 1, first_point, score=1))
                            for second_point in reversed(new_second_points):
                                first.add_token(Token(first.tokens[-1].f + 1, second_point, score=1))
                            for tok in second.tokens:
                                first.add_token(tok)
                            second.is_valid = False

    def tok_score_sum(self, tokens):
        if False:
            while True:
                i = 10
        score = 0
        for tok in tokens:
            score += tok.score
        return score

def make_est(c1, c2, c3):
    if False:
        print('Hello World!')
    a3 = (c3 - c2 - (c2 - c1)) / c.dT ** 2
    v3 = (c3 - c2) / c.dT + a3 * c.dT
    c4_e = c3 + v3 * c.dT + a3 * c.dT ** 2 / 2
    return c4_e

class Tracklet(object):

    def __init__(self, start_frame, tracklet_box=None, tokens=[], score=0, length=0):
        if False:
            return 10
        self.start_frame = start_frame
        self.tracklet_box = tracklet_box
        self.tokens = tokens
        self.score = score
        self.length = length
        self.con_est = 0
        self.is_valid = True

    def save_tracklet(self):
        if False:
            for i in range(10):
                print('nop')
        if self.score < c.TRACKLET_SCORE_THRESH:
            return
        if self.tracklet_box is not None:
            self.tracklet_box.add_tracklet(Tracklet(start_frame=copy.deepcopy(self.start_frame), tokens=copy.deepcopy(self.tokens), score=copy.deepcopy(self.score), length=copy.deepcopy(self.length)))

    def add_token(self, token):
        if False:
            for i in range(10):
                print('nop')
        self.tokens.append(token)
        self.score += token.score
        self.length += 1

    def insert_token(self, token, index):
        if False:
            for i in range(10):
                print('nop')
        if index < len(self.tokens):
            self.tokens.insert(index, token)
            self.length += 1
            self.score += token.score

    def del_token(self):
        if False:
            print('Hello World!')
        self.score -= self.tokens[-1].score
        self.length -= 1
        del self.tokens[-1]

    def est_next(self):
        if False:
            for i in range(10):
                print('nop')
        if self.length >= 3:
            est = make_est(self.tokens[-3].coords, self.tokens[-2].coords, self.tokens[-1].coords)
            return est
        else:
            return None

    def add_est(self, token):
        if False:
            return 10
        if self.con_est < c.MAX_EST:
            self.add_token(token)
            self.con_est += 1
            return True
        else:
            self.con_est = 0
            return False

class Token(object):

    def __init__(self, f, coords, score=0):
        if False:
            for i in range(10):
                print('nop')
        self.f = f
        self.coords = coords
        self.score = score
        self.is_valid = True

    def calc_similarity(self, token):
        if False:
            i = 10
            return i + 15
        error = self.coords - token.coords
        return calc_dist(error)

def calc_dist(vect):
    if False:
        for i in range(10):
            print('nop')
    a = 0
    for el in vect[:3]:
        a += el ** 2
    return np.sqrt(a)

def calc_theta_phi(diff, r):
    if False:
        print('Hello World!')
    r_p = np.sqrt(diff[c.X_3D] ** 2 + diff[c.Y_3D] ** 2)
    theta = np.arccos(diff[c.Y_3D] / r_p)
    phi = np.arccos(r_p / r)
    return (theta, phi)

def score_node(est, candidate):
    if False:
        i = 10
        return i + 15
    diff = est - candidate
    r = calc_dist(diff)
    (theta, phi) = calc_theta_phi(diff, r)
    if r < c.dM:
        s1 = np.exp(-r / c.dM)
        s2 = np.exp(-theta / c.thetaM)
        s3 = np.exp(-phi / c.phiM)
        return s1 + s2 + s3
    else:
        return 0

def evaluate(candidates_3D, tracklet, f, f_max):
    if False:
        while True:
            i = 10
    if f < f_max:
        est = tracklet.est_next()
        valid_cand = False
        if candidates_3D[f] != []:
            valid_cand = False
            for (i, cand) in enumerate(candidates_3D[f]):
                c4 = cand[c.C_CAND]
                candidates_3D[f][i][c.C_INIT] = True
                score = score_node(est, c4)
                if score > c.TOKEN_SCORE_THRESH:
                    valid_cand = True
                    tracklet.add_token(Token(f, c4, score))
                    evaluate(candidates_3D, tracklet, f + 1, f_max)
                    tracklet.del_token()
        if valid_cand is False:
            if tracklet.add_est(Token(f, est)):
                evaluate(candidates_3D, tracklet, f + 1, f_max)
                tracklet.del_token()
            else:
                tracklet.save_tracklet()
    else:
        tracklet.save_tracklet()

def check_init_toks(c1, c2, c3):
    if False:
        for i in range(10):
            print('nop')
    d1 = calc_dist(c2 - c1)
    d2 = calc_dist(c3 - c2)
    if d1 < c.dM and d2 < c.dM and (d1 > 0) and (d2 > 0):
        return True
    else:
        return False

def get_tracklets(candidates_3D):
    if False:
        for i in range(10):
            print('nop')
    for (f, frame) in enumerate(candidates_3D):
        for (i, candidate) in enumerate(frame):
            candidates_3D[f][i] = [False, np.array(candidate)]
    frame_num = len(candidates_3D)
    tracklet_box = TrackletBox()
    init_set = False
    (c1, c2, c3, c4) = ([], [], [], [])
    for f in range(3, frame_num):
        if init_set is False:
            if c1 == [] or c2 == [] or c3 == []:
                continue
            else:
                init_set = True
        tracklet = Tracklet(f, tracklet_box)
        for c1_c in c1:
            if c1_c[c.C_INIT] is True:
                continue
            tracklet.add_token(Token(f - 3, c1_c[c.C_CAND], score=1))
            for c2_c in c2:
                if c2_c[c.C_INIT] is True:
                    continue
                tracklet.add_token(Token(f - 2, c2_c[c.C_CAND], score=1))
                for c3_c in c3:
                    if c3_c[c.C_INIT] is True:
                        continue
                    tracklet.add_token(Token(f - 1, c3_c[c.C_CAND], score=1))
                    c1_c[c.C_INIT] = True
                    c2_c[c.C_INIT] = True
                    c3_c[c.C_INIT] = True
                    if check_init_toks(c1_c[c.C_CAND], c2_c[c.C_CAND], c3_c[c.C_CAND]):
                        evaluate(candidates_3D, tracklet, f, f_max=frame_num)
                    tracklet.del_token()
                tracklet.del_token()
            tracklet.del_token()
        init_set = False
        (c1, c2, c3, c4) = ([], [], [], [])
    tracklet_box.merge_tracklets()
    tracklet_box.validate_tracklets()
    tracklet_box.split_tracklets()
    return tracklet_box

def find_best_tracklet(tracklet_box):
    if False:
        return 10
    (best_score, best_tracklet) = (0, None)
    for t in tracklet_box.tracklets:
        if t.is_valid:
            if t.score > best_score:
                best_score = t.score
                best_tracklet = t
    return best_tracklet

def curve_func(t, a, b, c, d):
    if False:
        for i in range(10):
            print('nop')
    return a + b * t + c * t ** 2 + d * t ** 3

def d1_curve_func(t, a, b, c, d):
    if False:
        print('Hello World!')
    return b + 2 * c * t + 3 * d * t ** 2

def analyse_tracklets(candidates_3D):
    if False:
        while True:
            i = 10
    tracklet_box = get_tracklets(candidates_3D)
    best_tracklet = find_best_tracklet(tracklet_box)
    x_points = []
    y_points = []
    z_points = []
    if best_tracklet is None:
        return None
    for (i, tok) in enumerate(best_tracklet.tokens):
        x_points.append(tok.coords[c.X_3D])
        y_points.append(tok.coords[c.Y_3D])
        z_points.append(tok.coords[c.Z_3D])
    t = np.linspace(best_tracklet.start_frame * 1 / 90, (best_tracklet.start_frame + best_tracklet.length) * 1 / 90, best_tracklet.length)
    (x_params, covmatrix) = curve_fit(curve_func, t, x_points)
    (y_params, covmatrix) = curve_fit(curve_func, t, y_points)
    (z_params, covmatrix) = curve_fit(curve_func, t, z_points)
    return (x_params, y_params, z_params, x_points, y_points, z_points)
if __name__ == '__main__':
    RESOLUTION = (640, 480)
    (w, h) = RESOLUTION
    ROOT_P = 'D:\\documents\\local uni\\FYP\\code'
    os.chdir(ROOT_P + '\\' + 'img\\simulation_tests')
    candidates_3D = np.load('candidates_3D.npy', allow_pickle=True)
    tracklet_box = get_tracklets(candidates_3D)
    best_tracklet = find_best_tracklet(tracklet_box)
    x_points = []
    y_points = []
    z_points = []
    if best_tracklet is None:
        quit()
    for (i, tok) in enumerate(best_tracklet.tokens):
        x_points.append(tok.coords[c.X_3D])
        y_points.append(tok.coords[c.Y_3D])
        z_points.append(tok.coords[c.Z_3D])
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_xlim(-11 / 2, 11 / 2)
    ax.set_ylim(0, 24)
    ax.set_zlim(0, 3)
    ax.scatter(xs=x_points, ys=y_points, zs=z_points, c=np.arange(len(x_points)), cmap='winter')
    t = np.linspace(best_tracklet.start_frame * 1 / 90, (best_tracklet.start_frame + best_tracklet.length) * 1 / 90, best_tracklet.length)
    (x_params, covmatrix) = curve_fit(curve_func, t, x_points)
    (y_params, covmatrix) = curve_fit(curve_func, t, y_points)
    (z_params, covmatrix) = curve_fit(curve_func, t, z_points)
    t = np.linspace(0, 2, 1000)
    x_est = curve_func(t, *x_params)
    y_est = curve_func(t, *y_params)
    z_est = curve_func(t, *z_params)
    xd1_est = d1_curve_func(t, *x_params)
    yd1_est = d1_curve_func(t, *y_params)
    zd1_est = d1_curve_func(t, *z_params)
    bounce_pos = len(z_est[z_est > 0]) - 1
    x_vel = xd1_est[bounce_pos]
    y_vel = yd1_est[bounce_pos]
    z_vel = zd1_est[bounce_pos]
    print(x_vel, y_vel, z_vel)
    print(f'velocity: {np.sqrt(x_vel ** 2 + y_vel ** 2 + z_vel ** 2):2.2f} m/s')
    print(f'bounce_loc: {x_est[bounce_pos]:0.2f}, {y_est[bounce_pos]:0.2f},{z_est[bounce_pos]:0.2f}')
    ax.plot3D(x_est, y_est, z_est)