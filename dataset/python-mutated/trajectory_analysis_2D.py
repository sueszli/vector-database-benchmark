import numpy as np
import os
import time
import copy
import graph_utils as gr
ROOT_P = 'D:\\documents\\local uni\\FYP\\code'
FRAME_NUM = 0
FRAME_DATA = 1
CAND_W = 0
CAND_H = 1
CAND_S = 2
CAND_X = 3
CAND_Y = 4
CAND_INIT = 0
CAND_DATA = 1
COORD_X = 0
COORD_Y = 1
VM = 150
FRAMERATE = 90
dT = 1 / FRAMERATE
WIN_SIZE = 30
WIN_OVERLAP = 5
MAX_EST = 3
EXTRAPOLATE_N = 3
MIN_SHARED_TOKS = 3
MAX_SHARED_TOKS = 5

def kph_2_mps(kph):
    if False:
        for i in range(10):
            print('nop')
    return kph * 10 / 36
dM = 50
thetaM = np.pi
TRACKLET_SCORE_THRESH = 1
TOKEN_SIM_THRESH = dM / 2
TOKEN_SCORE_THRESH = 1
SCORE_TOK_THRESH = 1

class TrackletBox(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.tracklets = []

    def add_tracklet(self, tracklet):
        if False:
            while True:
                i = 10
        self.tracklets.append(tracklet)

    def validate_tracklets(self):
        if False:
            while True:
                i = 10
        for tracklet in self.tracklets:
            score_tok = tracklet.score / tracklet.length
            if score_tok < SCORE_TOK_THRESH:
                tracklet.is_valid = False

    def split_tracklets(self):
        if False:
            while True:
                i = 10
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
                    elif vel[j][Z] > 0 and vel[j - 1][Z] < 0 and (vel[j - 2][Z] < 0) and (vel[j - 3][Z] < 0):
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
            while True:
                i = 10
        graph = gr.create_graph(self.tracklets)
        for (t1, tracklet_1) in enumerate(self.tracklets):
            for (t2, tracklet_2) in enumerate(self.tracklets):
                if t1 == t2:
                    continue
                if tracklet_2.start_frame < tracklet_1.start_frame:
                    continue
                cons_count_max = 0
                cons_pos_max = None
                for token1 in reversed(self.tracklets[t1].tokens[-MAX_SHARED_TOKS:]):
                    cons_count = 0
                    cons_pos = 0
                    cons = False
                    for token2 in self.tracklets[t2].tokens[:MAX_SHARED_TOKS]:
                        if token1.f == token2.f:
                            sim = token1.calc_similarity(token2)
                        else:
                            continue
                        if sim < TOKEN_SIM_THRESH:
                            cons = True
                            cons_count += 1
                            cons_pos = self.tracklets[t2].tokens.index(token2)
                        else:
                            break
                    if cons == True and cons_count > cons_count_max:
                        cons_pos_max = cons_pos
                if cons_pos_max is not None:
                    graph[t1].append(t2)
                    for (t, tok) in enumerate(self.tracklets[t2].tokens):
                        if t <= cons_pos_max:
                            self.tracklets[t2].score -= self.tracklets[t2].tokens[t].score
                            self.tracklets[t2].tokens[t].score = 0
        (start_nodes, end_nodes) = gr.get_start_end_nodes(graph)
        for item in graph.items():
            print(item)
        longest_path = {}
        path_list = []
        for node_s in start_nodes:
            for (node, conn) in graph.items():
                longest_path[node] = {'score': 0, 'path': []}
            gr.get_longest_paths(self.tracklets, longest_path, graph, node_s)
            for node_e in end_nodes:
                path_list.append(longest_path[node_e])
        score = 0
        best_path = None
        for path in path_list:
            if path['score'] > score:
                score = path['score']
                best_path = path
        if best_path is not None:
            merged_track = Tracklet(start_frame=self.tracklets[best_path['path'][0]].start_frame)
            f = -1
            for t in best_path['path']:
                for tok in self.tracklets[t].tokens:
                    if tok.f > f:
                        merged_track.add_token(tok)
                        f = tok.f
            for tracklet in self.tracklets:
                tracklet.is_valid = False
            self.add_tracklet(merged_track)

    def tok_score_sum(self, tokens):
        if False:
            for i in range(10):
                print('nop')
        score = 0
        for tok in tokens:
            score += tok.score
        return score

def make_est(c1, c2, c3):
    if False:
        for i in range(10):
            print('nop')
    a3 = (c3 - c2 - (c2 - c1)) / dT ** 2
    v3 = (c3 - c2) / dT + a3 * dT
    c4_e = c3 + v3 * dT + a3 * dT ** 2 / 2
    return c4_e

class Tracklet(object):

    def __init__(self, start_frame, tracklet_box=None, tokens=[], score=0, length=0):
        if False:
            for i in range(10):
                print('nop')
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
        if self.score < TRACKLET_SCORE_THRESH:
            return
        if self.tracklet_box is not None:
            self.tracklet_box.add_tracklet(Tracklet(start_frame=copy.deepcopy(self.start_frame), tokens=copy.deepcopy(self.tokens), score=copy.deepcopy(self.score), length=copy.deepcopy(self.length)))

    def add_token(self, token):
        if False:
            print('Hello World!')
        self.tokens.append(token)
        self.score += token.score
        self.length += 1

    def insert_token(self, token, index):
        if False:
            while True:
                i = 10
        if index < len(self.tokens):
            self.tokens.insert(index, token)
            self.length += 1
            self.score += token.score

    def del_token(self):
        if False:
            return 10
        self.score -= self.tokens[-1].score
        self.length -= 1
        del self.tokens[-1]

    def est_next(self):
        if False:
            i = 10
            return i + 15
        if self.length >= 3:
            est = make_est(self.tokens[-3].data, self.tokens[-2].data, self.tokens[-1].data)
            return est
        else:
            return None

    def add_est(self, token):
        if False:
            i = 10
            return i + 15
        if self.con_est < MAX_EST:
            self.add_token(token)
            self.con_est += 1
            return True
        else:
            self.con_est = 0
            return False

class Token(object):

    def __init__(self, f, data, score=0):
        if False:
            while True:
                i = 10
        self.f = f
        self.data = data
        self.score = score
        self.is_valid = True

    def calc_similarity(self, token):
        if False:
            while True:
                i = 10
        error = self.data[CAND_X:] - token.data[CAND_X:]
        return calc_dist(error)

def calc_dist(vect):
    if False:
        return 10
    a = 0
    for el in vect:
        a += el ** 2
    return np.sqrt(a)

def calc_theta(diff, r):
    if False:
        while True:
            i = 10
    r_p = np.sqrt(diff[COORD_X] ** 2 + diff[COORD_Y] ** 2)
    theta = np.arccos(diff[COORD_Y] / r_p)
    return theta

def score_node(est, candidate):
    if False:
        return 10
    diff = est[CAND_X:] - candidate[CAND_X:]
    r = calc_dist(diff)
    theta = calc_theta(diff, r)
    if r < dM:
        s1 = np.exp(-r / dM)
        s2 = np.exp(-theta / thetaM)
        return s1 + s2
    else:
        return 0

def evaluate(candidates, tracklet, f, f_max):
    if False:
        while True:
            i = 10
    if f < f_max:
        est = tracklet.est_next()
        valid_cand = False
        if candidates[f] != []:
            valid_cand = False
            for (i, cand) in enumerate(candidates[f]):
                c4 = cand[CAND_DATA]
                cand[CAND_INIT] = True
                score = score_node(est, c4)
                if score > TOKEN_SCORE_THRESH:
                    valid_cand = True
                    if f_max - WIN_SIZE <= f:
                        tracklet.add_token(Token(f, c4, score))
                    else:
                        tracklet.add_token(Token(f, c4, 0))
                    evaluate(candidates, tracklet, f + 1, f_max)
                    tracklet.del_token()
        if valid_cand is False:
            if tracklet.add_est(Token(f, est)):
                evaluate(candidates, tracklet, f + 1, f_max)
                tracklet.del_token()
            else:
                tracklet.save_tracklet()
    else:
        tracklet.save_tracklet()

def check_init_toks(c1, c2, c3):
    if False:
        print('Hello World!')
    d1 = calc_dist(c2 - c1)
    d2 = calc_dist(c3 - c2)
    if d1 < dM and d2 < dM and (d1 > 0) and (d2 > 0):
        return True
    else:
        return False

def get_tracklets(candidates):
    if False:
        return 10
    num_frames = len(candidates)
    tracklet_box = TrackletBox()
    for f in range(num_frames):
        win_start = 0
        win_end = 0
        if f == 0:
            win_start = 0
            win_end = win_start + WIN_SIZE
        elif f % WIN_SIZE == 0 and f != 0:
            win_start = f - WIN_OVERLAP
            win_end = f + WIN_SIZE
            if win_end > num_frames:
                win_end = num_frames
            for frame in candidates:
                for cand in frame:
                    cand[CAND_INIT] = False
        else:
            continue
        init_set = False
        (c1, c2, c3) = ([], [], [])
        for cur_frame in range(win_start + 3, win_end):
            if init_set is False:
                c1 = candidates[cur_frame - 3]
                c2 = candidates[cur_frame - 2]
                c3 = candidates[cur_frame - 1]
                if len(c1) == 0 or len(c2) == 0 or len(c3) == 0:
                    continue
                else:
                    init_set = True
            if init_set:
                tracklet = Tracklet(cur_frame - 3, tracklet_box)
                for c1_c in c1:
                    if c1_c[CAND_INIT] is True:
                        continue
                    tracklet.add_token(Token(cur_frame - 3, c1_c[CAND_DATA], score=0))
                    for c2_c in c2:
                        if c2_c[CAND_INIT] is True:
                            continue
                        tracklet.add_token(Token(cur_frame - 2, c2_c[CAND_DATA], score=0))
                        for c3_c in c3:
                            if c3_c[CAND_INIT] is True:
                                continue
                            tracklet.add_token(Token(cur_frame - 1, c3_c[CAND_DATA], score=0))
                            if check_init_toks(c1_c[CAND_DATA][CAND_X:], c2_c[CAND_DATA][CAND_X:], c3_c[CAND_DATA][CAND_X:]):
                                c1_c[CAND_INIT] = True
                                c2_c[CAND_INIT] = True
                                c3_c[CAND_INIT] = True
                                evaluate(candidates, tracklet, cur_frame, win_end)
                            tracklet.del_token()
                        tracklet.del_token()
                    tracklet.del_token()
                init_set = False
                (c1, c2, c3) = ([], [], [])
    return tracklet_box

def find_best_tracklet(tracklet_box):
    if False:
        for i in range(10):
            print('nop')
    (best_score, best_tracklet) = (0, None)
    for t in tracklet_box.tracklets:
        if t.is_valid:
            if t.score > best_score:
                best_score = t.score
                best_tracklet = t
    return best_tracklet

def curve_func(t, a, b, c, d):
    if False:
        i = 10
        return i + 15
    return a + b * t + c * t ** 2 + d * t ** 3

def d1_curve_func(t, a, b, c, d):
    if False:
        print('Hello World!')
    return b + 2 * c * t + 3 * d * t ** 2
if __name__ == '__main__':
    RESOLUTION = (640, 480)
    (w, h) = RESOLUTION
    os.chdir(ROOT_P + '\\' + 'img\\inside_tests_2')
    left_candidates_l = np.load('left_ball_candidates.npy', allow_pickle=True)
    right_candidates_l = np.load('right_ball_candidates.npy', allow_pickle=True)
    left_candidates = []
    right_candidates = []
    for f in range(int(left_candidates_l[-1][FRAME_NUM]) + 1):
        if f == int(left_candidates_l[f][FRAME_NUM]):
            candidates = []
            for cand in left_candidates_l[f][FRAME_DATA]:
                candidates.append([False, cand])
            left_candidates.append(candidates)
        else:
            left_candidates.append([])
    for f in range(int(right_candidates_l[-1][FRAME_NUM]) + 1):
        if f == int(right_candidates_l[f][FRAME_NUM]):
            candidates = []
            for cand in right_candidates_l[f][FRAME_DATA]:
                candidates.append([False, cand])
            right_candidates.append(candidates)
        else:
            right_candidates.append([])
    min_len = min(len(left_candidates), len(right_candidates))
    left_candidates = left_candidates[:min_len]
    right_candidates = right_candidates[:min_len]
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.set_zlabel('frame')
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_zlim(min_len, 0)
    left_tracklets = get_tracklets(left_candidates)
    left_tracklets.merge_tracklets()
    for t in left_tracklets.tracklets:
        if t.is_valid:
            print(t.start_frame, t.start_frame + t.length)
            for tok in t.tokens:
                ax.scatter(xs=tok.data[CAND_X], ys=tok.data[CAND_Y], zs=tok.f)
    plt.show()