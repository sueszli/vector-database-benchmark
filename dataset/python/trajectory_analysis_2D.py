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


VM = 150			# max ball velocity
FRAMERATE = 90
dT = 1/FRAMERATE	# inter frame time
WIN_SIZE = 30
WIN_OVERLAP = 5
MAX_EST = 3
EXTRAPOLATE_N = 3
MIN_SHARED_TOKS = 3
MAX_SHARED_TOKS = 5

def kph_2_mps(kph):
	return kph*10/36

dM = 50			# max dist in px
thetaM = np.pi

TRACKLET_SCORE_THRESH = 1
TOKEN_SIM_THRESH = dM/2
TOKEN_SCORE_THRESH = 1
SCORE_TOK_THRESH = 1

class TrackletBox(object):
	def __init__(self):
		self.tracklets = []

	def add_tracklet(self, tracklet):
		self.tracklets.append(tracklet)

	def validate_tracklets(self):
		for tracklet in self.tracklets:
			score_tok = tracklet.score/tracklet.length
			if score_tok < SCORE_TOK_THRESH:
				tracklet.is_valid = False

	def split_tracklets(self):
		new_tracklets = []
		for t in self.tracklets:
			acc = []
			vel = []
			if t.is_valid:
				for i, tok in enumerate(t.tokens):
					if i==0:
						vel.append(0*tok.coords)
					else:
						vel.append(t.tokens[i].coords-t.tokens[i-1].coords)

				for j, v in enumerate(vel):
					if j<3:
						acc.append(0)
					else:
						if vel[j][Z] > 0 and vel[j-1][Z] < 0 and vel[j-2][Z] < 0 and vel[j-3][Z] < 0:
							acc.append(1)
						else: 
							acc.append(-1)

				split_start_f = 0
				for k, a in enumerate(acc):
					if k<2 or k>=len(acc)-1:
						pass
					else:
						if acc[k] > 0 and acc[k-1] <= 0 and acc[k+1] <=0 :
							new_track = Tracklet(split_start_f, \
							tokens = t.tokens[split_start_f:k], \
							score = self.tok_score_sum(t.tokens[split_start_f:k]), \
							length = len(t.tokens[split_start_f:k]))

							t.is_valid = False
							self.tracklets.append(new_track)
							split_start_f = k

	# def connect_tracklets(self, connections):
		

	def merge_tracklets(self):
		graph = gr.create_graph(self.tracklets)
	
		# -- Temporal overlap -- ##
		for t1, tracklet_1 in enumerate(self.tracklets):
			for t2, tracklet_2 in enumerate(self.tracklets):
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
					for t, tok in enumerate(self.tracklets[t2].tokens):
						if t<=cons_pos_max:
							self.tracklets[t2].score -= self.tracklets[t2].tokens[t].score
							self.tracklets[t2].tokens[t].score = 0

		start_nodes, end_nodes = gr.get_start_end_nodes(graph)

		for item in graph.items():
			print(item)

		longest_path = {}
		path_list = []
		for node_s in start_nodes:
			for node, conn in graph.items():
				longest_path[node] = {'score':0, 'path':[]}

			gr.get_longest_paths(self.tracklets, longest_path, graph, node_s)
			
			for node_e in end_nodes:
				path_list.append(longest_path[node_e])

		score = 0
		best_path = None
		for path in path_list:
			if path['score'] > score:
				score=path['score']
				best_path = path

		if best_path is not None:
			merged_track = Tracklet(start_frame=self.tracklets[best_path['path'][0]].start_frame)
			f=-1
			for t in best_path['path']:
				for tok in self.tracklets[t].tokens:
					if tok.f > f:
						merged_track.add_token(tok)
						f = tok.f

			for tracklet in self.tracklets:
				tracklet.is_valid = False

			self.add_tracklet(merged_track)

		# print(graph)
		# print(end_nodes)

		# for con_i, con in enumerate(connections):
		# 	# if the tracklet is an initial tracklet, not joined to another.
		# 	if con_i is not in connected:
		# 		self.connect_tracklets(connections, con_i)
		
		# for connected_t in connected_tracklets:
		# 	for t in connected_t:
		# 		print(f"s_frame: {t.start_frame}, len: {t.length}, score: {t.score}")
				# self.add_tracklet(t)
		# add all these tracklets to self.tracklets

		# tracklets intersect after extrapolation
		# for t1 in self.tracklets:
		# 	if not t1.is_valid:
		# 		continue
		# 	for t2 in self.tracklets:
		# 		if not t2.is_valid:
		# 			continue
		# 		first = None
		# 		second = None
		# 		if t1.start_frame+t1.length < t2.start_frame:
		# 			first = t1
		# 			second = t2
		# 		elif t2.start_frame+t2.length < t1.start_frame:
		# 			first = t2
		# 			second = t1


		# 		if first is not None and second is not None:
		# 			if first.length > 3 and second.length > 3:
		# 				first_extrapolation_points = []
		# 				second_extrapolation_points = []

		# 				for i in range(3):
		# 					first_extrapolation_points.append(first.tokens[i-3].data)
		# 					second_extrapolation_points.append(second.tokens[2-i].data)

		# 				for i in range(EXTRAPOLATE_N):
		# 					first_extrapolation_points.append(
		# 						make_est(	first_extrapolation_points[-3],
		# 									first_extrapolation_points[-2],
		# 									first_extrapolation_points[-1]))
							
		# 					second_extrapolation_points.append(
		# 						make_est(	second_extrapolation_points[-3],
		# 									second_extrapolation_points[-2],
		# 									second_extrapolation_points[-1]))

		# 				first_extrapolation_points = first_extrapolation_points[-EXTRAPOLATE_N:]
		# 				second_extrapolation_points = second_extrapolation_points[-EXTRAPOLATE_N:]

		# 				best_match = TOKEN_SIM_THRESH
		# 				best_f_p = None
		# 				best_s_p = None

		# 				for i, f_p in enumerate(first_extrapolation_points):
		# 					for j, s_p in enumerate(second_extrapolation_points):
		# 						sim = calc_dist(f_p[CAND_X:]-s_p[CAND_X:])
		# 						if sim < TOKEN_SIM_THRESH:
		# 							best_match = sim
		# 							best_f_p = i
		# 							best_s_p = j
		# 							break
		# 					if best_f_p is not None:
		# 						break

		# 				if best_f_p is not None and best_s_p is not None:
		# 					new_first_points = first_extrapolation_points[:i]
		# 					new_second_points = second_extrapolation_points[:j]

		# 					for first_point in new_first_points:
		# 						first.add_token(Token(first.tokens[-1].f+1,first_point,score=1))

		# 					for second_point in reversed(new_second_points):
		# 						first.add_token(Token(first.tokens[-1].f+1,second_point,score=1))

		# 					for tok in second.tokens:
		# 						first.add_token(tok)

		# 					second.is_valid = False

	def tok_score_sum(self, tokens):
		score = 0
		for tok in tokens:
			score += tok.score

		return score

def make_est(c1,c2,c3):
	a3 = ((c3-c2)-(c2-c1))/(dT**2)			# acceleration
	v3 = ((c3-c2)/(dT)) + a3*dT 			# velocity

	c4_e = c3+v3*dT+((a3*dT**2)/(2)) 		# next point estimation
	return c4_e

class Tracklet(object):
	def __init__(self, start_frame, tracklet_box=None, tokens=[], score=0,length=0):
		self.start_frame = start_frame
		self.tracklet_box = tracklet_box
		self.tokens = tokens
		self.score = score
		self.length = length
		self.con_est = 0
		self.is_valid = True

	def save_tracklet(self):
		if self.score < TRACKLET_SCORE_THRESH:
			return
		if self.tracklet_box is not None:
			self.tracklet_box.add_tracklet(Tracklet(start_frame = copy.deepcopy(self.start_frame),\
						tokens = copy.deepcopy(self.tokens), score = copy.deepcopy(self.score), \
						length = copy.deepcopy(self.length)))
			# self.tracklet_box.add_tracklet(copy.deepcopy(self))


	def add_token(self, token):
		self.tokens.append(token)
		self.score += token.score
		self.length += 1

	def insert_token(self, token, index):
		if index < len(self.tokens):
			self.tokens.insert(index, token)
			self.length+=1
			self.score+=token.score

	def del_token(self):
		self.score -= self.tokens[-1].score
		self.length -= 1
		del self.tokens[-1]

	def est_next(self):
		if self.length >= 3:
			est = make_est(	self.tokens[-3].data,\
							self.tokens[-2].data,\
							self.tokens[-1].data)
			return est
		else:
			return None

	def add_est(self, token):
		if self.con_est < MAX_EST:
			self.add_token(token)
			self.con_est += 1
			return True
		else:
			self.con_est = 0
			return False

class Token(object):
	def __init__(self, f, data, score=0):
		self.f = f 
		self.data = data
		self.score = score
		self.is_valid = True

	def calc_similarity(self, token):
		error = self.data[CAND_X:]-token.data[CAND_X:]
		return calc_dist(error)

def calc_dist(vect):
	a = 0
	for el in vect:
		a += el**2

	return np.sqrt(a)

def calc_theta(diff, r):
	r_p = np.sqrt(diff[COORD_X]**2+diff[COORD_Y]**2)

	theta = np.arccos(diff[COORD_Y]/r_p)
	return theta

def score_node(est, candidate):
	diff = est[CAND_X:]-candidate[CAND_X:]
	r = calc_dist(diff)
	theta = calc_theta(diff, r)
	
	if r<dM:
		s1 = np.exp(-r/dM)
		s2 = np.exp(-theta/thetaM)

		return s1+s2
	else:
		return 0

def evaluate(candidates, tracklet, f, f_max):
	# want to stop recursion when
	# - 3 consecutive estimates are made
	# - f == max
	if f < f_max:
		est = tracklet.est_next()

		# are there tokens to compare?
		# yes: compare the tokens and continue
		# no: add the estimate as a token and continue
		valid_cand = False
		if candidates[f] != []:
			valid_cand = False
			for i, cand in enumerate(candidates[f]):
				c4 = cand[CAND_DATA]
				cand[CAND_INIT] = True
				score = score_node(est, c4)
				if score > TOKEN_SCORE_THRESH:
					valid_cand = True
					if f_max-WIN_SIZE <= f:
						tracklet.add_token(Token(f, c4, score))
					else:
						tracklet.add_token(Token(f, c4, 0))

					evaluate(candidates, tracklet, f+1, f_max)
					tracklet.del_token()

		if valid_cand is False:
			if tracklet.add_est(Token(f, est)):
				evaluate(candidates, tracklet, f+1, f_max)
				tracklet.del_token()
			else:
				# added 3 estimates, stop recursion and save tracklet
				tracklet.save_tracklet()
	else:
		# tracklet is max length
		# save tracklet and stop recursion
		tracklet.save_tracklet()

def check_init_toks(c1,c2,c3):
	d1 = calc_dist(c2-c1)
	d2 = calc_dist(c3-c2)

	if d1<dM and d2<dM and d1>0 and d2>0:
		return True
	else:
		return False


def get_tracklets(candidates):
	num_frames = len(candidates)
	tracklet_box = TrackletBox()
	
	for f in range(num_frames):
		win_start = 0
		win_end = 0

		if f == 0:
			win_start = 0
			win_end = win_start+WIN_SIZE

		elif f % WIN_SIZE == 0 and f != 0:
			win_start = f-WIN_OVERLAP
			win_end = f+WIN_SIZE

			if win_end > num_frames:
				win_end = num_frames

			for frame in candidates:
				for cand in frame:
					cand[CAND_INIT] = False
		
		else:
			continue

		init_set = False
		c1,c2,c3 = [],[],[]

		for cur_frame in range(win_start+3, win_end):
			if init_set is False:
				c1 = candidates[cur_frame-3]
				c2 = candidates[cur_frame-2]
				c3 = candidates[cur_frame-1]

				if len(c1) == 0 or len(c2) == 0 or len(c3) ==0:
					continue
				else:
					init_set = True

			if init_set:
				tracklet = Tracklet(cur_frame-3, tracklet_box)
				for c1_c in c1:
					if c1_c[CAND_INIT] is True:	continue
					tracklet.add_token(Token(cur_frame-3, c1_c[CAND_DATA], score=0))
					for c2_c in c2:
						if c2_c[CAND_INIT] is True:	continue
						tracklet.add_token(Token(cur_frame-2, c2_c[CAND_DATA], score=0))
						for c3_c in c3:
							if c3_c[CAND_INIT] is True:	continue
							tracklet.add_token(Token(cur_frame-1, c3_c[CAND_DATA], score=0))

							if check_init_toks(c1_c[CAND_DATA][CAND_X:], c2_c[CAND_DATA][CAND_X:], c3_c[CAND_DATA][CAND_X:]):

								c1_c[CAND_INIT] = True
								c2_c[CAND_INIT] = True
								c3_c[CAND_INIT] = True

								evaluate(candidates, tracklet, cur_frame, win_end)

							tracklet.del_token()
						tracklet.del_token()
					tracklet.del_token()

				init_set = False
				c1,c2,c3 = [],[],[]

	return tracklet_box

def find_best_tracklet(tracklet_box):
	best_score, best_tracklet = 0, None
	for t in tracklet_box.tracklets:
		if t.is_valid:
			# print(f"f_start: {t.start_frame}, f_end: {t.start_frame+t.length}, score: {t.score:0.2f}, score/tok: {t.score/t.length:0.2f}")

			if t.score>best_score:
				best_score = t.score
				best_tracklet = t

	return best_tracklet

def curve_func(t,a,b,c,d):
	return a+b*t+c*t**2+d*t**3

def d1_curve_func(t,a,b,c,d):
	return b+2*c*t+3*d*t**2

if __name__ == "__main__":
	RESOLUTION = (640,480)
	w,h = RESOLUTION

	os.chdir(ROOT_P + '\\' + 'img\\inside_tests_2')

	left_candidates_l = np.load('left_ball_candidates.npy', allow_pickle=True)
	right_candidates_l = np.load('right_ball_candidates.npy', allow_pickle=True)

	left_candidates = []
	right_candidates = []

	# check all frames exist
	for f in range(int(left_candidates_l[-1][FRAME_NUM])+1):
		if f == int(left_candidates_l[f][FRAME_NUM]):
			candidates = []
			for cand in left_candidates_l[f][FRAME_DATA]:
				candidates.append([False,cand])
			left_candidates.append(candidates)

		else:
			left_candidates.append([])

	for f in range(int(right_candidates_l[-1][FRAME_NUM])+1):
		if f == int(right_candidates_l[f][FRAME_NUM]):
			candidates = []
			for cand in right_candidates_l[f][FRAME_DATA]:
				candidates.append([False,cand])
			right_candidates.append(candidates)
		else:
			right_candidates.append([])

	# trim candidates
	min_len = min(len(left_candidates), len(right_candidates))
	left_candidates = left_candidates[:min_len]
	right_candidates = right_candidates[:min_len]

	# -- Plot points -- ##
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (px)')
	ax.set_ylabel('y (px)')
	ax.set_zlabel('frame')
	ax.set_xlim(0, w)
	ax.set_ylim(0, h)
	ax.set_zlim(min_len,0)

	# for f, frame in enumerate(left_candidates):
	# 	if frame != []:
	# 		for cand in frame:
	# 			ax.scatter(xs=cand[CAND_DATA][CAND_X],ys=cand[CAND_DATA][CAND_Y],zs=f)

	# plt.show()

	# quit()

	left_tracklets = get_tracklets(left_candidates)
	left_tracklets.merge_tracklets()

	for t in left_tracklets.tracklets:
		if t.is_valid:
			print(t.start_frame, t.start_frame+t.length)
			for tok in t.tokens:
				ax.scatter(xs=tok.data[CAND_X],ys=tok.data[CAND_Y],zs=tok.f)

	plt.show()


	# for frame in candidates_3D:
	# 	for cand in frame:
	# 		ax.scatter(xs=cand[0],ys=cand[1],zs=cand[2])
	
	# plt.show()
	# quit()

	# tracklet_box = get_tracklets(candidates_3D)

	# best_tracklet = find_best_tracklet(tracklet_box)

	# from scipy.optimize import curve_fit

	# x_points = []
	# y_points = []
	# z_points = []

	# if best_tracklet is None:
	# 	quit()

	# for i, tok in enumerate(best_tracklet.tokens):
	# 	x_points.append(tok.coords[X])
	# 	y_points.append(tok.coords[Y])
	# 	z_points.append(tok.coords[Z])

	# ax.scatter(xs=x_points,ys=y_points,zs=z_points,c=np.arange(len(x_points)), cmap='winter')

	# t = np.linspace(best_tracklet.start_frame*1/90, \
	# 				(best_tracklet.start_frame+best_tracklet.length)*1/90, \
	# 				best_tracklet.length)

	# x_params, covmatrix = curve_fit(curve_func, t, x_points)
	# y_params, covmatrix = curve_fit(curve_func, t, y_points)
	# z_params, covmatrix = curve_fit(curve_func, t, z_points)

	# t = np.linspace(0,2,1000)

	# x_est = curve_func(t,*x_params)
	# y_est = curve_func(t,*y_params)
	# z_est = curve_func(t,*z_params)

	# xd1_est = d1_curve_func(t,*x_params)
	# yd1_est = d1_curve_func(t,*y_params)
	# zd1_est = d1_curve_func(t,*z_params)

	# for i, z in enumerate(z_est):
	# 	if z<=0:
	# 		bounce_pos = i
	# 		break
	# # bounce_pos = len(z_est[z_est>0])-1

	# x_vel = xd1_est[bounce_pos]
	# y_vel = yd1_est[bounce_pos]
	# z_vel = zd1_est[bounce_pos]

	# print(x_vel,y_vel,z_vel)
	# print(f"velocity: {np.sqrt(x_vel**2+y_vel**2+z_vel**2):2.2f} m/s")
	# print(f"bounce_loc: {x_est[bounce_pos]:0.2f}, {y_est[bounce_pos]:0.2f},{z_est[bounce_pos]:0.2f}")

	# z_est[bounce_pos:] = None

	# print(bounce_pos)

	# ax.plot3D(x_est,y_est,z_est)
	# plt.show()