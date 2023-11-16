import numpy as np
import os
import time
import copy

import consts as c
import graph_utils as gr

class TrackletBox(object):
	def __init__(self):
		self.tracklets = []

	def __del__(self):
		for tracklet in self.tracklets:
			del tracklet

	def add_tracklet(self, tracklet):
		self.tracklets.append(tracklet)

	def validate_tracklets(self):
		for tracklet in self.tracklets:
			score_tok = tracklet.score/tracklet.length
			if score_tok < c.SCORE_TOK_THRESH:
				tracklet.is_valid = False

	def merge_tracklets(self):		
		graph = gr.create_graph(self.tracklets)

		## -- Same start frame -- ##
		for t1 in self.tracklets:
			hiscore = t1.score
			for t2 in self.tracklets:
				if t1 is not t2:
					if t1.start_frame == t2.start_frame:
						# tracklets start at the same point
						# remove the tracklet with the lower score
						if t2.score > hiscore:
							hiscore = t2.score
							t1.is_valid = False
						else:
							t2.is_valid = False

		# -- Temporal overlap -- ##
		for t in range(len(self.tracklets)-1):
			cons_count_max = 0
			cons_pos_max = None
			for tok1 in reversed(self.tracklets[t].tokens[-c.MAX_SHARED_TOKS:]):
				cons_count = 0
				cons_pos = 0
				cons = False
				for tok2 in self.tracklets[t+1].tokens[:c.MAX_SHARED_TOKS]:
					if tok1.f == tok2.f:
						sim = tok1.calc_similarity(tok2)
					else:
						continue

					if sim < c.TOKEN_SIM_THRESH:
						cons = True
						cons_count += 1
						cons_pos = self.tracklets[t+1].tokens.index(tok2)
					else:
						break

					if cons == True and cons_count > cons_count_max:
						cons_pos_max = cons_pos

			if cons_pos_max is not None:
				graph[t].append(t+1)
				for i, tok in enumerate(self.tracklets[t+1].tokens):
					if i<=cons_pos_max:
						self.tracklets[t+1].score -= self.tracklets[t+1].tokens[i].score
						self.tracklets[t+1].tokens[i].score = 0

			else:
				if self.tracklets[t].length > 3 and self.tracklets[t+1].length > 3:
					first_extrapolation_points = []
					second_extrapolation_points = []

					for i in range(3):
						first_extrapolation_points.append(self.tracklets[t].tokens[i-3].coords)
						second_extrapolation_points.append(self.tracklets[t+1].tokens[2-i].coords)

					for i in range(c.EXTRAPOLATE_N):
						first_extrapolation_points.append(
							make_est(	first_extrapolation_points[-3],
										first_extrapolation_points[-2],
										first_extrapolation_points[-1]))
						
						second_extrapolation_points.append(
							make_est(	second_extrapolation_points[-3],
										second_extrapolation_points[-2],
										second_extrapolation_points[-1]))

					first_extrapolation_points = first_extrapolation_points[-c.EXTRAPOLATE_N:]
					second_extrapolation_points = second_extrapolation_points[-c.EXTRAPOLATE_N:]

					best_match = c.TOKEN_SIM_THRESH
					best_f_p = None
					best_s_p = None
					for i, f_p in enumerate(first_extrapolation_points):
						for j, s_p in enumerate(second_extrapolation_points):
							sim = calc_dist(f_p-s_p)
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
							self.tracklets[t].add_token(Token(self.tracklets[t].tokens[-1].f+1,first_point,score=1))

						for second_point in reversed(new_second_points):
							self.tracklets[t].add_token(Token(self.tracklets[t].tokens[-1].f+1,second_point,score=1))

						graph[t].append(t+1)

		start_nodes, end_nodes = gr.get_start_end_nodes(graph)

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
						merged_track.add_token(Token(f=tok.f, coords=tok.coords, score=tok.score))
						f = tok.f
				self.tracklets[t].is_valid = False

			return merged_track

		else: return None


	def tok_score_sum(self, tokens):
		score = 0
		for tok in tokens:
			score += tok.score

		return score

def make_est(c1,c2,c3):
	a3 = ((c3-c2)-(c2-c1))/(c.dT**2)			# acceleration
	v3 = ((c3-c2)/(c.dT)) + a3*c.dT 			# velocity

	c4_e = c3+v3*c.dT+((a3*c.dT**2)/(2)) 		# next point estimation
	return c4_e

class Tracklet(object):
	def __init__(self, start_frame, score=0,length=0):
		self.start_frame = start_frame
		self.tokens = list()
		self.score = score
		self.length = length
		self.con_est = 0
		self.is_valid = True

	def __del__(self):
		for token in self.tokens:
			del token

	def save_tracklet(self, tracklet_box):
		if self.score < c.TRACKLET_SCORE_THRESH:
			return
		if tracklet_box is not None:
			tracklet_box.add_tracklet(copy.deepcopy(self))

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
			est = make_est(self.tokens[-3].coords,self.tokens[-2].coords,self.tokens[-1].coords)
			return est
		else:
			return None

	def add_est(self, token):
		if self.con_est < c.MAX_EST:
			self.add_token(token)
			self.con_est += 1
			return True
		else:
			self.con_est = 0
			return False

class Token(object):
	def __init__(self, f, coords, score=0):
		self.f = f 
		self.coords = coords
		self.score = score
		self.is_valid = True

	def __del__(self):
		del self

	def calc_similarity(self, token):
		error = self.coords-token.coords
		return calc_dist(error)

def calc_dist(vect):
	a = 0
	for el in vect[:3]:
		a += el**2

	return np.sqrt(a)

def calc_theta_phi(diff, r):
	r_p = np.sqrt(diff[c.X_3D]**2+diff[c.Y_3D]**2)

	if r_p > 0 and r>0:
		theta = np.arccos(diff[c.Y_3D]/r_p)
		phi = np.arccos(r_p/r)
		return theta, phi
	else:
		return c.PI, c.PI

def score_node(est, candidate):
	diff = est-candidate
	r = calc_dist(diff)
	theta, phi = calc_theta_phi(diff, r)

	if r<c.dM:
		s1 = np.exp(-r/c.dM)
		s2 = np.exp(-theta/c.thetaM)
		s3 = np.exp(-phi/c.phiM)
		return s1+s2+s3
	else:
		return 0

def evaluate(candidates_3D, tracklet, tracklet_box, f, f_max):
	# want to stop recursion when
	# - 3 consecutive estimates are made
	# - f == max
	if f < f_max:
		est = tracklet.est_next()

		# are there tokens to compare?
		# yes: compare the tokens and continue
		# no: add the estimate as a token and continue
		valid_cand = False
		if candidates_3D[f] != []:
			valid_cand = False
			for i, cand in enumerate(candidates_3D[f]):
				c4 = cand[c.CAND_DATA]
				candidates_3D[f][i][c.CAND_INIT] = True
				score = score_node(est, c4)
				if score > c.TOKEN_SCORE_THRESH:
					valid_cand = True
					if f_max-c.WIN_SIZE <= f:
						tracklet.add_token(Token(f, c4, score))
					else:
						tracklet.add_token(Token(f, c4, 0))

					evaluate(candidates_3D, tracklet, tracklet_box, f+1, f_max)
					tracklet.del_token()

		if valid_cand is False:
			if tracklet.add_est(Token(f, est)):
				evaluate(candidates_3D, tracklet, tracklet_box, f+1, f_max)
				tracklet.del_token()
			else:
				# added 3 estimates, stop recursion and save tracklet
				tracklet.save_tracklet(tracklet_box)

	else:
		# tracklet is max length
		# save tracklet and stop recursion
		tracklet.save_tracklet(tracklet_box)

def check_init_toks(c1,c2,c3):
	for cand in [c1,c2,c3]:
		if 	cand[c.X_3D] > c.XMAX or cand[c.X_3D] < c.XMIN \
		or	cand[c.Y_3D] > c.YMAX or cand[c.Y_3D] < c.YMIN \
		or	cand[c.Z_3D] > c.ZMAX or cand[c.Z_3D] < c.ZMIN:
			return False

	d1 = calc_dist(c2-c1)
	d2 = calc_dist(c3-c2)

	if d1<c.dM and d2<c.dM and d1>0 and d2>0:
		return True
	else:
		return False


def get_tracklets(candidates_3D):
	for f, frame in enumerate(candidates_3D):
		for cand, candidate in enumerate(frame):
			candidates_3D[f][cand] = [False, np.array(candidate)]

	## -- Shift Token Transfer -- ##
	num_frames = len(candidates_3D)
	tracklet_box = TrackletBox()

	for f in range(num_frames):
		win_start = 0
		win_end = 0

		if f == 0:
			win_start = 0
			if num_frames > c.WIN_SIZE:
				win_end = win_start+c.WIN_SIZE
			else:
				win_end = num_frames

		elif f % c.WIN_SIZE == 0 and f != 0:
			win_start = f-c.WIN_OVERLAP
			win_end = f+c.WIN_SIZE

			if win_end > num_frames:
				win_end = num_frames

			for frame in candidates_3D:
				for cand in frame:
					cand[c.CAND_INIT] = False
		
		else:
			continue

		init_set = False
		c1,c2,c3 = [],[],[]

		for cur_frame in range(win_start+3, win_end):
			if init_set is False:
				c1 = candidates_3D[cur_frame-3]
				c2 = candidates_3D[cur_frame-2]
				c3 = candidates_3D[cur_frame-1]

				if (c1 == []) or (c2 == []) or (c3 == []):
					continue
				else:
					init_set = True

			if init_set:
				tracklet = Tracklet(start_frame=cur_frame-3)

				for c1_c in c1:
					if c1_c[c.CAND_INIT] is True:	continue
					tracklet.add_token(Token(cur_frame-3,c1_c[c.CAND_DATA], score=0))
					for c2_c in c2:
						if c2_c[c.CAND_INIT] is True:	continue
						tracklet.add_token(Token(cur_frame-2,c2_c[c.CAND_DATA], score=0))
						for c3_c in c3:
							if c3_c[c.CAND_INIT] is True:	continue
							tracklet.add_token(Token(cur_frame-1,c3_c[c.CAND_DATA], score=0))

							c1_c[c.CAND_INIT] = True
							c2_c[c.CAND_INIT] = False
							c3_c[c.CAND_INIT] = False

							if check_init_toks(c1_c[c.CAND_DATA],c2_c[c.CAND_DATA],c3_c[c.CAND_DATA]):
								evaluate(candidates_3D, tracklet, tracklet_box, cur_frame, f_max=win_end)				

							tracklet.del_token()
						tracklet.del_token()
					tracklet.del_token()

				init_set = False
				c1,c2,c3 = [],[],[]

	print('done')
	best_tracklet = tracklet_box.merge_tracklets()

	return best_tracklet

def find_best_tracklet(tracklet_box):
	best_score, best_tracklet = 0, None
	for t in tracklet_box.tracklets:
		if t.is_valid:
			print(f"f_start: {t.start_frame}, f_end: {t.start_frame+t.length}, score: {t.score:0.2f}, score/tok: {t.score/t.length:0.2f}")

			if t.score>best_score:
				best_score = t.score
				best_tracklet = t

	return best_tracklet

def curve_func(t,a,b,c):
	return a+b*t+c*t**2

def d1_curve_func(t,a,b,c):
	return b+2*c*t

def split_tracklet(tracklet):
		acc = []
		vel = []
		for i, tok in enumerate(tracklet.tokens):
			if i==0:
				vel.append(0*tok.coords)
			else:
				vel.append(tracklet.tokens[i].coords-tracklet.tokens[i-1].coords)

		for j, v in enumerate(vel):
			if j<3:
				acc.append(0)
			else:
				if vel[j][c.Z_3D] > 0 and vel[j-1][c.Z_3D] < 0 and vel[j-2][c.Z_3D] < 0 and vel[j-3][c.Z_3D] < 0:
					acc.append(1)
				else: 
					acc.append(-1)

		for k, a in enumerate(acc):
			if k<2 or k>=len(acc)-1:
				pass
			else:
				if acc[k] > 0 and acc[k-1] <= 0 and acc[k+1] <=0 :
					new_track = Tracklet(start_frame=tracklet.start_frame, score=0,length=0)
					for tok in tracklet.tokens[:k]:
						new_track.add_token(tok)
				
					return new_track

		return tracklet

if __name__ == "__main__":
	RESOLUTION = (640,480)
	w,h = RESOLUTION

	os.chdir(ROOT_P + '\\' + 'img\\simulation_tests')

	candidates_3D = np.load('candidates_3D.npy', allow_pickle=True)

	# -- Plot points -- ##
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure(figsize=(15*1.25,4*1.25))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('x (m)')
	ax.set_ylabel('y (m)')
	ax.set_zlabel('z (m)')
	ax.set_xlim(c.XMIN/2, c.XMAX/2)
	ax.set_ylim(0, c.YMAX)
	ax.set_zlim(0, 2)
	ax.view_init(elev=20,azim=-20)

	tracklet_box = get_tracklets(candidates_3D)
	best_tracklet = find_best_tracklet(tracklet_box)

	if best_tracklet is None:
		print('no valid trajectories found')
		quit()

	best_tracklet = split_tracklet(best_tracklet)

	from scipy.optimize import curve_fit

	x_points = []
	y_points = []
	z_points = []

	if best_tracklet is None:
		quit()

	for i, tok in enumerate(best_tracklet.tokens):
		x_points.append(tok.coords[c.X_3D])
		y_points.append(tok.coords[c.Y_3D])
		z_points.append(tok.coords[c.Z_3D])

	for tracklet in tracklet_box.tracklets:
			for tok in tracklet.tokens:
				ax.scatter(xs=tok.coords[c.X_3D],ys=tok.coords[c.Y_3D],zs=tok.coords[c.Z_3D],c='blue',alpha=0.2)

	# ax.scatter(xs=x_points,ys=y_points,zs=z_points,c=np.arange(len(x_points)), cmap='winter')

	t = np.linspace(best_tracklet.start_frame*1/90, \
					(best_tracklet.start_frame+best_tracklet.length)*1/90, \
					best_tracklet.length)

	x_params, covmatrix = curve_fit(curve_func, t, x_points)
	y_params, covmatrix = curve_fit(curve_func, t, y_points)
	z_params, covmatrix = curve_fit(curve_func, t, z_points)

	t = np.linspace(0,2,1000)

	X_OFFSET = 0.016436118692901215
	Y_OFFSET = 0.6083691217642057
	Z_OFFSET = -0.04876521114374302

	x_params[0]-=X_OFFSET
	y_params[0]-=Y_OFFSET
	z_params[0]-=Z_OFFSET

	x_est = curve_func(t,*x_params)
	y_est = curve_func(t,*y_params)
	z_est = curve_func(t,*z_params)

	xd1_est = d1_curve_func(t,*x_params)
	yd1_est = d1_curve_func(t,*y_params)
	zd1_est = d1_curve_func(t,*z_params)

	bounce_pos = 0
	for i, z in enumerate(z_est):
		bounce_pos = i
		if z<=0:
			bounce_pos = i
			break
	# bounce_pos = len(z_est[z_est>0])-1

	x_vel = xd1_est[bounce_pos]
	y_vel = yd1_est[bounce_pos]
	z_vel = zd1_est[bounce_pos]

	print(f"velocity: {np.sqrt(x_vel**2+y_vel**2+z_vel**2):2.2f} m/s")
	print(f"bounce_loc: {x_est[bounce_pos]:0.2f}, {y_est[bounce_pos]:0.2f},{z_est[bounce_pos]:0.2f}")

	z_est[bounce_pos:] = None

	ax.plot3D(x_est,y_est,z_est,c='red')
	plt.show()