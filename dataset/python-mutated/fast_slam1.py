"""

FastSLAM 1.0 example

author: Atsushi Sakai (@Atsushi_twi)

"""
import math
import matplotlib.pyplot as plt
import numpy as np
Q = np.diag([3.0, np.deg2rad(10.0)]) ** 2
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2
Q_sim = np.diag([0.3, np.deg2rad(2.0)]) ** 2
R_sim = np.diag([0.5, np.deg2rad(10.0)]) ** 2
OFFSET_YAW_RATE_NOISE = 0.01
DT = 0.1
SIM_TIME = 50.0
MAX_RANGE = 20.0
M_DIST_TH = 2.0
STATE_SIZE = 3
LM_SIZE = 2
N_PARTICLE = 100
NTH = N_PARTICLE / 1.5
show_animation = True

class Particle:

    def __init__(self, n_landmark):
        if False:
            for i in range(10):
                print('nop')
        self.w = 1.0 / N_PARTICLE
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.lm = np.zeros((n_landmark, LM_SIZE))
        self.lmP = np.zeros((n_landmark * LM_SIZE, LM_SIZE))

def fast_slam1(particles, u, z):
    if False:
        while True:
            i = 10
    particles = predict_particles(particles, u)
    particles = update_with_observation(particles, z)
    particles = resampling(particles)
    return particles

def normalize_weight(particles):
    if False:
        return 10
    sum_w = sum([p.w for p in particles])
    try:
        for i in range(N_PARTICLE):
            particles[i].w /= sum_w
    except ZeroDivisionError:
        for i in range(N_PARTICLE):
            particles[i].w = 1.0 / N_PARTICLE
        return particles
    return particles

def calc_final_state(particles):
    if False:
        return 10
    xEst = np.zeros((STATE_SIZE, 1))
    particles = normalize_weight(particles)
    for i in range(N_PARTICLE):
        xEst[0, 0] += particles[i].w * particles[i].x
        xEst[1, 0] += particles[i].w * particles[i].y
        xEst[2, 0] += particles[i].w * particles[i].yaw
    xEst[2, 0] = pi_2_pi(xEst[2, 0])
    return xEst

def predict_particles(particles, u):
    if False:
        return 10
    for i in range(N_PARTICLE):
        px = np.zeros((STATE_SIZE, 1))
        px[0, 0] = particles[i].x
        px[1, 0] = particles[i].y
        px[2, 0] = particles[i].yaw
        ud = u + (np.random.randn(1, 2) @ R ** 0.5).T
        px = motion_model(px, ud)
        particles[i].x = px[0, 0]
        particles[i].y = px[1, 0]
        particles[i].yaw = px[2, 0]
    return particles

def add_new_landmark(particle, z, Q_cov):
    if False:
        while True:
            i = 10
    r = z[0]
    b = z[1]
    lm_id = int(z[2])
    s = math.sin(pi_2_pi(particle.yaw + b))
    c = math.cos(pi_2_pi(particle.yaw + b))
    particle.lm[lm_id, 0] = particle.x + r * c
    particle.lm[lm_id, 1] = particle.y + r * s
    dx = r * c
    dy = r * s
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)
    Gz = np.array([[dx / d, dy / d], [-dy / d2, dx / d2]])
    particle.lmP[2 * lm_id:2 * lm_id + 2] = np.linalg.inv(Gz) @ Q_cov @ np.linalg.inv(Gz.T)
    return particle

def compute_jacobians(particle, xf, Pf, Q_cov):
    if False:
        i = 10
        return i + 15
    dx = xf[0, 0] - particle.x
    dy = xf[1, 0] - particle.y
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)
    zp = np.array([d, pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)
    Hv = np.array([[-dx / d, -dy / d, 0.0], [dy / d2, -dx / d2, -1.0]])
    Hf = np.array([[dx / d, dy / d], [-dy / d2, dx / d2]])
    Sf = Hf @ Pf @ Hf.T + Q_cov
    return (zp, Hv, Hf, Sf)

def update_kf_with_cholesky(xf, Pf, v, Q_cov, Hf):
    if False:
        i = 10
        return i + 15
    PHt = Pf @ Hf.T
    S = Hf @ PHt + Q_cov
    S = (S + S.T) * 0.5
    s_chol = np.linalg.cholesky(S).T
    s_chol_inv = np.linalg.inv(s_chol)
    W1 = PHt @ s_chol_inv
    W = W1 @ s_chol_inv.T
    x = xf + W @ v
    P = Pf - W1 @ W1.T
    return (x, P)

def update_landmark(particle, z, Q_cov):
    if False:
        return 10
    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])
    (zp, Hv, Hf, Sf) = compute_jacobians(particle, xf, Pf, Q)
    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = pi_2_pi(dz[1, 0])
    (xf, Pf) = update_kf_with_cholesky(xf, Pf, dz, Q_cov, Hf)
    particle.lm[lm_id, :] = xf.T
    particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf
    return particle

def compute_weight(particle, z, Q_cov):
    if False:
        while True:
            i = 10
    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
    (zp, Hv, Hf, Sf) = compute_jacobians(particle, xf, Pf, Q_cov)
    dx = z[0:2].reshape(2, 1) - zp
    dx[1, 0] = pi_2_pi(dx[1, 0])
    try:
        invS = np.linalg.inv(Sf)
    except np.linalg.linalg.LinAlgError:
        print('singular')
        return 1.0
    num = np.exp(-0.5 * (dx.T @ invS @ dx))[0, 0]
    den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))
    w = num / den
    return w

def update_with_observation(particles, z):
    if False:
        i = 10
        return i + 15
    for iz in range(len(z[0, :])):
        landmark_id = int(z[2, iz])
        for ip in range(N_PARTICLE):
            if abs(particles[ip].lm[landmark_id, 0]) <= 0.01:
                particles[ip] = add_new_landmark(particles[ip], z[:, iz], Q)
            else:
                w = compute_weight(particles[ip], z[:, iz], Q)
                particles[ip].w *= w
                particles[ip] = update_landmark(particles[ip], z[:, iz], Q)
    return particles

def resampling(particles):
    if False:
        while True:
            i = 10
    '\n    low variance re-sampling\n    '
    particles = normalize_weight(particles)
    pw = []
    for i in range(N_PARTICLE):
        pw.append(particles[i].w)
    pw = np.array(pw)
    n_eff = 1.0 / (pw @ pw.T)
    if n_eff < NTH:
        w_cum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
        resample_id = base + np.random.rand(base.shape[0]) / N_PARTICLE
        inds = []
        ind = 0
        for ip in range(N_PARTICLE):
            while ind < w_cum.shape[0] - 1 and resample_id[ip] > w_cum[ind]:
                ind += 1
            inds.append(ind)
        tmp_particles = particles[:]
        for i in range(len(inds)):
            particles[i].x = tmp_particles[inds[i]].x
            particles[i].y = tmp_particles[inds[i]].y
            particles[i].yaw = tmp_particles[inds[i]].yaw
            particles[i].lm = tmp_particles[inds[i]].lm[:, :]
            particles[i].lmP = tmp_particles[inds[i]].lmP[:, :]
            particles[i].w = 1.0 / N_PARTICLE
    return particles

def calc_input(time):
    if False:
        return 10
    if time <= 3.0:
        v = 0.0
        yaw_rate = 0.0
    else:
        v = 1.0
        yaw_rate = 0.1
    u = np.array([v, yaw_rate]).reshape(2, 1)
    return u

def observation(xTrue, xd, u, rfid):
    if False:
        for i in range(10):
            print('nop')
    xTrue = motion_model(xTrue, u)
    z = np.zeros((3, 0))
    for i in range(len(rfid[:, 0])):
        dx = rfid[i, 0] - xTrue[0, 0]
        dy = rfid[i, 1] - xTrue[1, 0]
        d = math.hypot(dx, dy)
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5
            angle_with_noize = angle + np.random.randn() * Q_sim[1, 1] ** 0.5
            zi = np.array([dn, pi_2_pi(angle_with_noize), i]).reshape(3, 1)
            z = np.hstack((z, zi))
    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5
    ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5 + OFFSET_YAW_RATE_NOISE
    ud = np.array([ud1, ud2]).reshape(2, 1)
    xd = motion_model(xd, ud)
    return (xTrue, z, xd, ud)

def motion_model(x, u):
    if False:
        i = 10
        return i + 15
    F = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    B = np.array([[DT * math.cos(x[2, 0]), 0], [DT * math.sin(x[2, 0]), 0], [0.0, DT]])
    x = F @ x + B @ u
    x[2, 0] = pi_2_pi(x[2, 0])
    return x

def pi_2_pi(angle):
    if False:
        return 10
    return (angle + math.pi) % (2 * math.pi) - math.pi

def main():
    if False:
        while True:
            i = 10
    print(__file__ + ' start!!')
    time = 0.0
    RFID = np.array([[10.0, -2.0], [15.0, 10.0], [15.0, 15.0], [10.0, 20.0], [3.0, 15.0], [-5.0, 20.0], [-5.0, 5.0], [-10.0, 15.0]])
    n_landmark = RFID.shape[0]
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    xDR = np.zeros((STATE_SIZE, 1))
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    particles = [Particle(n_landmark) for _ in range(N_PARTICLE)]
    while SIM_TIME >= time:
        time += DT
        u = calc_input(time)
        (xTrue, z, xDR, ud) = observation(xTrue, xDR, u, RFID)
        particles = fast_slam1(particles, ud, z)
        xEst = calc_final_state(particles)
        x_state = xEst[0:STATE_SIZE]
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(RFID[:, 0], RFID[:, 1], '*k')
            for i in range(N_PARTICLE):
                plt.plot(particles[i].x, particles[i].y, '.r')
                plt.plot(particles[i].lm[:, 0], particles[i].lm[:, 1], 'xb')
            plt.plot(hxTrue[0, :], hxTrue[1, :], '-b')
            plt.plot(hxDR[0, :], hxDR[1, :], '-k')
            plt.plot(hxEst[0, :], hxEst[1, :], '-r')
            plt.plot(xEst[0], xEst[1], 'xk')
            plt.axis('equal')
            plt.grid(True)
            plt.pause(0.001)
if __name__ == '__main__':
    main()