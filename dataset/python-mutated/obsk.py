import itertools
import numpy as np
from copy import deepcopy

class Node:

    def __init__(self, label, qpos_ids, qvel_ids, act_ids, body_fn=None, bodies=None, extra_obs=None, tendons=None):
        if False:
            for i in range(10):
                print('nop')
        self.label = label
        self.qpos_ids = qpos_ids
        self.qvel_ids = qvel_ids
        self.act_ids = act_ids
        self.bodies = bodies
        self.extra_obs = {} if extra_obs is None else extra_obs
        self.body_fn = body_fn
        self.tendons = tendons
        pass

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.label

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.label

class HyperEdge:

    def __init__(self, *edges):
        if False:
            print('Hello World!')
        self.edges = set(edges)

    def __contains__(self, item):
        if False:
            print('Hello World!')
        return item in self.edges

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'HyperEdge({})'.format(self.edges)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'HyperEdge({})'.format(self.edges)

def get_joints_at_kdist(agent_id, agent_partitions, hyperedges, k=0, kagents=False):
    if False:
        for i in range(10):
            print('nop')
    ' Identify all joints at distance <= k from agent agent_id\n\n    :param agent_id: id of agent to be considered\n    :param agent_partitions: list of joint tuples in order of agentids\n    :param edges: list of tuples (joint1, joint2)\n    :param k: kth degree\n    :param kagents: True (observe all joints of an agent if a single one is) or False (individual joint granularity)\n    :return:\n        dict with k as key, and list of joints at that distance\n    '
    assert not kagents, 'kagents not implemented!'
    agent_joints = agent_partitions[agent_id]

    def _adjacent(lst, kagents=False):
        if False:
            for i in range(10):
                print('nop')
        ret = set([])
        for l in lst:
            ret = ret.union(set(itertools.chain(*[e.edges.difference({l}) for e in hyperedges if l in e])))
        return ret
    seen = set([])
    new = set([])
    k_dict = {}
    for _k in range(k + 1):
        if not _k:
            new = set(agent_joints)
        else:
            print(hyperedges)
            new = _adjacent(new) - seen
        seen = seen.union(new)
        k_dict[_k] = sorted(list(new), key=lambda x: x.label)
    return k_dict

def build_obs(env, k_dict, k_categories, global_dict, global_categories, vec_len=None):
    if False:
        for i in range(10):
            print('nop')
    'Given a k_dict from get_joints_at_kdist, extract observation vector.\n\n    :param k_dict: k_dict\n    :param qpos: qpos numpy array\n    :param qvel: qvel numpy array\n    :param vec_len: if None no padding, else zero-pad to vec_len\n    :return:\n    observation vector\n    '
    body_set_dict = {}
    obs_lst = []
    for k in sorted(list(k_dict.keys())):
        cats = k_categories[k]
        for _t in k_dict[k]:
            for c in cats:
                if c in _t.extra_obs:
                    items = _t.extra_obs[c](env).tolist()
                    obs_lst.extend(items if isinstance(items, list) else [items])
                elif c in ['qvel', 'qpos']:
                    items = getattr(env.sim.data, c)[getattr(_t, '{}_ids'.format(c))]
                    obs_lst.extend(items if isinstance(items, list) else [items])
                elif c in ['qfrc_actuator']:
                    items = getattr(env.sim.data, c)[getattr(_t, '{}_ids'.format('qvel'))]
                    obs_lst.extend(items if isinstance(items, list) else [items])
                elif c in ['cvel', 'cinert', 'cfrc_ext']:
                    if _t.bodies is not None:
                        for b in _t.bodies:
                            if c not in body_set_dict:
                                body_set_dict[c] = set()
                            if b not in body_set_dict[c]:
                                items = getattr(env.sim.data, c)[b].tolist()
                                items = getattr(_t, 'body_fn', lambda _id, x: x)(b, items)
                                obs_lst.extend(items if isinstance(items, list) else [items])
                                body_set_dict[c].add(b)
    body_set_dict = {}
    for c in global_categories:
        if c in ['qvel', 'qpos']:
            for j in global_dict.get('joints', []):
                items = getattr(env.sim.data, c)[getattr(j, '{}_ids'.format(c))]
                obs_lst.extend(items if isinstance(items, list) else [items])
        else:
            for b in global_dict.get('bodies', []):
                if c not in body_set_dict:
                    body_set_dict[c] = set()
                if b not in body_set_dict[c]:
                    obs_lst.extend(getattr(env.sim.data, c)[b].tolist())
                    body_set_dict[c].add(b)
    if vec_len is not None:
        pad = np.array((vec_len - len(obs_lst)) * [0])
        if len(pad):
            return np.concatenate([np.array(obs_lst), pad])
    return np.array(obs_lst)

def build_actions(agent_partitions, k_dict):
    if False:
        for i in range(10):
            print('nop')
    pass

def get_parts_and_edges(label, partitioning):
    if False:
        while True:
            i = 10
    if label in ['half_cheetah', 'HalfCheetah-v2']:
        bthigh = Node('bthigh', -6, -6, 0)
        bshin = Node('bshin', -5, -5, 1)
        bfoot = Node('bfoot', -4, -4, 2)
        fthigh = Node('fthigh', -3, -3, 3)
        fshin = Node('fshin', -2, -2, 4)
        ffoot = Node('ffoot', -1, -1, 5)
        edges = [HyperEdge(bfoot, bshin), HyperEdge(bshin, bthigh), HyperEdge(bthigh, fthigh), HyperEdge(fthigh, fshin), HyperEdge(fshin, ffoot)]
        root_x = Node('root_x', 0, 0, -1, extra_obs={'qpos': lambda env: np.array([])})
        root_z = Node('root_z', 1, 1, -1)
        root_y = Node('root_y', 2, 2, -1)
        globals = {'joints': [root_x, root_y, root_z]}
        if partitioning == '2x3':
            parts = [(bfoot, bshin, bthigh), (ffoot, fshin, fthigh)]
        elif partitioning == '6x1':
            parts = [(bfoot,), (bshin,), (bthigh,), (ffoot,), (fshin,), (fthigh,)]
        else:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        return (parts, edges, globals)
    elif label in ['Ant-v2']:
        torso = 1
        front_left_leg = 2
        aux_1 = 3
        ankle_1 = 4
        front_right_leg = 5
        aux_2 = 6
        ankle_2 = 7
        back_leg = 8
        aux_3 = 9
        ankle_3 = 10
        right_back_leg = 11
        aux_4 = 12
        ankle_4 = 13
        hip1 = Node('hip1', -8, -8, 2, bodies=[torso, front_left_leg], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
        ankle1 = Node('ankle1', -7, -7, 3, bodies=[front_left_leg, aux_1, ankle_1], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
        hip2 = Node('hip2', -6, -6, 4, bodies=[torso, front_right_leg], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
        ankle2 = Node('ankle2', -5, -5, 5, bodies=[front_right_leg, aux_2, ankle_2], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
        hip3 = Node('hip3', -4, -4, 6, bodies=[torso, back_leg], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
        ankle3 = Node('ankle3', -3, -3, 7, bodies=[back_leg, aux_3, ankle_3], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
        hip4 = Node('hip4', -2, -2, 0, bodies=[torso, right_back_leg], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
        ankle4 = Node('ankle4', -1, -1, 1, bodies=[right_back_leg, aux_4, ankle_4], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
        edges = [HyperEdge(ankle4, hip4), HyperEdge(ankle1, hip1), HyperEdge(ankle2, hip2), HyperEdge(ankle3, hip3), HyperEdge(hip4, hip1, hip2, hip3)]
        free_joint = Node('free', 0, 0, -1, extra_obs={'qpos': lambda env: env.sim.data.qpos[:7], 'qvel': lambda env: env.sim.data.qvel[:6], 'cfrc_ext': lambda env: np.clip(env.sim.data.cfrc_ext[0:1], -1, 1)})
        globals = {'joints': [free_joint]}
        if partitioning == '2x4':
            parts = [(hip1, ankle1, hip2, ankle2), (hip3, ankle3, hip4, ankle4)]
        elif partitioning == '2x4d':
            parts = [(hip1, ankle1, hip3, ankle3), (hip2, ankle2, hip4, ankle4)]
        elif partitioning == '4x2':
            parts = [(hip1, ankle1), (hip2, ankle2), (hip3, ankle3), (hip4, ankle4)]
        else:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        return (parts, edges, globals)
    elif label in ['Hopper-v2']:
        thigh_joint = Node('thigh_joint', -3, -3, 0, extra_obs={'qvel': lambda env: np.clip(np.array([env.sim.data.qvel[-3]]), -10, 10)})
        leg_joint = Node('leg_joint', -2, -2, 1, extra_obs={'qvel': lambda env: np.clip(np.array([env.sim.data.qvel[-2]]), -10, 10)})
        foot_joint = Node('foot_joint', -1, -1, 2, extra_obs={'qvel': lambda env: np.clip(np.array([env.sim.data.qvel[-1]]), -10, 10)})
        edges = [HyperEdge(foot_joint, leg_joint), HyperEdge(leg_joint, thigh_joint)]
        root_x = Node('root_x', 0, 0, -1, extra_obs={'qpos': lambda env: np.array([]), 'qvel': lambda env: np.clip(np.array([env.sim.data.qvel[1]]), -10, 10)})
        root_z = Node('root_z', 1, 1, -1, extra_obs={'qvel': lambda env: np.clip(np.array([env.sim.data.qvel[1]]), -10, 10)})
        root_y = Node('root_y', 2, 2, -1, extra_obs={'qvel': lambda env: np.clip(np.array([env.sim.data.qvel[2]]), -10, 10)})
        globals = {'joints': [root_x, root_y, root_z]}
        if partitioning == '3x1':
            parts = [(thigh_joint,), (leg_joint,), (foot_joint,)]
        else:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        return (parts, edges, globals)
    elif label in ['Humanoid-v2', 'HumanoidStandup-v2']:
        abdomen_y = Node('abdomen_y', -16, -16, 0)
        abdomen_z = Node('abdomen_z', -17, -17, 1)
        abdomen_x = Node('abdomen_x', -15, -15, 2)
        right_hip_x = Node('right_hip_x', -14, -14, 3)
        right_hip_z = Node('right_hip_z', -13, -13, 4)
        right_hip_y = Node('right_hip_y', -12, -12, 5)
        right_knee = Node('right_knee', -11, -11, 6)
        left_hip_x = Node('left_hip_x', -10, -10, 7)
        left_hip_z = Node('left_hip_z', -9, -9, 8)
        left_hip_y = Node('left_hip_y', -8, -8, 9)
        left_knee = Node('left_knee', -7, -7, 10)
        right_shoulder1 = Node('right_shoulder1', -6, -6, 11)
        right_shoulder2 = Node('right_shoulder2', -5, -5, 12)
        right_elbow = Node('right_elbow', -4, -4, 13)
        left_shoulder1 = Node('left_shoulder1', -3, -3, 14)
        left_shoulder2 = Node('left_shoulder2', -2, -2, 15)
        left_elbow = Node('left_elbow', -1, -1, 16)
        edges = [HyperEdge(abdomen_x, abdomen_y, abdomen_z), HyperEdge(right_hip_x, right_hip_y, right_hip_z), HyperEdge(left_hip_x, left_hip_y, left_hip_z), HyperEdge(left_elbow, left_shoulder1, left_shoulder2), HyperEdge(right_elbow, right_shoulder1, right_shoulder2), HyperEdge(left_knee, left_hip_x, left_hip_y, left_hip_z), HyperEdge(right_knee, right_hip_x, right_hip_y, right_hip_z), HyperEdge(left_shoulder1, left_shoulder2, abdomen_x, abdomen_y, abdomen_z), HyperEdge(right_shoulder1, right_shoulder2, abdomen_x, abdomen_y, abdomen_z), HyperEdge(abdomen_x, abdomen_y, abdomen_z, left_hip_x, left_hip_y, left_hip_z), HyperEdge(abdomen_x, abdomen_y, abdomen_z, right_hip_x, right_hip_y, right_hip_z)]
        globals = {}
        if partitioning == '9|8':
            parts = [(left_shoulder1, left_shoulder2, abdomen_x, abdomen_y, abdomen_z, right_shoulder1, right_shoulder2, right_elbow, left_elbow), (left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z, right_knee, left_knee)]
        else:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        return (parts, edges, globals)
    elif label in ['Reacher-v2']:
        body0 = 1
        body1 = 2
        fingertip = 3
        joint0 = Node('joint0', -4, -4, 0, bodies=[body0, body1], extra_obs={'qpos': lambda env: np.array([np.sin(env.sim.data.qpos[-4]), np.cos(env.sim.data.qpos[-4])])})
        joint1 = Node('joint1', -3, -3, 1, bodies=[body1, fingertip], extra_obs={'fingertip_dist': lambda env: env.get_body_com('fingertip') - env.get_body_com('target'), 'qpos': lambda env: np.array([np.sin(env.sim.data.qpos[-3]), np.cos(env.sim.data.qpos[-3])])})
        edges = [HyperEdge(joint0, joint1)]
        worldbody = 0
        target = 4
        target_x = Node('target_x', -2, -2, -1, extra_obs={'qvel': lambda env: np.array([])})
        target_y = Node('target_y', -1, -1, -1, extra_obs={'qvel': lambda env: np.array([])})
        globals = {'bodies': [worldbody, target], 'joints': [target_x, target_y]}
        if partitioning == '2x1':
            parts = [(joint0,), (joint1,)]
        else:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        return (parts, edges, globals)
    elif label in ['Swimmer-v2']:
        joint0 = Node('rot2', -2, -2, 0)
        joint1 = Node('rot3', -1, -1, 1)
        edges = [HyperEdge(joint0, joint1)]
        globals = {}
        if partitioning == '2x1':
            parts = [(joint0,), (joint1,)]
        else:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        return (parts, edges, globals)
    elif label in ['Walker2d-v2']:
        thigh_joint = Node('thigh_joint', -6, -6, 0)
        leg_joint = Node('leg_joint', -5, -5, 1)
        foot_joint = Node('foot_joint', -4, -4, 2)
        thigh_left_joint = Node('thigh_left_joint', -3, -3, 3)
        leg_left_joint = Node('leg_left_joint', -2, -2, 4)
        foot_left_joint = Node('foot_left_joint', -1, -1, 5)
        edges = [HyperEdge(foot_joint, leg_joint), HyperEdge(leg_joint, thigh_joint), HyperEdge(foot_left_joint, leg_left_joint), HyperEdge(leg_left_joint, thigh_left_joint), HyperEdge(thigh_joint, thigh_left_joint)]
        globals = {}
        if partitioning == '2x3':
            parts = [(foot_joint, leg_joint, thigh_joint), (foot_left_joint, leg_left_joint, thigh_left_joint)]
        else:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        return (parts, edges, globals)
    elif label in ['coupled_half_cheetah']:
        tendon = 0
        bthigh = Node('bthigh', -6, -6, 0, tendons=[tendon], extra_obs={'ten_J': lambda env: env.sim.data.ten_J[tendon], 'ten_length': lambda env: env.sim.data.ten_length, 'ten_velocity': lambda env: env.sim.data.ten_velocity})
        bshin = Node('bshin', -5, -5, 1)
        bfoot = Node('bfoot', -4, -4, 2)
        fthigh = Node('fthigh', -3, -3, 3)
        fshin = Node('fshin', -2, -2, 4)
        ffoot = Node('ffoot', -1, -1, 5)
        bthigh2 = Node('bthigh2', -6, -6, 0, tendons=[tendon], extra_obs={'ten_J': lambda env: env.sim.data.ten_J[tendon], 'ten_length': lambda env: env.sim.data.ten_length, 'ten_velocity': lambda env: env.sim.data.ten_velocity})
        bshin2 = Node('bshin2', -5, -5, 1)
        bfoot2 = Node('bfoot2', -4, -4, 2)
        fthigh2 = Node('fthigh2', -3, -3, 3)
        fshin2 = Node('fshin2', -2, -2, 4)
        ffoot2 = Node('ffoot2', -1, -1, 5)
        edges = [HyperEdge(bfoot, bshin), HyperEdge(bshin, bthigh), HyperEdge(bthigh, fthigh), HyperEdge(fthigh, fshin), HyperEdge(fshin, ffoot), HyperEdge(bfoot2, bshin2), HyperEdge(bshin2, bthigh2), HyperEdge(bthigh2, fthigh2), HyperEdge(fthigh2, fshin2), HyperEdge(fshin2, ffoot2)]
        globals = {}
        root_x = Node('root_x', 0, 0, -1, extra_obs={'qpos': lambda env: np.array([])})
        root_z = Node('root_z', 1, 1, -1)
        root_y = Node('root_y', 2, 2, -1)
        globals = {'joints': [root_x, root_y, root_z]}
        if partitioning == '1p1':
            parts = [(bfoot, bshin, bthigh, ffoot, fshin, fthigh), (bfoot2, bshin2, bthigh2, ffoot2, fshin2, fthigh2)]
        else:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        return (parts, edges, globals)
    elif label in ['manyagent_swimmer']:
        try:
            n_agents = int(partitioning.split('x')[0])
            n_segs_per_agents = int(partitioning.split('x')[1])
            n_segs = n_agents * n_segs_per_agents
        except Exception as e:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        joints = [Node('rot{:d}'.format(i), -n_segs + i, -n_segs + i, i) for i in range(0, n_segs)]
        edges = [HyperEdge(joints[i], joints[i + 1]) for i in range(n_segs - 1)]
        globals = {}
        parts = [tuple(joints[i * n_segs_per_agents:(i + 1) * n_segs_per_agents]) for i in range(n_agents)]
        return (parts, edges, globals)
    elif label in ['manyagent_ant']:
        try:
            n_agents = int(partitioning.split('x')[0])
            n_segs_per_agents = int(partitioning.split('x')[1])
            n_segs = n_agents * n_segs_per_agents
        except Exception as e:
            raise Exception('UNKNOWN partitioning config: {}'.format(partitioning))
        edges = []
        joints = []
        for si in range(n_segs):
            torso = 1 + si * 7
            front_right_leg = 2 + si * 7
            aux1 = 3 + si * 7
            ankle1 = 4 + si * 7
            back_leg = 5 + si * 7
            aux2 = 6 + si * 7
            ankle2 = 7 + si * 7
            off = -4 * (n_segs - 1 - si)
            hip1n = Node('hip1_{:d}'.format(si), -4 - off, -4 - off, 2 + 4 * si, bodies=[torso, front_right_leg], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
            ankle1n = Node('ankle1_{:d}'.format(si), -3 - off, -3 - off, 3 + 4 * si, bodies=[front_right_leg, aux1, ankle1], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
            hip2n = Node('hip2_{:d}'.format(si), -2 - off, -2 - off, 0 + 4 * si, bodies=[torso, back_leg], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
            ankle2n = Node('ankle2_{:d}'.format(si), -1 - off, -1 - off, 1 + 4 * si, bodies=[back_leg, aux2, ankle2], body_fn=lambda _id, x: np.clip(x, -1, 1).tolist())
            edges += [HyperEdge(ankle1n, hip1n), HyperEdge(ankle2n, hip2n), HyperEdge(hip1n, hip2n)]
            if si:
                edges += [HyperEdge(hip1m, hip2m, hip1n, hip2n)]
            hip1m = deepcopy(hip1n)
            hip2m = deepcopy(hip2n)
            joints.append([hip1n, ankle1n, hip2n, ankle2n])
        free_joint = Node('free', 0, 0, -1, extra_obs={'qpos': lambda env: env.sim.data.qpos[:7], 'qvel': lambda env: env.sim.data.qvel[:6], 'cfrc_ext': lambda env: np.clip(env.sim.data.cfrc_ext[0:1], -1, 1)})
        globals = {'joints': [free_joint]}
        parts = [[x for sublist in joints[i * n_segs_per_agents:(i + 1) * n_segs_per_agents] for x in sublist] for i in range(n_agents)]
        return (parts, edges, globals)