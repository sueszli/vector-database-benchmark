import numpy as np
from gym import spaces
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary

class TrajAviary(BaseSingleAgentAviary):
    """Single agent RL problem: learn to track trajectory from scratch"""

    def __init__(self, drone_model: DroneModel=DroneModel.CF2X, initial_xyzs=None, initial_rpys=None, physics: Physics=Physics.PYB, freq: int=240, aggregate_phy_steps: int=1, gui=False, record=False, obs: ObservationType=ObservationType.KIN, act: ActionType=ActionType.RPM):
        if False:
            i = 10
            return i + 15
        "Initialization of a single agent RL environment.\n\n        Using the generic single agent RL superclass.\n\n        Parameters\n        ----------\n        drone_model : DroneModel, optional\n            The desired drone type (detailed in an .urdf file in folder `assets`).\n        initial_xyzs: ndarray | None, optional\n            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.\n        initial_rpys: ndarray | None, optional\n            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).\n        physics : Physics, optional\n            The desired implementation of PyBullet physics/custom dynamics.\n        freq : int, optional\n            The frequency (Hz) at which the physics engine steps.\n        aggregate_phy_steps : int, optional\n            The number of physics steps within one call to `BaseAviary.step()`.\n        gui : bool, optional\n            Whether to use PyBullet's GUI.\n        record : bool, optional\n            Whether to save a video of the simulation in folder `files/videos/`.\n        obs : ObservationType, optional\n            The type of observation space (kinematic information or vision)\n        act : ActionType, optional\n            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)\n\n        "
        super().__init__(drone_model=drone_model, initial_xyzs=np.array([[0, 0, 1]]).reshape(1, 3), initial_rpys=initial_rpys, physics=physics, freq=freq, aggregate_phy_steps=aggregate_phy_steps, gui=gui, record=record, obs=obs, act=act)
        self.TRAJ_STEPS = int(self.SIM_FREQ * self.EPISODE_LEN_SEC / self.AGGR_PHY_STEPS)
        self.CTRL_TIMESTEP = self.AGGR_PHY_STEPS * self.TIMESTEP
        self.TARGET_POSITION = np.array([[0, 4.0 * np.sin(0.006 * self.TRAJ_STEPS), 1.0] for i in range(self.TRAJ_STEPS)])
        self.TARGET_VELOCITY = np.zeros([self.TRAJ_STEPS, 3])
        self.TARGET_VELOCITY[1:, :] = (self.TARGET_POSITION[1:, :] - self.TARGET_POSITION[0:-1, :]) / self.CTRL_TIMESTEP

    def _computeReward(self):
        if False:
            while True:
                i = 10
        'Computes the current reward value.\n\n        Returns\n        -------\n        float\n            The reward.\n\n        '
        state = self._getDroneStateVector(0)
        i = min(int(self.step_counter / self.AGGR_PHY_STEPS), self.TRAJ_STEPS - 1)
        return -1 * (np.linalg.norm(self.TARGET_POSITION[i, :] - state[0:3]) ** 2 + np.linalg.norm(self.TARGET_VELOCITY[i, :] - state[7:10]))

    def _computeDone(self):
        if False:
            return 10
        'Computes the current done value.\n\n        Returns\n        -------\n        bool\n            Whether the current episode is done.\n\n        '
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeInfo(self):
        if False:
            return 10
        'Computes the current info dict(s).\n\n        Unused.\n\n        Returns\n        -------\n        dict[str, int]\n            Dummy value.\n\n        '
        return {'answer': 42}

    def _clipAndNormalizeState(self, state):
        if False:
            while True:
                i = 10
        "Normalizes a drone's state to the [-1,1] range.\n\n        Parameters\n        ----------\n        state : ndarray\n            (20,)-shaped array of floats containing the non-normalized state of a single drone.\n\n        Returns\n        -------\n        ndarray\n            (20,)-shaped array of floats containing the normalized state of a single drone.\n\n        "
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1
        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC
        MAX_PITCH_ROLL = np.pi
        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        if self.GUI:
            self._clipAndNormalizeStateWarning(state, clipped_pos_xy, clipped_pos_z, clipped_rp, clipped_vel_xy, clipped_vel_z)
        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        norm_and_clipped = np.hstack([normalized_pos_xy, normalized_pos_z, state[3:7], normalized_rp, normalized_y, normalized_vel_xy, normalized_vel_z, normalized_ang_vel, state[16:20]]).reshape(20)
        return norm_and_clipped

    def _clipAndNormalizeStateWarning(self, state, clipped_pos_xy, clipped_pos_z, clipped_rp, clipped_vel_xy, clipped_vel_z):
        if False:
            return 10
        'Debugging printouts associated to `_clipAndNormalizeState`.\n\n        Print a warning if values in a state vector is out of the clipping range.\n        \n        '
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print('[WARNING] it', self.step_counter, 'in TuneAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]'.format(state[0], state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print('[WARNING] it', self.step_counter, 'in TuneAviary._clipAndNormalizeState(), clipped z position [{:.2f}]'.format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print('[WARNING] it', self.step_counter, 'in TuneAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]'.format(state[7], state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print('[WARNING] it', self.step_counter, 'in TuneAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]'.format(state[10], state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print('[WARNING] it', self.step_counter, 'in TuneAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]'.format(state[12]))