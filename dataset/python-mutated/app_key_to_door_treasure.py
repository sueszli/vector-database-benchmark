from minigrid.minigrid_env import *
from minigrid.utils.rendering import *
from minigrid.core.world_object import WorldObj

class Ball(WorldObj):

    def __init__(self, color='blue'):
        if False:
            return 10
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        if False:
            print('Hello World!')
        return False

    def render(self, img):
        if False:
            while True:
                i = 10
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class AppleKeyToDoorTreasure(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, grid_size=19, apple=2):
        if False:
            print('Hello World!')
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.apple = apple
        mission_space = MissionSpace(mission_func=lambda : 'Reach the goal')
        super().__init__(mission_space=mission_space, grid_size=grid_size, max_steps=100)

    def _gen_grid(self, width, height):
        if False:
            i = 10
            return i + 15
        self.grid = Grid(width, height)
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)
        room_w = width // 2
        room_h = height // 2
        for j in range(0, 2):
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h
                if i + 1 < 2:
                    if j + 1 < 2:
                        self.grid.vert_wall(xR, yT, room_h)
                    else:
                        self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB))
                        self.grid.set(*pos, None)
                if j + 1 < 2:
                    if i + 1 < 2:
                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.grid.set(*pos, None)
                    else:
                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.put_obj(Door('yellow', is_locked=True), *pos)
        pos1 = (self._rand_int(room_w + 1, 2 * room_w), self._rand_int(room_h + 1, 2 * room_h))
        self.put_obj(Key('yellow'), *pos1)
        pos2_dummy_list = []
        for i in range(self.apple):
            pos2 = (self._rand_int(1, room_w), self._rand_int(1, room_h))
            while pos2 in pos2_dummy_list:
                pos2 = (self._rand_int(1, room_w), self._rand_int(1, room_h))
            self.put_obj(Ball('red'), *pos2)
            pos2_dummy_list.append(pos2)
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            (goal.init_pos, goal.cur_pos) = self._goal_default_pos
        else:
            self.place_obj(Goal())
        self.mission = 'Reach the goal'

    def _reward_ball(self):
        if False:
            i = 10
            return i + 15
        '\n        Compute the reward to be given upon finding the apple\n        '
        return 1

    def _reward_goal(self):
        if False:
            i = 10
            return i + 15
        '\n        Compute the reward to be given upon success\n        '
        return 10

    def step(self, action):
        if False:
            return 10
        self.step_count += 1
        reward = 0
        done = False
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward_goal()
            if fwd_cell != None and fwd_cell.type == 'ball':
                reward = self._reward_ball()
                self.grid.set(*fwd_pos, None)
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)
        elif action == self.actions.done:
            pass
        else:
            assert False, 'unknown action'
        if self.step_count >= self.max_steps:
            done = True
        obs = self.gen_obs()
        return (obs, reward, done, done, {})

class AppleKeyToDoorTreasure_13x13(AppleKeyToDoorTreasure):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(agent_pos=(2, 8), goal_pos=(7, 1), grid_size=13, apple=2)

class AppleKeyToDoorTreasure_19x19(AppleKeyToDoorTreasure):

    def __init__(self):
        if False:
            return 10
        super().__init__(agent_pos=(2, 14), goal_pos=(10, 1), grid_size=19, apple=2)

class AppleKeyToDoorTreasure_13x13_1(AppleKeyToDoorTreasure):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(agent_pos=(2, 8), goal_pos=(7, 1), grid_size=13, apple=1)

class AppleKeyToDoorTreasure_7x7_1(AppleKeyToDoorTreasure):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__(agent_pos=(1, 5), goal_pos=(4, 1), grid_size=7, apple=1)

class AppleKeyToDoorTreasure_19x19_3(AppleKeyToDoorTreasure):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__(agent_pos=(2, 14), goal_pos=(10, 1), grid_size=19, apple=3)
if __name__ == '__main__':
    AppleKeyToDoorTreasure()._gen_grid(13, 13)