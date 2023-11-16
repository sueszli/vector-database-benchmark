"""
## referenced https://www.kaggle.com/eugenkeil/simple-baseline-bot by @eugenkeil

## referenced https://www.kaggle.com/david1013/tunable-baseline-bot by @david1013

"""
from kaggle_environments.envs.football.helpers import *
from math import sqrt
from enum import Enum
import random
import torch
import torch.nn as nn
import numpy as np
from ding.torch_utils import tensor_to_list, one_hot, to_ndarray
from ding.utils import MODEL_REGISTRY
from ding.torch_utils import to_tensor, to_dtype
'\nReadable Reminder\n*********************\nclass Action(Enum):\n    Idle = 0\n    Left = 1\n    TopLeft = 2\n    Top = 3\n    TopRight = 4\n    Right = 5\n    BottomRight = 6\n    Bottom = 7\n    BottomLeft = 8\n    LongPass= 9\n    HighPass = 10\n    ShortPass = 11\n    Shot = 12\n    Sprint = 13\n    ReleaseDirection = 14\n    ReleaseSprint = 15\n    Slide = 16\n    Dribble = 17\n    ReleaseDribble = 18\n\n\nsticky_index_to_action = [\n    Action.Left,\n    Action.TopLeft,\n    Action.Top,\n    Action.TopRight,\n    Action.Right,\n    Action.BottomRight,\n    Action.Bottom,\n    Action.BottomLeft,\n    Action.Sprint,\n    Action.Dribble\n]\n\n\nclass PlayerRole(Enum):\n    GoalKeeper = 0\n    CenterBack = 1\n    LeftBack = 2\n    RightBack = 3\n    DefenceMidfield = 4\n    CentralMidfield = 5\n    LeftMidfield = 6\n    RIghtMidfield = 7\n    AttackMidfield = 8\n    CentralFront = 9\n\n\nclass GameMode(Enum):\n    Normal = 0\n    KickOff = 1\n    GoalKick = 2\n    FreeKick = 3\n    Corner = 4\n    ThrowIn = 5\n    Penalty = 6\n'

class Stiuation(Enum):
    Delaying = 0
    Offencing = 1
    Deffencing = 2

class Line(object):

    def __init__(self, pos1, pos2):
        if False:
            while True:
                i = 10
        self.a = 1
        (x1, y1) = pos1
        (x2, y2) = pos2
        if y2 - y1 != 0.0:
            self.b = (x2 - x1) / (y2 - y1)
        else:
            self.b = 100000.0
        self.c = -x1 - self.b * y2
        self.length = dist(pos1, pos2)

    def distToLine(self, pos):
        if False:
            print('Hello World!')
        return (self.a * pos[0] + self.b * pos[1] + self.c) / sqrt(self.a ** 2 + self.b ** 2)
roles = [0, 7, 9, 2, 1, 1, 3, 5, 5, 5, 6]
passes = [Action.ShortPass, Action.LongPass, Action.HighPass]
offenseScore = {0: [-8.0, 0.0], 1: [0.6, 0.8], 2: [0.6, 0.85], 3: [0.6, 0.85], 4: [0.7, 0.9], 5: [0.8, 0.9], 6: [1, 1], 7: [1, 1], 8: [1, 1.1], 9: [1.1, 1.2]}
passBias = 2.0
defenceThreatDist = 0.3
threatAvg = 3.0
shotDistAbs = 0.03
shotDistFactor = 0.6
offenseGoalDistFactor = 3.0
offenseKeeperDistFactor = 0.5
offenseTirenessFactor = 0.3
sprintTirenessFactor = 0.5
passForShotFactor = 0.6
FREEKICK_SHOT_AREA = [[0.5, 1], [-0.2, 0.2]]
START_SHOT_AREA1 = [[0.6, 0.75], [-0.2, 0.2]]
START_SHOT_AREA2 = [[0.75, 0.95], [-0.13, 0.13]]
PASS_FOR_SHOT_AREA1 = [[0.75, 1], [-0.42, -0.18]]
PASS_FOR_SHOT_AREA2 = [[0.75, 1], [0.18, 0.42]]
KEEPER_ZONE_AREA = [[0.75, 1], [-0.2, 0.2]]
LONG_SHOT_RANGE_AREA = [[0.5, 1], [-0.25, 0.25]]
SPRINT_AREA = [[-0.1, 0.6], [-0.42, 0.42]]
DEFENCE_SPRING_AREA = [[-0.7, 0.4], [-0.4, 0.4]]
SLIDE_AREA = [[-0.65, 0], [-0.42, 0.42]]
takenSelfFactor = 0.5
passFactors = {Action.HighPass: [1.0, 1.2, 3.0], Action.ShortPass: [1.1, 1.5, 1.5], Action.LongPass: [1.0, 1.2, 2]}

def dist(pos1, pos2):
    if False:
        return 10
    return sqrt((pos1[1] - pos2[1]) ** 2 + (pos1[0] - pos2[0]) ** 2)

def dirSign(x):
    if False:
        i = 10
        return i + 15
    if abs(x) < 0.01:
        return 1
    elif x < 0:
        return 0
    return 2

def plusPos(pos1, pos2):
    if False:
        while True:
            i = 10
    return [pos1[0] + pos2[0], pos1[1] + pos2[1]]

def vec2dir(vec):
    if False:
        return 10
    p = sqrt(vec[0] ** 2 + vec[1] ** 2)
    coef = 1 / p
    return [vec[0] * coef, vec[1] * coef]
TOTAL_STEP = 3000
directions = [[Action.TopLeft, Action.Top, Action.TopRight], [Action.Left, Action.Idle, Action.Right], [Action.BottomLeft, Action.Bottom, Action.BottomRight]]

def insideArea(pos, area):
    if False:
        return 10
    return area[0][0] <= pos[0] <= area[0][1] and area[1][0] <= pos[1] <= area[1][1]

def gotoDir(x, y):
    if False:
        i = 10
        return i + 15
    xdir = dirSign(x)
    ydir = dirSign(y)
    return directions[ydir][xdir]

class Processer(object):

    def __init__(self):
        if False:
            return 10
        self._obs = {}
        self._curPos = None
        self._keeperPos = None
        self._goalPos = [1, 0]
        self._shot_dir_ready = False
        self._pass_dir_ready = False
        self._ball_is_free = False
        self._we_have_ball = False
        self._enemy_have_ball = False
        self._our_goalkeeper_have_ball = False
        self._shot_buf_player = None
        self._shot_buf_step = -1
        self._pass_buf_player = None
        self._pass_buf_step = -1
        self._score_diff = 0
        self._pass_type = Action.ShortPass

    def preprocess(self):
        if False:
            return 10
        self._game_mode = self._obs['game_mode']
        self._cur_player = self._obs['active']
        if self._obs['score'].shape[0] == 2:
            self._score_diff = self._obs['score'][0] - self._obs['score'][1]
        else:
            self._score_diff = self._obs['score']
        self._curPos = self._obs['left_team'][self._obs['active']]
        self._curDir = self._obs['left_team_direction'][self._obs['active']]
        self._keeperPos = self._obs['right_team'][0]
        self._ballPos = self._obs['ball']
        self._ourPos = self._obs['left_team']
        self._enemyPos = self._obs['right_team']
        self._ball_is_free = self._obs['ball_owned_team'] == -1
        self._we_have_ball = self._obs['ball_owned_team'] == 0
        self._enemy_have_ball = self._obs['ball_owned_team'] == 1
        self._our_goalkeeper_have_ball = self._obs['ball_owned_player'] == 0 and self._we_have_ball
        self._our_active_have_ball = self._we_have_ball and self._obs['ball_owned_player'] == self._obs['active']
        self._controlled_role = self._obs['left_team_roles'][self._obs['active']]
        self._most_foward_enemy_pos = self.getMostForwardEnemyPos()
        self._closest_enemey_pos = self.getClosestEnemyPos()
        self._closest_enemey_to_cur_vec = [self._curPos[0] - self._closest_enemey_pos[0], self._curPos[1] - self._closest_enemey_pos[1]]
        self._closest_enemey_to_cur_dir = vec2dir(self._closest_enemey_to_cur_vec)
        self._cloest_enemey_dist = dist(self._curPos, self._closest_enemey_pos)
        self._remain_step = self._obs['steps_left']
        self._cur_tireness = self._obs['left_team_tired_factor'][self._obs['active']]
        self._our_tireness = self._obs['left_team_tired_factor']
        self._dribbling = Action.Dribble in self._obs['sticky_actions']
        self._sprinting = Action.Sprint in self._obs['sticky_actions']
        self._our_goalkeeper_active = self._cur_player == 0
        self._ball_dir = self._obs['ball_direction']
        self._ball_owner_dir = self.getBallOwnerDir()
        self._ball_owner_pos = self.getBallOwnerPos()
        if self._enemy_have_ball:
            (self._closest_to_enemy_pos, self._closest_to_enemy_player) = self.getClosestToEnemy()
        if not self._shot_dir_ready:
            self._shot_buf_player = -1

    def getRole(self, i):
        if False:
            while True:
                i = 10
        return roles[i]

    def getBallOwnerPos(self):
        if False:
            print('Hello World!')
        if self._ball_is_free:
            return None
        elif self._we_have_ball:
            return self._obs['left_team'][self._obs['ball_owned_player']]
        else:
            return self._obs['right_team'][self._obs['ball_owned_player']]

    def getBallOwnerDir(self):
        if False:
            print('Hello World!')
        if self._ball_is_free:
            return None
        elif self._we_have_ball:
            return self._obs['left_team_direction'][self._obs['ball_owned_player']]
        else:
            return self._obs['right_team_direction'][self._obs['ball_owned_player']]

    def gobetweenKeeperGate(self):
        if False:
            print('Hello World!')
        xdir = dirSign(self._keeperPos[0] / 2 + self._goalPos[0] / 2 - self._curPos[0] - 0.05)
        ydir = dirSign(self._keeperPos[1] / 2 + self._goalPos[1] / 2 - self._curPos[1])
        return directions[ydir][xdir]

    def gotoDst(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        xdir = dirSign(x - self._curPos[0])
        ydir = dirSign(y - self._curPos[1])
        return directions[ydir][xdir]

    def getMostForwardEnemyPos(self):
        if False:
            for i in range(10):
                print('nop')
        ret = [0, 0]
        i = 0
        for pos in self._obs['right_team']:
            if i == 0:
                i += 1
                continue
            if pos[0] > ret[0]:
                ret = pos
        return ret

    def getAvgDefenceDistToPlayer(self, *args):
        if False:
            i = 10
            return i + 15
        if len(args) == 0:
            i = self._cur_player
        else:
            i = args[0]
        sumDist = 0
        for pos in self._enemyPos:
            if dist(pos, self._ourPos[i]) < defenceThreatDist:
                sumDist += dist(pos, self._ourPos[i])
        return sumDist / threatAvg

    def getClosestEnemy(self, *args):
        if False:
            while True:
                i = 10
        if len(args) == 0:
            i = self._cur_player
        else:
            i = args[0]
        closest_pos = self._keeperPos
        closest_index = 0
        index = 0
        closest_dist = 2
        for pos in self._obs['right_team']:
            if dist(pos, self._ourPos[i]) < dist(self._ourPos[i], closest_pos):
                closest_pos = pos
                closest_index = index
                closest_dist = dist(pos, self._ourPos[i])
            index += 1
        return [closest_pos, closest_index, closest_dist]

    def getClosestEnemyPos(self, *args):
        if False:
            i = 10
            return i + 15
        if len(args) == 0:
            i = self._cur_player
        else:
            i = args[0]
        return self.getClosestEnemy(i)[0]

    def getClosestEnemyDist(self, *args):
        if False:
            print('Hello World!')
        if len(args) == 0:
            i = self._cur_player
        else:
            i = args[0]
        return self.getClosestEnemy(i)[2]

    def should_sprint(self):
        if False:
            print('Hello World!')
        if self._cur_tireness * sprintTirenessFactor > (TOTAL_STEP - self._remain_step) / TOTAL_STEP + 0.2:
            return False
        if self._enemy_have_ball:
            return insideArea(self._curPos, DEFENCE_SPRING_AREA)
        if self._we_have_ball:
            return insideArea(self._curPos, SPRINT_AREA)

    def shotWill(self):
        if False:
            while True:
                i = 10
        if insideArea(self._curPos, START_SHOT_AREA1) or insideArea(self._curPos, START_SHOT_AREA2):
            return True
        elif not insideArea(self._keeperPos, KEEPER_ZONE_AREA) and insideArea(self._curPos, LONG_SHOT_RANGE_AREA):
            return True
        if dist(self._curPos, self._keeperPos) < shotDistFactor * dist(self._keeperPos, self._goalPos) + shotDistAbs:
            return True
        return False

    def getClosestToEnemy(self):
        if False:
            i = 10
            return i + 15
        retpos = self._obs['left_team'][0]
        index = 0
        retindex = index
        for pos in self._obs['left_team']:
            if dist(pos, self._ball_owner_pos) < dist(retpos, self._ball_owner_pos):
                retpos = pos
                retindex = index
            index += 1
        return (retpos, retindex)

    def getMinxLeftTeam(self):
        if False:
            return 10
        i = 0
        retpos = [1, 0]
        for pos in self._ourPos:
            if i == 0:
                i += 1
                continue
            if pos[0] < retpos[0]:
                retpos = pos
        return retpos

    def should_slide(self):
        if False:
            return 10
        if not self._enemy_have_ball:
            return False
        if self._curPos[0] < self._ball_owner_pos[0] - 0.01 and self._curPos[0] < self._ballPos[0] - 0.007 and (dist(self._curPos, self._ball_owner_pos) < 0.03) and (self._curDir[0] < 0) and insideArea(self._curPos, SLIDE_AREA) and True:
            return True
        return False

    def should_chase(self):
        if False:
            while True:
                i = 10
        if self._curPos[0] > self._ball_owner_pos[0] + 0.02 and self._curPos[0] != self._closest_to_enemy_pos[0]:
            return False
        minLeftTeamPos = self.getMinxLeftTeam()
        if self._curPos[0] > self._ball_owner_pos[0] + 0.03 and self._ball_owner_pos[0] - minLeftTeamPos[0] > 1.5 * abs(self._ball_owner_pos[1] - minLeftTeamPos[1]):
            return False
        return True

    def shotAway(self):
        if False:
            for i in range(10):
                print('nop')
        return False
        if self._curPos[0] < -0.7 and self._our_active_have_ball:
            return True
        return False

    def judgeOffside(self, *args):
        if False:
            print('Hello World!')
        if len(args) == 0:
            LeftTeam = 0
            for pos in self._obs['left_team']:
                LeftTeam = max(LeftTeam, pos[0])
        else:
            LeftTeam = self._ourPos[args[0]][0]
        maxRightTeam = self.getMostForwardEnemyPos()[0]
        return LeftTeam > maxRightTeam

    def passWill(self):
        if False:
            i = 10
            return i + 15
        curOffenceMark = self.offenseMark(self._cur_player)
        (bestPassMark, bestPassType, bestPassIndex) = self.getBestPass()
        if bestPassMark > curOffenceMark + passBias:
            return (True, bestPassType, bestPassIndex)
        else:
            return (False, Action.ShortPass, -1)

    def getBestPass(self):
        if False:
            while True:
                i = 10
        if not self._our_active_have_ball:
            return (-1, Action.ShortPass, -1)
        bestPassType = Action.ShortPass
        bestPassIndex = -1
        bestPassMark = -10
        for index in range(11):
            if index == self._cur_player:
                continue
            (passMark, passType) = self.passMarkTo(index)
            if passMark > bestPassMark:
                bestPassMark = passMark
                bestPassType = passType
                bestPassIndex = index
        return (bestPassMark, bestPassType, bestPassIndex)

    def passMarkTo(self, i):
        if False:
            print('Hello World!')
        bestPassType = Action.ShortPass
        bestPassMark = -10
        for t in passes:
            if self.getPassSuccessMark(i, t) + self.offenseMark(i) > bestPassMark:
                bestPassType = t
                bestPassMark = self.getPassSuccessMark(i, t) + self.offenseMark(i)
        return (bestPassMark, bestPassType)

    def getRoleOffenceScore(self, i):
        if False:
            return 10
        r = roles[i]
        (adder, multier) = offenseScore[r]
        return (adder, multier)

    def offenseMark(self, i):
        if False:
            print('Hello World!')
        mark = 0.0
        mark += self.getClosestEnemyDist(i)
        mark += self.getAvgDefenceDistToPlayer(i)
        mark += 3.0 / (dist(self._ourPos[i], self._goalPos) + 0.2)
        mark -= 0.5 / (dist(self._ourPos[i], self._keeperPos) + 0.2)
        (adder, multier) = self.getRoleOffenceScore(i)
        mark *= multier
        mark += adder
        mark += 1.0 - self._our_tireness[i] * offenseTirenessFactor
        if insideArea(self._ourPos[i], PASS_FOR_SHOT_AREA1) or insideArea(self._ourPos[i], PASS_FOR_SHOT_AREA2):
            mark = mark * passForShotFactor
        return mark

    def getPassSuccessMark(self, i, passType):
        if False:
            for i in range(10):
                print('nop')
        if i == self._cur_player:
            return -10
        if self.judgeOffside(i):
            return -10
        mark = 0.0
        interceptFactor = passFactors[passType][0]
        distFactor = passFactors[passType][1]
        takenFactor = passFactors[passType][2]
        l = Line(self._curPos, self._ourPos[i])
        minDist = 2
        for pos in self._enemyPos:
            minDist = min(minDist, l.distToLine(pos))
        mark += minDist * interceptFactor
        taken = self.getClosestEnemyDist(i) + takenSelfFactor * self.getClosestEnemyDist()
        mark += taken * takenFactor
        mark += l.length * distFactor
        return mark

    def shotFreeKick(self):
        if False:
            print('Hello World!')
        if insideArea(self._curPos, FREEKICK_SHOT_AREA):
            return True
        return False

    def cutAngleWithClosest(self):
        if False:
            i = 10
            return i + 15
        x = self._keeperPos[0] / 2 + self._goalPos[0] / 2 - self._curPos[0]
        y = self._keeperPos[1] / 2 + self._goalPos[1] / 2 - self._curPos[1]
        x += self._closest_enemey_to_cur_dir[0] * (0.05 / (self._cloest_enemey_dist + 0.03))
        y += self._closest_enemey_to_cur_dir[1] * (0.05 / (self._cloest_enemey_dist + 0.03))
        return gotoDir(x, y)

    def process(self, obs):
        if False:
            i = 10
            return i + 15
        self._obs = obs
        self.preprocess()
        if self._game_mode == GameMode.Penalty:
            return Action.Shot
        if self._game_mode == GameMode.Corner:
            if self._pass_dir_ready:
                return self._pass_type
            (bestPassMark, bestPassType, bestPassIndex) = self.getBestPass()
            self._pass_dir_ready = True
            self._pass_type = bestPassType
            return self.gotoDst(self._ourPos[bestPassIndex][0], self._ourPos[bestPassIndex][1])
        if self._game_mode == GameMode.FreeKick:
            if self.shotFreeKick():
                return Action.Shot
            else:
                if self._pass_dir_ready:
                    return self._pass_type
                (bestPassMark, bestPassType, bestPassIndex) = self.getBestPass()
                self._pass_dir_ready = True
                self._pass_type = bestPassType
                return self.gotoDst(self._ourPos[bestPassIndex][0], self._ourPos[bestPassIndex][1])
        if self._game_mode == GameMode.KickOff:
            return Action.ShortPass
        if self._game_mode == GameMode.ThrowIn:
            if self._pass_dir_ready:
                return self._pass_type
            (bestPassMark, bestPassType, bestPassIndex) = self.getBestPass()
            self._pass_dir_ready = True
            self._pass_type = bestPassType
            return self.gotoDst(self._ourPos[bestPassIndex][0], self._ourPos[bestPassIndex][1])
        if self._our_active_have_ball and (not self._our_goalkeeper_have_ball):
            if self._shot_dir_ready and self._cur_player == self._shot_buf_player and (self._remain_step == self._shot_buf_step - 1):
                self._shot_dir_ready = False
                self._shot_buf_player = -1
                self._shot_buf_step = -1
                return Action.Shot
            if self.shotWill():
                self._shot_buf_player = self._cur_player
                self._shot_buf_step = self._remain_step
                self._shot_dir_ready = True
                return self.gobetweenKeeperGate()
            if self._pass_dir_ready and self._cur_player == self._pass_buf_player and (self._remain_step == self._pass_buf_step - 1):
                self._pass_dir_ready = False
                self._pass_buf_player = -1
                self._pass_buf_step = -1
                return self._pass_type
            else:
                self._shot_dir_ready = False
                self._pass_dir_ready = False
                (doPass, doPassType, doPassIndex) = self.passWill()
                if doPass:
                    self._pass_dir_ready = True
                    self._pass_type = doPassType
                    self._pass_buf_step = self._remain_step
                    self._pass_buf_player = self._cur_player
                    return self.gotoDst(self._ourPos[doPassIndex][0], self._ourPos[doPassIndex][1])
                if self._closest_enemey_to_cur_vec[0] > 0:
                    if not self._sprinting and self.should_sprint():
                        return Action.Sprint
                    if self._dribbling and dist(self._curPos, self._closest_enemey_pos) > 0.02:
                        return Action.ReleaseDribble
                    return self.gobetweenKeeperGate()
                elif dist(self._curPos, self._closest_enemey_pos) < 0.02:
                    return self.cutAngleWithClosest()
                else:
                    if self._dribbling:
                        return Action.ReleaseDribble
                    if not self._sprinting:
                        return Action.Sprint
                    return self.gobetweenKeeperGate()
        elif self._we_have_ball and (not self._our_goalkeeper_have_ball) and (not self._our_active_have_ball):
            self._shot_dir_ready = False
            return self.gotoDst(self._goalPos[0], self._goalPos[1])
        elif self._our_goalkeeper_have_ball:
            self._shot_dir_ready = False
            if self._our_goalkeeper_active:
                return Action.HighPass
            if self._sprinting:
                return Action.ReleaseSprint
            return self.gobetweenKeeperGate()
        self._shot_dir_ready = False
        if self._dribbling:
            return Action.ReleaseDribble
        if self._ball_is_free:
            if not self._sprinting and self.should_sprint():
                return Action.Sprint
            return self.gotoDst(self._ballPos[0] + 2 * self._ball_dir[0], self._ballPos[1] + 2 * self._ball_dir[1])
        if self._enemy_have_ball:
            '\n            if not self.should_chase():\n                if self._sprinting:\n                    return Action.ReleaseSprint\n                return Action.Idle\n            if self.should_slide():\n                return Action.Slide\n            '
            if not self._sprinting and self.should_sprint() and self.should_chase():
                return Action.Sprint
            return self.gotoDst(self._ballPos[0] + 1 * self._ball_dir[0] + 1 * self._ball_owner_dir[0], self._ballPos[1] + 1 * self._ball_dir[1] + 1 * self._ball_owner_dir[1])
        return self.gotoDst(self._goalPos[0], self._goalPos[1])
processer = Processer()

def agent(obs):
    if False:
        i = 10
        return i + 15
    global processer
    return processer.process(obs)

def raw_obs_to_readable(obs):
    if False:
        while True:
            i = 10
    obs['sticky_actions'] = {sticky_index_to_action[nr] for (nr, action) in enumerate(obs['sticky_actions']) if action}
    obs['game_mode'] = GameMode(obs['game_mode'])
    if 'designated' in obs:
        del obs['designated']
    obs['left_team_roles'] = [PlayerRole(role) for role in obs['left_team_roles']]
    obs['right_team_roles'] = [PlayerRole(role) for role in obs['right_team_roles']]
    return obs

def rule_agent(obs):
    if False:
        for i in range(10):
            print('nop')
    obs = raw_obs_to_readable(obs)
    return agent(obs).value

def idel_agent(obs):
    if False:
        for i in range(10):
            print('nop')
    return 0

def random_agent(obs):
    if False:
        return 10
    return random.randint(0, 18)
agents_map = {'random': random_agent, 'rule': rule_agent, 'idel': idel_agent}

@MODEL_REGISTRY.register('football_rule')
class FootballRuleBaseModel(torch.nn.Module):

    def __init__(self, cfg={}):
        if False:
            print('Hello World!')
        super(FootballRuleBaseModel, self).__init__()
        self.agent_type = cfg.get('agent_type', 'rule')
        self._agent = agents_map[self.agent_type]
        self._dummy_param = nn.Parameter(torch.zeros(1, 1))

    def forward(self, data):
        if False:
            print('Hello World!')
        actions = []
        data = data['raw_obs']
        if isinstance(data['score'], list):
            data['score'] = torch.stack(data['score'], dim=-1)
        data = [{k: v[i] for (k, v) in data.items()} for i in range(data['left_team'].shape[0])]
        for d in data:
            if isinstance(d['steps_left'], torch.Tensor):
                d = {k: v.cpu() for (k, v) in d.items()}
                d = to_ndarray(d)
                for k in ['active', 'designated', 'ball_owned_player', 'ball_owned_team']:
                    d[k] = int(d[k])
                actions.append(self._agent(d))
        return {'action': torch.LongTensor(actions), 'logit': one_hot(torch.LongTensor(actions), 19)}