import argparse
import numpy as np

class Agent(object):

    def __init__(self, agentNum: int, IL: int, AO: int, AS: int, c_h: float, c_p: float, eta: int, compuType: str, config: argparse.Namespace) -> None:
        if False:
            while True:
                i = 10
        self.agentNum = agentNum
        self.IL = IL
        self.OO = 0
        self.ASInitial = AS
        self.ILInitial = IL
        self.AOInitial = AO
        self.config = config
        self.curState = []
        self.nextState = []
        self.curReward = 0
        self.cumReward = 0
        self.totRew = 0
        self.c_h = c_h
        self.c_p = c_p
        self.eta = eta
        self.AS = np.zeros((1, 1))
        self.AO = np.zeros((1, 1))
        self.action = 0
        self.compType = compuType
        self.alpha_b = self.config.alpha_b[self.agentNum]
        self.betta_b = self.config.betta_b[self.agentNum]
        if self.config.demandDistribution == 0:
            self.a_b = np.mean((self.config.demandUp, self.config.demandLow))
            self.b_b = np.mean((self.config.demandUp, self.config.demandLow)) * (np.mean((self.config.leadRecItemLow[self.agentNum], self.config.leadRecItemUp[self.agentNum])) + np.mean((self.config.leadRecOrderLow[self.agentNum], self.config.leadRecOrderUp[self.agentNum])))
        elif self.config.demandDistribution == 1 or self.config.demandDistribution == 3 or self.config.demandDistribution == 4:
            self.a_b = self.config.demandMu
            self.b_b = self.config.demandMu * (np.mean((self.config.leadRecItemLow[self.agentNum], self.config.leadRecItemUp[self.agentNum])) + np.mean((self.config.leadRecOrderLow[self.agentNum], self.config.leadRecOrderUp[self.agentNum])))
        elif self.config.demandDistribution == 2:
            self.a_b = 8
            self.b_b = 3 / 4.0 * 8 * (np.mean((self.config.leadRecItemLow[self.agentNum], self.config.leadRecItemUp[self.agentNum])) + np.mean((self.config.leadRecOrderLow[self.agentNum], self.config.leadRecOrderUp[self.agentNum])))
        elif self.config.demandDistribution == 3:
            self.a_b = 10
            self.b_b = 7 * (np.mean((self.config.leadRecItemLow[self.agentNum], self.config.leadRecItemUp[self.agentNum])) + np.mean((self.config.leadRecOrderLow[self.agentNum], self.config.leadRecOrderUp[self.agentNum])))
        else:
            raise Exception('The demand distribution is not defined or it is not a valid type.!')
        self.hist = []
        self.hist2 = []
        self.srdqnBaseStock = []
        self.T = 0
        self.bsBaseStock = 0
        self.init_bsBaseStock = 0
        self.nextObservation = []
        if self.compType == 'srdqn':
            self.currentState = np.stack([self.curState for _ in range(self.config.multPerdInpt)], axis=0)

    def resetPlayer(self, T: int):
        if False:
            return 10
        self.IL = self.ILInitial
        self.OO = 0
        self.AS = np.squeeze(np.zeros((1, T + max(self.config.leadRecItemUp) + max(self.config.leadRecOrderUp) + 10)))
        self.AO = np.squeeze(np.zeros((1, T + max(self.config.leadRecItemUp) + max(self.config.leadRecOrderUp) + 10)))
        if self.agentNum != 0:
            for i in range(self.config.leadRecOrderUp_aux[self.agentNum - 1]):
                self.AO[i] = self.AOInitial[self.agentNum - 1]
        for i in range(self.config.leadRecItemUp[self.agentNum]):
            self.AS[i] = self.ASInitial
        self.curReward = 0
        self.cumReward = 0
        self.action = []
        self.hist = []
        self.hist2 = []
        self.srdqnBaseStock = []
        self.T = T
        self.curObservation = self.getCurState(1)
        self.nextObservation = []
        if self.compType == 'srdqn':
            self.currentState = np.stack([self.curObservation for _ in range(self.config.multPerdInpt)], axis=0)

    def recieveItems(self, time: int) -> None:
        if False:
            print('Hello World!')
        self.IL = self.IL + self.AS[time]
        self.OO = self.OO - self.AS[time]

    def actionValue(self, curTime: int) -> int:
        if False:
            while True:
                i = 10
        if self.config.fixedAction:
            a = self.config.actionList[np.argmax(self.action)]
        elif self.compType == 'srdqn':
            a = max(0, self.config.actionList[np.argmax(self.action)] * self.config.action_step + self.AO[curTime])
        elif self.compType == 'rnd':
            a = max(0, self.config.actionList[np.argmax(self.action)] + self.AO[curTime])
        else:
            a = max(0, self.config.actionListOpt[np.argmax(self.action)])
        return a

    def getReward(self) -> None:
        if False:
            print('Hello World!')
        self.curReward = (self.c_p * max(0, -self.IL) + self.c_h * max(0, self.IL)) / 200.0
        self.curReward = -self.curReward
        self.cumReward = self.config.gamma * self.cumReward + self.curReward

    def getCurState(self, t: int) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        if self.config.ifUseASAO:
            if self.config.if_use_AS_t_plus_1:
                curState = np.array([-1 * (self.IL < 0) * self.IL, 1 * (self.IL > 0) * self.IL, self.OO, self.AS[t], self.AO[t]])
            else:
                curState = np.array([-1 * (self.IL < 0) * self.IL, 1 * (self.IL > 0) * self.IL, self.OO, self.AS[t - 1], self.AO[t]])
        else:
            curState = np.array([-1 * (self.IL < 0) * self.IL, 1 * (self.IL > 0) * self.IL, self.OO])
        if self.config.ifUseActionInD:
            a = self.config.actionList[np.argmax(self.action)]
            curState = np.concatenate((curState, np.array([a])))
        return curState