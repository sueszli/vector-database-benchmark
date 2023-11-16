import numpy as np

class SetPoints:

    def __init__(self, state, nDesired=0, eDesired=0, dDesired=0, yDesired=0, args=None):
        if False:
            return 10
        "\n        Example uses:\n        1. SP = SetPoints(state='Trajectory', nDesired=-10, eDesired=40, dDesired=10, yDesired=0)\n        2. SP = SetPoints(state='Step')\n        3. SP = SetPoints(state='Wave', args='Y')\n            * Complete args list = 'Y', 'RP', 'T'\n            * Must replace:\n                rollControl, pitchControl, yawControl, thrustControl, landState = C.positionControl(actualPos, desiredPos)\n               with\n                rollControl = desiredPos[1]; pitchControl = desiredPos[0]; yawControl = desiredPos[3]; thrustControl = desiredPos[2]; \n        "
        self.args = args
        self.northDesired = nDesired
        self.eastDesired = eDesired
        self.downDesired = dDesired
        self.yawDesired = yDesired
        self.northDesiredList = []
        self.eastDesiredList = []
        self.downDesiredList = []
        self.yawDesiredList = []
        self.index = 0
        if state not in ['Trajectory', 'Step', 'Wave']:
            print('Selected state does not exist. Use one of the following:                   \n\t Trajectory                   \n\t Step                         \n\t Wave')
            exit()
        else:
            self.state = state

    def reset(self, nDesired, eDesired, dDesired, yDesired):
        if False:
            return 10
        self.northDesired = nDesired
        self.eastDesired = eDesired
        self.downDesired = dDesired
        self.yawDesired = yDesired

    def update(self, posIC, velIC, accIC):
        if False:
            return 10
        self.northDesiredList = []
        self.eastDesiredList = []
        self.downDesiredList = []
        self.yawDesiredList = []
        self.index = 0
        if self.state == 'Trajectory':
            self.createTrajectory(posIC, velIC, accIC)
        elif self.state == 'Step':
            self.createStep(posIC)
        elif self.state == 'Wave':
            self.downDesired = 0.5
            self.createWave(axis=self.args)
        else:
            print('State setpoint error')

    def getDesired(self):
        if False:
            i = 10
            return i + 15
        if self.index >= len(self.northDesiredList):
            northSP = self.northDesired
        else:
            northSP = self.northDesiredList[self.index]
        if self.index >= len(self.eastDesiredList):
            eastSP = self.eastDesired
        else:
            eastSP = self.eastDesiredList[self.index]
        if self.index >= len(self.downDesiredList):
            downSP = self.downDesired
        else:
            downSP = self.downDesiredList[self.index]
        if self.index >= len(self.yawDesiredList):
            yawSP = self.yawDesired
        else:
            yawSP = self.yawDesiredList[self.index]
        self.index += 1
        return [northSP, eastSP, downSP, yawSP]

    def createTrajectory(self, posIC, velIC, accIC):
        if False:
            while True:
                i = 10
        self.northDesiredList = self.trajectoryGen(posIC[0], velIC[0], accIC[0], self.northDesired, T=4)
        self.eastDesiredList = self.trajectoryGen(posIC[1], velIC[1], accIC[1], self.eastDesired, T=4)
        self.downDesiredList = self.trajectoryGen(posIC[2], velIC[2], accIC[2], self.downDesired, T=8)

    def trajectoryGen(self, pos0, vel0, acc0, endPos, T, sampleRate=1 / 30):
        if False:
            for i in range(10):
                print('nop')
        tt = np.linspace(0, T, round(T / sampleRate), endpoint=True)
        A = np.array([[0, 0, 0, 0, 0, 1], [np.power(T, 5), np.power(T, 4), np.power(T, 3), np.power(T, 2), T, 1], [0, 0, 0, 0, 1, 0], [5 * np.power(T, 4), 4 * np.power(T, 3), 3 * np.power(T, 2), 2 * T, 1, 0], [0, 0, 0, 2, 0, 0], [20 * np.power(T, 3), 12 * np.power(T, 2), 6 * T, 2, 0, 0]])
        b = np.array([pos0, endPos, 0, 0, 0, 0])
        x = np.linalg.solve(A, b)
        A = x[0]
        B = x[1]
        C = x[2]
        D = x[3]
        E = x[4]
        F = x[5]
        pos = A * np.power(tt, 5) + B * np.power(tt, 4) + C * np.power(tt, 3) + D * np.power(tt, 2) + E * tt + F
        return pos.tolist()

    def createStep(self, posIC, sampleRate=1 / 30):
        if False:
            for i in range(10):
                print('nop')
        n = int(1.0 / sampleRate)
        self.northDesiredList = [posIC[0]] * n
        self.eastDesiredList = [posIC[1]]
        self.downDesiredList = [posIC[2]]
        self.northDesired = posIC[0] + 30
        self.eastDesired = posIC[1]
        self.downDesired = posIC[2]

    def createWave(self, axis):
        if False:
            while True:
                i = 10
        if axis == 'Y':
            self.northDesired = 0
            self.eastDesired = 0
            self.downDesired = 0.5
            self.yawDesiredList = self.sineWaveGenerator(30)
        elif axis == 'RP':
            self.northDesiredList = self.sineWaveGenerator(3)
            self.eastDesiredList = self.sineWaveGenerator(3)
            self.downDesired = 0.5
            self.yawDesired = 0
        elif axis == 'T':
            self.northDesired = 0
            self.eastDesired = 0
            self.downDesiredList = self.sineWaveGenerator(A=0.05, b=0.5)
            self.yawDesired = 0
        else:
            print('Error in selected state')

    def sineWaveGenerator(self, A, b=0, T=5, sampleRate=1 / 30, plotFlag=False):
        if False:
            for i in range(10):
                print('nop')
        x = np.linspace(0, T, round(T / sampleRate), endpoint=True)
        f = 30
        fs = 30
        y = A * np.sin(2 * np.pi * f * (x / fs)) + b
        if plotFlag is True:
            import matplotlib.pyplot as plt
            plt.plot(x, y)
            plt.show()
        return y

    def dampedSineWaveGenerator(self, A, b=0, T=6, sampleRate=1 / 30, plotFlag=False):
        if False:
            for i in range(10):
                print('nop')
        x = np.linspace(0, T / 3, round(T / 3 / sampleRate), endpoint=True)
        xx = np.linspace(0, T, round(T / sampleRate), endpoint=True)
        f = 30
        fs = 30
        y1 = np.exp(-x) * A * np.sin(2 * np.pi * f * (x / fs)) + b
        y2 = A * np.sin(2 * np.pi * f * (x / fs)) + b
        y3 = np.concatenate((np.zeros(1), -np.flip(y1), y2[1:-1], y1, np.zeros(1)), axis=0)
        if plotFlag is True:
            import matplotlib.pyplot as plt
            plt.plot(xx, y3)
            plt.show()
        return y3

def main():
    if False:
        i = 10
        return i + 15
    SP = SetPoints('Wave')
    SP.dampedSineWaveGenerator(A=10, plotFlag=True)
if __name__ == '__main__':
    main()