import math

class RungeKutta(object):

    def __init__(self, functions, initConditions, t0, dh, save=True):
        if False:
            print('Hello World!')
        (self.Trajectory, self.save) = ([[t0] + initConditions], save)
        self.functions = [lambda *args: 1.0] + list(functions)
        (self.N, self.dh) = (len(self.functions), dh)
        self.coeff = [1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0]
        self.InArgCoeff = [0.0, 0.5, 0.5, 1.0]

    def iterate(self):
        if False:
            for i in range(10):
                print('nop')
        step = self.Trajectory[-1][:]
        (istep, iac) = (step[:], self.InArgCoeff)
        (k, ktmp) = (self.N * [0.0], self.N * [0.0])
        for (ic, c) in enumerate(self.coeff):
            for (if_, f) in enumerate(self.functions):
                arguments = [x + k[i] * iac[ic] for (i, x) in enumerate(istep)]
                try:
                    feval = f(*arguments)
                except OverflowError:
                    return False
                if abs(feval) > 100.0:
                    return False
                ktmp[if_] = self.dh * feval
            k = ktmp[:]
            step = [s + c * k[ik] for (ik, s) in enumerate(step)]
        if self.save:
            self.Trajectory += [step]
        else:
            self.Trajectory = [step]
        return True

    def solve(self, finishtime):
        if False:
            print('Hello World!')
        while self.Trajectory[-1][0] < finishtime:
            if not self.iterate():
                break

    def solveNSteps(self, nSteps):
        if False:
            while True:
                i = 10
        for i in range(nSteps):
            if not self.iterate():
                break

    def series(self):
        if False:
            for i in range(10):
                print('nop')
        return zip(*self.Trajectory)
sysSM = (lambda *a: 41.0 / 96.0 / math.pi ** 2 * a[1] ** 3, lambda *a: -19.0 / 96.0 / math.pi ** 2 * a[2] ** 3, lambda *a: -42.0 / 96.0 / math.pi ** 2 * a[3] ** 3, lambda *a: 1.0 / 16.0 / math.pi ** 2 * (9.0 / 2.0 * a[4] ** 3 - 8.0 * a[3] ** 2 * a[4] - 9.0 / 4.0 * a[2] ** 2 * a[4] - 17.0 / 12.0 * a[1] ** 2 * a[4]), lambda *a: 1.0 / 16.0 / math.pi ** 2 * (24.0 * a[5] ** 2 + 12.0 * a[4] ** 2 * a[5] - 9.0 * a[5] * (a[2] ** 2 + 1.0 / 3.0 * a[1] ** 2) - 6.0 * a[4] ** 4 + 9.0 / 8.0 * a[2] ** 4 + 3.0 / 8.0 * a[1] ** 4 + 3.0 / 4.0 * a[2] ** 2 * a[1] ** 2))

def drange(start, stop, step):
    if False:
        return 10
    r = start
    while r < stop:
        yield r
        r += step

def phaseDiagram(system, trajStart, trajPlot, h=0.1, tend=1.0, range=1.0):
    if False:
        print('Hello World!')
    tstart = 0.0
    for i in drange(0, range, 0.1 * range):
        for j in drange(0, range, 0.1 * range):
            rk = RungeKutta(system, trajStart(i, j), tstart, h)
            rk.solve(tend)
            for tr in rk.Trajectory:
                (x, y) = trajPlot(tr)
                print(x, y)
            print()
            continue
            l = (len(rk.Trajectory) - 1) / 3
            if l > 0 and 2 * l < len(rk.Trajectory):
                p1 = rk.Trajectory[l]
                p2 = rk.Trajectory[2 * l]
                (x1, y1) = trajPlot(p1)
                (x2, y2) = trajPlot(p2)
                dx = -0.5 * (y2 - y1)
                dy = 0.5 * (x2 - x1)
                print(x1 + dx, y1 + dy)
                print(x2, y2)
                print(x1 - dx, y1 - dy)
                print()

def singleTraj(system, trajStart, h=0.02, tend=1.0):
    if False:
        while True:
            i = 10
    tstart = 0.0
    rk = RungeKutta(system, trajStart, tstart, h)
    rk.solve(tend)
    for i in range(len(rk.Trajectory)):
        tr = rk.Trajectory[i]
        print(' '.join(['{:.4f}'.format(t) for t in tr]))
singleTraj(sysSM, [0.354, 0.654, 1.278, 0.983, 0.131], h=0.5, tend=math.log(10 ** 17))