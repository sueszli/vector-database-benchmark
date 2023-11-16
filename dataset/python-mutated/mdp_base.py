import collections
from collections import defaultdict
from fractions import Fraction

def topoSort(roots, getParents):
    if False:
        while True:
            i = 10
    'Return a topological sorting of nodes in a graph.\n\n    roots - list of root nodes to search from\n    getParents - function which returns the parents of a given node\n    '
    results = []
    visited = set()
    stack = [(node, 0) for node in roots]
    while stack:
        (current, state) = stack.pop()
        if state == 0:
            if current not in visited:
                visited.add(current)
                stack.append((current, 1))
                stack.extend(((parent, 0) for parent in getParents(current)))
        else:
            assert current in visited
            results.append(current)
    return results

def getDamages(L, A, D, B, stab, te):
    if False:
        for i in range(10):
            print('nop')
    x = 2 * L // 5
    x = (x + 2) * A * B // (D * 50) + 2
    if stab:
        x += x // 2
    x = int(x * te)
    return [x * z // 255 for z in range(217, 256)]

def getCritDist(L, p, A1, A2, D1, D2, B, stab, te):
    if False:
        while True:
            i = 10
    p = min(p, Fraction(1))
    norm = getDamages(L, A1, D1, B, stab, te)
    crit = getDamages(L * 2, A2, D2, B, stab, te)
    dist = defaultdict(Fraction)
    for (mult, vals) in zip([1 - p, p], [norm, crit]):
        mult /= len(vals)
        for x in vals:
            dist[x] += mult
    return dist

def plus12(x):
    if False:
        for i in range(10):
            print('nop')
    return x + x // 8
stats_t = collections.namedtuple('stats_t', ['atk', 'df', 'speed', 'spec'])
NOMODS = stats_t(0, 0, 0, 0)
fixeddata_t = collections.namedtuple('fixeddata_t', ['maxhp', 'stats', 'lvl', 'badges', 'basespeed'])
halfstate_t = collections.namedtuple('halfstate_t', ['fixed', 'hp', 'status', 'statmods', 'stats'])

def applyHPChange(hstate, change):
    if False:
        print('Hello World!')
    hp = min(hstate.fixed.maxhp, max(0, hstate.hp + change))
    return hstate._replace(hp=hp)

def applyBadgeBoosts(badges, stats):
    if False:
        while True:
            i = 10
    return stats_t(*[plus12(x) if b else x for (x, b) in zip(stats, badges)])
attack_stats_t = collections.namedtuple('attack_stats_t', ['power', 'isspec', 'stab', 'te', 'crit'])
attack_data = {'Ember': attack_stats_t(40, True, True, 0.5, False), 'Dig': attack_stats_t(100, False, False, 1, False), 'Slash': attack_stats_t(70, False, False, 1, True), 'Water Gun': attack_stats_t(40, True, True, 2, False), 'Bubblebeam': attack_stats_t(65, True, True, 2, False)}

def _applyActionSide1(state, act):
    if False:
        i = 10
        return i + 15
    (me, them, extra) = state
    if act == 'Super Potion':
        me = applyHPChange(me, 50)
        return {(me, them, extra): Fraction(1)}
    mdata = attack_data[act]
    aind = 3 if mdata.isspec else 0
    dind = 3 if mdata.isspec else 1
    pdiv = 64 if mdata.crit else 512
    dmg_dist = getCritDist(me.fixed.lvl, Fraction(me.fixed.basespeed, pdiv), me.stats[aind], me.fixed.stats[aind], them.stats[dind], them.fixed.stats[dind], mdata.power, mdata.stab, mdata.te)
    dist = defaultdict(Fraction)
    for (dmg, p) in dmg_dist.items():
        them2 = applyHPChange(them, -dmg)
        dist[me, them2, extra] += p
    return dist

def _applyAction(state, side, act):
    if False:
        return 10
    if side == 0:
        return _applyActionSide1(state, act)
    else:
        (me, them, extra) = state
        dist = _applyActionSide1((them, me, extra), act)
        return {(k[1], k[0], k[2]): v for (k, v) in dist.items()}

class Battle(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.successors = {}
        self.min = defaultdict(float)
        self.max = defaultdict(lambda : 1.0)
        self.frozen = set()
        self.win = (4, True)
        self.loss = (4, False)
        self.max[self.loss] = 0.0
        self.min[self.win] = 1.0
        self.frozen.update([self.win, self.loss])

    def _getSuccessorsA(self, statep):
        if False:
            return 10
        (st, state) = statep
        for action in ['Dig', 'Super Potion']:
            yield (1, state, action)

    def _applyActionPair(self, state, side1, act1, side2, act2, dist, pmult):
        if False:
            while True:
                i = 10
        for (newstate, p) in _applyAction(state, side1, act1).items():
            if newstate[0].hp == 0:
                newstatep = self.loss
            elif newstate[1].hp == 0:
                newstatep = self.win
            else:
                newstatep = (2, newstate, side2, act2)
            dist[newstatep] += p * pmult

    def _getSuccessorsB(self, statep):
        if False:
            print('Hello World!')
        (st, state, action) = statep
        dist = defaultdict(Fraction)
        for (eact, p) in [('Water Gun', Fraction(64, 130)), ('Bubblebeam', Fraction(66, 130))]:
            priority1 = state[0].stats.speed + 10000 * (action == 'Super Potion')
            priority2 = state[1].stats.speed + 10000 * (action == 'X Defend')
            if priority1 > priority2:
                self._applyActionPair(state, 0, action, 1, eact, dist, p)
            elif priority1 < priority2:
                self._applyActionPair(state, 1, eact, 0, action, dist, p)
            else:
                self._applyActionPair(state, 0, action, 1, eact, dist, p / 2)
                self._applyActionPair(state, 1, eact, 0, action, dist, p / 2)
        return {k: float(p) for (k, p) in dist.items() if p > 0}

    def _getSuccessorsC(self, statep):
        if False:
            return 10
        (st, state, side, action) = statep
        dist = defaultdict(Fraction)
        for (newstate, p) in _applyAction(state, side, action).items():
            if newstate[0].hp == 0:
                newstatep = self.loss
            elif newstate[1].hp == 0:
                newstatep = self.win
            else:
                newstatep = (0, newstate)
            dist[newstatep] += p
        return {k: float(p) for (k, p) in dist.items() if p > 0}

    def getSuccessors(self, statep):
        if False:
            return 10
        try:
            return self.successors[statep]
        except KeyError:
            st = statep[0]
        if st == 0:
            result = list(self._getSuccessorsA(statep))
        else:
            if st == 1:
                dist = self._getSuccessorsB(statep)
            elif st == 2:
                dist = self._getSuccessorsC(statep)
            result = sorted(dist.items(), key=lambda t: (-t[1], t[0]))
        self.successors[statep] = result
        return result

    def getSuccessorsList(self, statep):
        if False:
            i = 10
            return i + 15
        if statep[0] == 4:
            return []
        temp = self.getSuccessors(statep)
        if statep[0] != 0:
            temp = list(zip(*temp))[0] if temp else []
        return temp

    def evaluate(self, tolerance=0.15):
        if False:
            print('Hello World!')
        badges = (1, 0, 0, 0)
        starfixed = fixeddata_t(59, stats_t(40, 44, 56, 50), 11, NOMODS, 115)
        starhalf = halfstate_t(starfixed, 59, 0, NOMODS, stats_t(40, 44, 56, 50))
        charfixed = fixeddata_t(63, stats_t(39, 34, 46, 38), 26, badges, 65)
        charhalf = halfstate_t(charfixed, 63, 0, NOMODS, applyBadgeBoosts(badges, stats_t(39, 34, 46, 38)))
        initial_state = (charhalf, starhalf, 0)
        initial_statep = (0, initial_state)
        (dmin, dmax, frozen) = (self.min, self.max, self.frozen)
        stateps = topoSort([initial_statep], self.getSuccessorsList)
        itercount = 0
        while dmax[initial_statep] - dmin[initial_statep] > tolerance:
            itercount += 1
            for sp in stateps:
                if sp in frozen:
                    continue
                if sp[0] == 0:
                    dmin[sp] = max((dmin[sp2] for sp2 in self.getSuccessors(sp)))
                    dmax[sp] = max((dmax[sp2] for sp2 in self.getSuccessors(sp)))
                else:
                    dmin[sp] = sum((dmin[sp2] * p for (sp2, p) in self.getSuccessors(sp)))
                    dmax[sp] = sum((dmax[sp2] * p for (sp2, p) in self.getSuccessors(sp)))
                if dmin[sp] >= dmax[sp]:
                    dmax[sp] = dmin[sp] = (dmin[sp] + dmax[sp]) / 2
                    frozen.add(sp)
        return (dmax[initial_statep] + dmin[initial_statep]) / 2

def bench_mdp(loops):
    if False:
        print('Hello World!')
    expected = 0.89873589887
    max_diff = 1e-06
    range_it = range(loops)
    for _ in range_it:
        result = Battle().evaluate(0.192)
    if abs(result - expected) > max_diff:
        raise Exception('invalid result: got %s, expected %s (diff: %s, max diff: %s)' % (result, expected, result - expected, max_diff))

def run_benchmark():
    if False:
        i = 10
        return i + 15
    bench_mdp(1)
if __name__ == '__main__':
    run_benchmark()