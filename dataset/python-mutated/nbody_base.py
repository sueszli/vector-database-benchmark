"""
N-body benchmark from the Computer Language Benchmarks Game.

This is intended to support Unladen Swallow's pyperf.py. Accordingly, it has been
modified from the Shootout version:
- Accept standard Unladen Swallow benchmark options.
- Run report_energy()/advance() in a loop.
- Reimplement itertools.combinations() to work with older Python versions.

Pulled from:
http://benchmarksgame.alioth.debian.org/u64q/program.php?test=nbody&lang=python3&id=1

Contributed by Kevin Carson.
Modified by Tupteq, Fredrik Johansson, and Daniel Nanz.
"""
__contact__ = 'collinwinter@google.com (Collin Winter)'
DEFAULT_ITERATIONS = 20000
DEFAULT_REFERENCE = 'sun'

def combinations(l):
    if False:
        while True:
            i = 10
    'Pure-Python implementation of itertools.combinations(l, 2).'
    result = []
    for x in range(len(l) - 1):
        ls = l[x + 1:]
        for y in ls:
            result.append((l[x], y))
    return result
PI = 3.141592653589793
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24
BODIES = {'sun': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], SOLAR_MASS), 'jupiter': ([4.841431442464721, -1.1603200440274284, -0.10362204447112311], [0.001660076642744037 * DAYS_PER_YEAR, 0.007699011184197404 * DAYS_PER_YEAR, -6.90460016972063e-05 * DAYS_PER_YEAR], 0.0009547919384243266 * SOLAR_MASS), 'saturn': ([8.34336671824458, 4.124798564124305, -0.4035234171143214], [-0.002767425107268624 * DAYS_PER_YEAR, 0.004998528012349172 * DAYS_PER_YEAR, 2.3041729757376393e-05 * DAYS_PER_YEAR], 0.0002858859806661308 * SOLAR_MASS), 'uranus': ([12.894369562139131, -15.111151401698631, -0.22330757889265573], [0.002964601375647616 * DAYS_PER_YEAR, 0.0023784717395948095 * DAYS_PER_YEAR, -2.9658956854023756e-05 * DAYS_PER_YEAR], 4.366244043351563e-05 * SOLAR_MASS), 'neptune': ([15.379697114850917, -25.919314609987964, 0.17925877295037118], [0.0026806777249038932 * DAYS_PER_YEAR, 0.001628241700382423 * DAYS_PER_YEAR, -9.515922545197159e-05 * DAYS_PER_YEAR], 5.1513890204661145e-05 * SOLAR_MASS)}
SYSTEM = list(BODIES.values())
PAIRS = combinations(SYSTEM)

def advance(dt, n, bodies=SYSTEM, pairs=PAIRS):
    if False:
        return 10
    for i in range(n):
        for (([x1, y1, z1], v1, m1), ([x2, y2, z2], v2, m2)) in pairs:
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            mag = dt * (dx * dx + dy * dy + dz * dz) ** (-1.5)
            b1m = m1 * mag
            b2m = m2 * mag
            v1[0] -= dx * b2m
            v1[1] -= dy * b2m
            v1[2] -= dz * b2m
            v2[0] += dx * b1m
            v2[1] += dy * b1m
            v2[2] += dz * b1m
        for (r, [vx, vy, vz], m) in bodies:
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz

def report_energy(bodies=SYSTEM, pairs=PAIRS, e=0.0):
    if False:
        for i in range(10):
            print('nop')
    for (((x1, y1, z1), v1, m1), ((x2, y2, z2), v2, m2)) in pairs:
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        e -= m1 * m2 / (dx * dx + dy * dy + dz * dz) ** 0.5
    for (r, [vx, vy, vz], m) in bodies:
        e += m * (vx * vx + vy * vy + vz * vz) / 2.0
    return e

def offset_momentum(ref, bodies=SYSTEM, px=0.0, py=0.0, pz=0.0):
    if False:
        i = 10
        return i + 15
    for (r, [vx, vy, vz], m) in bodies:
        px -= vx * m
        py -= vy * m
        pz -= vz * m
    (r, v, m) = ref
    v[0] = px / m
    v[1] = py / m
    v[2] = pz / m

def bench_nbody(loops, reference, iterations):
    if False:
        for i in range(10):
            print('nop')
    offset_momentum(BODIES[reference])
    range_it = range(loops)
    for _ in range_it:
        report_energy()
        advance(0.01, iterations)
        report_energy()

def add_cmdline_args(cmd, args):
    if False:
        for i in range(10):
            print('nop')
    cmd.extend(('--iterations', str(args.iterations)))

def run_benchmark():
    if False:
        for i in range(10):
            print('nop')
    bench_nbody(1, DEFAULT_REFERENCE, DEFAULT_ITERATIONS)
if __name__ == '__main__':
    run_benchmark()