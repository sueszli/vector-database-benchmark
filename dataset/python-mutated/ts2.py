from __future__ import print_function
import os

def main():
    if False:
        print('Hello World!')
    'Examples: Multiple-choice Vector Bin Packing'
    from pyvpsolver.solvers import mvpsolver
    os.chdir(os.path.dirname(__file__) or os.curdir)
    test = (3333, 3333, 3333)
    dev = (3333, 3333, 3333)
    train = (10000000000.0, 10000000000.0, 10000000000.0)
    Ws = [test, dev, train]
    Cs = [1, 1, 1]
    Qs = [1, 1, 1]
    (ws1, b1) = ([(50, 25, 20)], 1)
    b = [b1]
    ws = [ws1]
    solution = mvpsolver.solve(Ws, Cs, Qs, ws, b, svg_file='tmp/graphA_mvbp.svg', script='vpsolver_glpk.sh', verbose=True)
    mvpsolver.print_solution(solution)
    (obj, patterns) = solution
    assert obj == 1
if __name__ == '__main__':
    main()