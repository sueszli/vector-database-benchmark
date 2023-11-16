import argparse
import math
from typing import List, Optional, Tuple
import matplotlib
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D, axes3d
from ivre import db, utils
from ivre.types import Filter, Record

def graphhost(host: Record) -> Tuple[List[int], List[int]]:
    if False:
        return 10
    if 'ports' not in host:
        return ([], [])
    (hh, pp) = ([], [])
    addr = utils.ip2int(host['addr'])
    for p in host['ports']:
        if p.get('state_state') == 'open':
            hh.append(addr)
            pp.append(p['port'])
    return (hh, pp)

def getgraph(flt: Filter=db.db.view.flt_empty) -> Tuple[List[int], List[int]]:
    if False:
        while True:
            i = 10
    (h, p) = ([], [])
    allhosts = db.db.view.get(flt)
    for ap in allhosts:
        (hh, pp) = graphhost(ap)
        h += hh
        p += pp
    return (h, p)

def graph3d(mainflt: Filter=db.db.view.flt_empty, alertflt: Optional[Filter]=None) -> None:
    if False:
        while True:
            i = 10
    (h, p) = getgraph(flt=mainflt)
    fig = matplotlib.pyplot.figure()
    if matplotlib.__version__.startswith('0.99'):
        ax = Axes3D(fig)
    else:
        ax = fig.add_subplot(111, projection='3d')
    ax.plot([x // 65535 for x in h], [x % 65535 for x in h], [math.log(x, 10) for x in p], '.')
    if alertflt is not None:
        (h, p) = getgraph(flt=db.db.view.flt_and(mainflt, alertflt))
        if h:
            ax.plot([x // 65535 for x in h], [x % 65535 for x in h], [math.log(x, 10) for x in p], '.', c='r')
    matplotlib.pyplot.show()

def graph2d(mainflt: Filter=db.db.view.flt_empty, alertflt: Optional[Filter]=None) -> None:
    if False:
        while True:
            i = 10
    (h, p) = getgraph(flt=mainflt)
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(h, p, '.')
    if alertflt is not None:
        (h, p) = getgraph(flt=db.db.view.flt_and(mainflt, alertflt))
        if h:
            ax.semilogy(h, p, '.', c='r')
    matplotlib.pyplot.show()

def main() -> None:
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Plot scan results.', parents=[db.db.view.argparser])
    parser.add_argument('--2d', '-2', action='store_const', dest='graph', const=graph2d, default=graph3d)
    parser.add_argument('--3d', '-3', action='store_const', dest='graph', const=graph3d)
    parser.add_argument('--alert-445', action='store_const', dest='alertflt', const=db.db.view.searchxp445(), default=db.db.view.searchhttpauth())
    parser.add_argument('--alert-nfs', action='store_const', dest='alertflt', const=db.db.view.searchnfs())
    args = parser.parse_args()
    args.graph(mainflt=db.db.view.parse_args(args), alertflt=args.alertflt)