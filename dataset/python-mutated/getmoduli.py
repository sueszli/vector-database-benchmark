"""This tool's output can be used with the tool fastgcd (available
here: https://factorable.net/resources.html) to efficiently perform
the attack described in the paper "Mining your Ps and Qs: Detection of
Widespread Weak Keys in Network Devices"
(https://factorable.net/paper.html).

To do so, you need to strip the output from the information after the
moduli. A simple sed with 's# .*##' will do the trick."""
import getopt
import sys
from typing import Dict, Set, Tuple, Type, Union
import ivre.db
import ivre.keys
import ivre.utils

def main() -> None:
    if False:
        i = 10
        return i + 15
    bases: Set[Type[Union[ivre.keys.PassiveKey, ivre.keys.NmapKey]]] = set()
    try:
        (opts, _) = getopt.getopt(sys.argv[1:], 'p:h', ['passive-ssl', 'active-ssl', 'passive-ssh', 'active-ssh', 'help'])
    except getopt.GetoptError as err:
        sys.stderr.write(str(err) + '\n')
        sys.exit(-1)
    for (o, a) in opts:
        if o == '--passive-ssl':
            bases.add(ivre.keys.SSLRsaPassiveKey)
        elif o == '--active-ssl':
            bases.add(ivre.keys.SSLRsaNmapKey)
        elif o == '--passive-ssh':
            bases.add(ivre.keys.SSHRsaPassiveKey)
        elif o == '--active-ssh':
            bases.add(ivre.keys.SSHRsaNmapKey)
        elif o in ['-h', '--help']:
            sys.stdout.write('usage: %s [-h] [--passive-ssl] [--active-ssl] [--passive-ssh] [--active-ssh]\n\n' % sys.argv[0])
            sys.stdout.write(__doc__)
            sys.stdout.write('\n\n')
            sys.exit(0)
        else:
            sys.stderr.write('%r %r not understood (this is probably a bug).\n' % (o, a))
            sys.exit(-1)
    moduli: Dict[int, Set[Tuple[str, int, str]]] = {}
    if not bases:
        bases = {ivre.keys.SSLRsaPassiveKey, ivre.keys.SSLRsaNmapKey, ivre.keys.SSHRsaNmapKey, ivre.keys.SSHRsaPassiveKey}
    for base in bases:
        for key in base():
            moduli.setdefault(key.key.public_numbers().n, set()).add((key.ip, key.port, key.service))
    for (mod, used) in moduli.items():
        sys.stdout.write('%x %d %s\n' % (mod, len(used), ','.join(('%s:%d' % (rec[0], rec[1]) for rec in used))))