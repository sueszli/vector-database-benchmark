from __future__ import print_function
import os
import logging
from miasm.analysis.sandbox import Sandbox_Win_x86_32
from miasm.jitter.loader.pe import vm2pe, ImpRecStrategy
from miasm.core.locationdb import LocationDB
from miasm.jitter.jitload import JitterException
parser = Sandbox_Win_x86_32.parser(description='Generic & dummy unpacker')
parser.add_argument('filename', help='PE Filename')
parser.add_argument('--oep', help='Stop and dump if this address is reached')
parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
options = parser.parse_args()
loc_db = LocationDB()
sb = Sandbox_Win_x86_32(loc_db, options.filename, options, globals(), parse_reloc=False)
if options.verbose is True:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)
if options.verbose is True:
    print(sb.jitter.vm)

def stop(jitter):
    if False:
        print('Hello World!')
    logging.info('User provided OEP reached')
    return False
if options.oep:
    sb.jitter.add_breakpoint(int(options.oep, 0), stop)
try:
    sb.run()
except (JitterException, ValueError) as e:
    logging.exception(e)
out_fname = '%s.dump' % options.filename
imprec = ImpRecStrategy(sb.jitter, sb.libs, 32)
imprec.recover_import()
print('Dump to %s' % out_fname)
vm2pe(sb.jitter, out_fname, libs=sb.libs, e_orig=sb.pe)