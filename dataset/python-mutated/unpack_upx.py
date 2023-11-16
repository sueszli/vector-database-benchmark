from __future__ import print_function
import os
import logging
from miasm.analysis.sandbox import Sandbox_Win_x86_32
from miasm.jitter.loader.pe import vm2pe
from miasm.core.locationdb import LocationDB
from miasm.os_dep.common import get_win_str_a

def kernel32_GetProcAddress(jitter):
    if False:
        return 10
    'Hook on GetProcAddress to note where UPX stores import pointers'
    (ret_ad, args) = jitter.func_args_stdcall(['libbase', 'fname'])
    dst_ad = jitter.cpu.EBX
    logging.error('EBX ' + hex(dst_ad))
    fname = args.fname if args.fname < 65536 else get_win_str_a(jitter, args.fname)
    logging.error(fname)
    ad = sb.libs.lib_get_add_func(args.libbase, fname, dst_ad)
    jitter.handle_function(ad)
    jitter.func_ret_stdcall(ret_ad, ad)
parser = Sandbox_Win_x86_32.parser(description='Generic UPX unpacker')
parser.add_argument('filename', help='PE Filename')
parser.add_argument('-v', '--verbose', help='verbose mode', action='store_true')
parser.add_argument('--graph', help='Export the CFG graph in graph.dot', action='store_true')
options = parser.parse_args()
options.load_hdr = True
loc_db = LocationDB()
sb = Sandbox_Win_x86_32(loc_db, options.filename, options, globals(), parse_reloc=False)
if options.verbose is True:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)
if options.verbose is True:
    print(sb.jitter.vm)
mdis = sb.machine.dis_engine(sb.jitter.bs, loc_db=loc_db)
mdis.dont_dis_nulstart_bloc = True
asmcfg = mdis.dis_multiblock(sb.entry_point)
leaves = list(asmcfg.get_bad_blocks())
assert len(leaves) == 1
l = leaves.pop()
logging.info(l)
end_offset = mdis.loc_db.get_location_offset(l.loc_key)
logging.info('final offset')
logging.info(hex(end_offset))
if options.graph is True:
    open('graph.dot', 'w').write(asmcfg.dot())
if options.verbose is True:
    print(sb.jitter.vm)

def stop(jitter):
    if False:
        for i in range(10):
            print('nop')
    logging.info('OEP reached')
    jitter.running = False
    return False
sb.jitter.add_breakpoint(end_offset, stop)
sb.run()
(bname, fname) = os.path.split(options.filename)
fname = os.path.join(bname, fname.replace('.', '_'))
out_fname = fname + '_unupx.bin'
vm2pe(sb.jitter, out_fname, libs=sb.libs, e_orig=sb.pe)