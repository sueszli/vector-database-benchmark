import argparse
import inspect
import logging
import re
import sys
logger = logging.getLogger()
if logger.handlers:
    console = logger.handlers[0]
else:
    console = logging.StreamHandler(sys.stderr)
    logger.addHandler(console)
console.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s'))

def _check_compatible(args_o, args_n, defaults_o, defaults_n):
    if False:
        return 10
    if len(args_o) > len(args_n):
        logger.debug('args num less then previous: %s vs %s', args_o, args_n)
        return False
    for idx in range(min(len(args_o), len(args_n))):
        if args_o[idx] != args_n[idx]:
            logger.debug("args's %d parameter diff with previous: %s vs %s", idx, args_o, args_n)
            return False
    if len(args_n) - len(defaults_n) > len(args_o) - len(defaults_o):
        logger.debug('defaults num less then previous: %s vs %s', defaults_o, defaults_n)
        return False
    for idx in range(min(len(defaults_o), len(defaults_n))):
        nidx_o = -1 - idx
        nidx_n = -1 - idx - (len(args_n) - len(args_o))
        if defaults_o[nidx_o] != defaults_n[nidx_n]:
            logger.debug("defaults's %d value diff with previous: %s vs %s", nidx_n, defaults_o, defaults_n)
            return False
    return True

def check_compatible(old_api_spec, new_api_spec):
    if False:
        return 10
    '\n    check compatible, FullArgSpec\n    '
    if not (isinstance(old_api_spec, inspect.FullArgSpec) and isinstance(new_api_spec, inspect.FullArgSpec)):
        logger.warning('new_api_spec or old_api_spec is not instance of inspect.FullArgSpec')
        return False
    return _check_compatible(old_api_spec.args, new_api_spec.args, [] if old_api_spec.defaults is None else old_api_spec.defaults, [] if new_api_spec.defaults is None else new_api_spec.defaults)

def check_compatible_str(old_api_spec_str, new_api_spec_str):
    if False:
        return 10
    patArgSpec = re.compile('args=(.*), varargs=.*defaults=(None|\\((.*)\\)), kwonlyargs=.*')
    mo_o = patArgSpec.search(old_api_spec_str)
    mo_n = patArgSpec.search(new_api_spec_str)
    if not (mo_o and mo_n):
        logger.warning('old_api_spec_str: %s', old_api_spec_str)
        logger.warning('new_api_spec_str: %s', new_api_spec_str)
        return False
    args_o = eval(mo_o.group(1))
    args_n = eval(mo_n.group(1))
    defaults_o = mo_o.group(2) if mo_o.group(3) is None else mo_o.group(3)
    defaults_n = mo_n.group(2) if mo_n.group(3) is None else mo_n.group(3)
    defaults_o = defaults_o.split(', ') if defaults_o else []
    defaults_n = defaults_n.split(', ') if defaults_n else []
    return _check_compatible(args_o, args_n, defaults_o, defaults_n)

def read_argspec_from_file(specfile):
    if False:
        return 10
    '\n    read FullArgSpec from spec file\n    '
    res_dict = {}
    patArgSpec = re.compile("^(paddle[^,]+)\\s+\\((ArgSpec.*),\\s\\(\\'document\\W*([0-9a-z]{32})")
    fullargspec_prefix = 'inspect.Full'
    for line in specfile.readlines():
        mo = patArgSpec.search(line)
        if mo and mo.group(2) != 'ArgSpec()':
            logger.debug('%s argspec: %s', mo.group(1), mo.group(2))
            try:
                res_dict[mo.group(1)] = eval(fullargspec_prefix + mo.group(2))
            except:
                res_dict[mo.group(1)] = fullargspec_prefix + mo.group(2)
    return res_dict
arguments = []

def parse_args():
    if False:
        i = 10
        return i + 15
    '\n    Parse input arguments\n    '
    global arguments
    parser = argparse.ArgumentParser(description='check api compatible across versions')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('prev', type=argparse.FileType('r'), help='the previous version (the version from develop branch)')
    parser.add_argument('post', type=argparse.FileType('r'), help='the post version (the version from PullRequest)')
    for item in arguments:
        parser.add_argument(item[0], dest=item[1], help=item[4], type=item[2], default=item[3])
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    if args.prev and args.post:
        prev_spec = read_argspec_from_file(args.prev)
        post_spec = read_argspec_from_file(args.post)
        diff_api_names = []
        for (as_post_name, as_post) in post_spec.items():
            as_prev = prev_spec.get(as_post_name)
            if as_prev is None:
                continue
            if isinstance(as_prev, str) or isinstance(as_post, str):
                as_prev_str = as_prev if isinstance(as_prev, str) else repr(as_prev)
                as_post_str = as_post if isinstance(as_post, str) else repr(as_post)
                if not check_compatible_str(as_prev_str, as_post_str):
                    diff_api_names.append(as_post_name)
            elif not check_compatible(as_prev, as_post):
                diff_api_names.append(as_post_name)
        if diff_api_names:
            print('\n'.join(diff_api_names))