import argparse
import os
import sys
from typing import Set
ZULIP_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ZULIP_PATH)
from scripts.lib.zulip_tools import get_environment, get_recent_deployments, parse_cache_script_args, purge_unused_caches
ENV = get_environment()
NODE_MODULES_CACHE_PATH = '/srv/zulip-npm-cache'

def get_caches_in_use(threshold_days: int) -> Set[str]:
    if False:
        for i in range(10):
            print('nop')
    setups_to_check = {ZULIP_PATH}
    caches_in_use = set()
    if ENV == 'prod':
        setups_to_check |= get_recent_deployments(threshold_days)
    if ENV == 'dev':
        CURRENT_CACHE = os.path.dirname(os.path.realpath(os.path.join(ZULIP_PATH, 'node_modules')))
        caches_in_use.add(CURRENT_CACHE)
    for setup_dir in setups_to_check:
        node_modules_link_path = os.path.join(setup_dir, 'node_modules')
        if not os.path.islink(node_modules_link_path):
            continue
        caches_in_use.add(os.path.dirname(os.readlink(node_modules_link_path)))
    return caches_in_use

def main(args: argparse.Namespace) -> None:
    if False:
        for i in range(10):
            print('nop')
    caches_in_use = get_caches_in_use(args.threshold_days)
    purge_unused_caches(NODE_MODULES_CACHE_PATH, caches_in_use, 'node modules cache', args)
if __name__ == '__main__':
    args = parse_cache_script_args('This script cleans unused Zulip npm caches.')
    main(args)