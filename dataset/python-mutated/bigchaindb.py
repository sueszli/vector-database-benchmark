"""Implementation of the `bigchaindb` command,
the command-line interface (CLI) for BigchainDB Server.
"""
import os
import logging
import argparse
import copy
import json
import sys
from bigchaindb.core import rollback
from bigchaindb.migrations.chain_migration_election import ChainMigrationElection
from bigchaindb.utils import load_node_key
from bigchaindb.common.transaction_mode_types import BROADCAST_TX_COMMIT
from bigchaindb.common.exceptions import DatabaseDoesNotExist, ValidationError
from bigchaindb.elections.vote import Vote
import bigchaindb
from bigchaindb import backend, ValidatorElection, BigchainDB
from bigchaindb.backend import schema
from bigchaindb.commands import utils
from bigchaindb.commands.utils import configure_bigchaindb, input_on_stderr
from bigchaindb.log import setup_logging
from bigchaindb.tendermint_utils import public_key_from_base64
from bigchaindb.commands.election_types import elections
from bigchaindb.version import __tm_supported_versions__
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@configure_bigchaindb
def run_show_config(args):
    if False:
        print('Hello World!')
    'Show the current configuration'
    config = copy.deepcopy(bigchaindb.config)
    del config['CONFIGURED']
    print(json.dumps(config, indent=4, sort_keys=True))

@configure_bigchaindb
def run_configure(args):
    if False:
        i = 10
        return i + 15
    'Run a script to configure the current node.'
    config_path = args.config or bigchaindb.config_utils.CONFIG_DEFAULT_PATH
    config_file_exists = False
    if config_path != '-':
        config_file_exists = os.path.exists(config_path)
    if config_file_exists and (not args.yes):
        want = input_on_stderr('Config file `{}` exists, do you want to override it? (cannot be undone) [y/N]: '.format(config_path))
        if want != 'y':
            return
    conf = copy.deepcopy(bigchaindb.config)
    print('Generating default configuration for backend {}'.format(args.backend), file=sys.stderr)
    database_keys = bigchaindb._database_keys_map[args.backend]
    conf['database'] = bigchaindb._database_map[args.backend]
    if not args.yes:
        for key in ('bind',):
            val = conf['server'][key]
            conf['server'][key] = input_on_stderr('API Server {}? (default `{}`): '.format(key, val), val)
        for key in ('scheme', 'host', 'port'):
            val = conf['wsserver'][key]
            conf['wsserver'][key] = input_on_stderr('WebSocket Server {}? (default `{}`): '.format(key, val), val)
        for key in database_keys:
            val = conf['database'][key]
            conf['database'][key] = input_on_stderr('Database {}? (default `{}`): '.format(key, val), val)
        for key in ('host', 'port'):
            val = conf['tendermint'][key]
            conf['tendermint'][key] = input_on_stderr('Tendermint {}? (default `{}`)'.format(key, val), val)
    if config_path != '-':
        bigchaindb.config_utils.write_config(conf, config_path)
    else:
        print(json.dumps(conf, indent=4, sort_keys=True))
    print('Configuration written to {}'.format(config_path), file=sys.stderr)
    print('Ready to go!', file=sys.stderr)

@configure_bigchaindb
def run_election(args):
    if False:
        for i in range(10):
            print('nop')
    'Initiate and manage elections'
    b = BigchainDB()
    globals()[f'run_election_{args.action}'](args, b)

def run_election_new(args, bigchain):
    if False:
        while True:
            i = 10
    election_type = args.election_type.replace('-', '_')
    globals()[f'run_election_new_{election_type}'](args, bigchain)

def create_new_election(sk, bigchain, election_class, data):
    if False:
        for i in range(10):
            print('nop')
    try:
        key = load_node_key(sk)
        voters = election_class.recipients(bigchain)
        election = election_class.generate([key.public_key], voters, data, None).sign([key.private_key])
        election.validate(bigchain)
    except ValidationError as e:
        logger.error(e)
        return False
    except FileNotFoundError as fd_404:
        logger.error(fd_404)
        return False
    resp = bigchain.write_transaction(election, BROADCAST_TX_COMMIT)
    if resp == (202, ''):
        logger.info('[SUCCESS] Submitted proposal with id: {}'.format(election.id))
        return election.id
    else:
        logger.error('Failed to commit election proposal')
        return False

def run_election_new_upsert_validator(args, bigchain):
    if False:
        return 10
    "Initiates an election to add/update/remove a validator to an existing BigchainDB network\n\n    :param args: dict\n        args = {\n        'public_key': the public key of the proposed peer, (str)\n        'power': the proposed validator power for the new peer, (str)\n        'node_id': the node_id of the new peer (str)\n        'sk': the path to the private key of the node calling the election (str)\n        }\n    :param bigchain: an instance of BigchainDB\n    :return: election_id or `False` in case of failure\n    "
    new_validator = {'public_key': {'value': public_key_from_base64(args.public_key), 'type': 'ed25519-base16'}, 'power': args.power, 'node_id': args.node_id}
    return create_new_election(args.sk, bigchain, ValidatorElection, new_validator)

def run_election_new_chain_migration(args, bigchain):
    if False:
        return 10
    "Initiates an election to halt block production\n\n    :param args: dict\n        args = {\n        'sk': the path to the private key of the node calling the election (str)\n        }\n    :param bigchain: an instance of BigchainDB\n    :return: election_id or `False` in case of failure\n    "
    return create_new_election(args.sk, bigchain, ChainMigrationElection, {})

def run_election_approve(args, bigchain):
    if False:
        for i in range(10):
            print('nop')
    "Approve an election\n\n    :param args: dict\n        args = {\n        'election_id': the election_id of the election (str)\n        'sk': the path to the private key of the signer (str)\n        }\n    :param bigchain: an instance of BigchainDB\n    :return: success log message or `False` in case of error\n    "
    key = load_node_key(args.sk)
    tx = bigchain.get_transaction(args.election_id)
    voting_powers = [v.amount for v in tx.outputs if key.public_key in v.public_keys]
    if len(voting_powers) > 0:
        voting_power = voting_powers[0]
    else:
        logger.error('The key you provided does not match any of the eligible voters in this election.')
        return False
    inputs = [i for i in tx.to_inputs() if key.public_key in i.owners_before]
    election_pub_key = ValidatorElection.to_public_key(tx.id)
    approval = Vote.generate(inputs, [([election_pub_key], voting_power)], tx.id).sign([key.private_key])
    approval.validate(bigchain)
    resp = bigchain.write_transaction(approval, BROADCAST_TX_COMMIT)
    if resp == (202, ''):
        logger.info('[SUCCESS] Your vote has been submitted')
        return approval.id
    else:
        logger.error('Failed to commit vote')
        return False

def run_election_show(args, bigchain):
    if False:
        i = 10
        return i + 15
    "Retrieves information about an election\n\n    :param args: dict\n        args = {\n        'election_id': the transaction_id for an election (str)\n        }\n    :param bigchain: an instance of BigchainDB\n    "
    election = bigchain.get_transaction(args.election_id)
    if not election:
        logger.error(f'No election found with election_id {args.election_id}')
        return
    response = election.show_election(bigchain)
    logger.info(response)
    return response

def _run_init():
    if False:
        i = 10
        return i + 15
    bdb = bigchaindb.BigchainDB()
    schema.init_database(connection=bdb.connection)

@configure_bigchaindb
def run_init(args):
    if False:
        print('Hello World!')
    'Initialize the database'
    _run_init()

@configure_bigchaindb
def run_drop(args):
    if False:
        print('Hello World!')
    'Drop the database'
    dbname = bigchaindb.config['database']['name']
    if not args.yes:
        response = input_on_stderr('Do you want to drop `{}` database? [y/n]: '.format(dbname))
        if response != 'y':
            return
    conn = backend.connect()
    try:
        schema.drop_database(conn, dbname)
    except DatabaseDoesNotExist:
        print("Cannot drop '{name}'. The database does not exist.".format(name=dbname), file=sys.stderr)

def run_recover(b):
    if False:
        i = 10
        return i + 15
    rollback(b)

@configure_bigchaindb
def run_start(args):
    if False:
        while True:
            i = 10
    'Start the processes to run the node'
    setup_logging()
    logger.info('BigchainDB Version %s', bigchaindb.__version__)
    run_recover(bigchaindb.lib.BigchainDB())
    if not args.skip_initialize_database:
        logger.info('Initializing database')
        _run_init()
    logger.info('Starting BigchainDB main process.')
    from bigchaindb.start import start
    start(args)

def run_tendermint_version(args):
    if False:
        print('Hello World!')
    'Show the supported Tendermint version(s)'
    supported_tm_ver = {'description': 'BigchainDB supports the following Tendermint version(s)', 'tendermint': __tm_supported_versions__}
    print(json.dumps(supported_tm_ver, indent=4, sort_keys=True))

def create_parser():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Control your BigchainDB node.', parents=[utils.base_parser])
    subparsers = parser.add_subparsers(title='Commands', dest='command')
    config_parser = subparsers.add_parser('configure', help='Prepare the config file.')
    config_parser.add_argument('backend', choices=['localmongodb'], default='localmongodb', const='localmongodb', nargs='?', help='The backend to use. It can only be "localmongodb", currently.')
    election_parser = subparsers.add_parser('election', help='Manage elections.')
    election_subparser = election_parser.add_subparsers(title='Action', dest='action')
    new_election_parser = election_subparser.add_parser('new', help='Calls a new election.')
    new_election_subparser = new_election_parser.add_subparsers(title='Election_Type', dest='election_type')
    for (name, data) in elections.items():
        args = data['args']
        generic_parser = new_election_subparser.add_parser(name, help=data['help'])
        for (arg, kwargs) in args.items():
            generic_parser.add_argument(arg, **kwargs)
    approve_election_parser = election_subparser.add_parser('approve', help='Approve the election.')
    approve_election_parser.add_argument('election_id', help='The election_id of the election.')
    approve_election_parser.add_argument('--private-key', dest='sk', required=True, help='Path to the private key of the election initiator.')
    show_election_parser = election_subparser.add_parser('show', help='Provides information about an election.')
    show_election_parser.add_argument('election_id', help='The transaction id of the election you wish to query.')
    subparsers.add_parser('show-config', help='Show the current configuration')
    subparsers.add_parser('init', help='Init the database')
    subparsers.add_parser('drop', help='Drop the database')
    start_parser = subparsers.add_parser('start', help='Start BigchainDB')
    start_parser.add_argument('--no-init', dest='skip_initialize_database', default=False, action='store_true', help='Skip database initialization')
    subparsers.add_parser('tendermint-version', help='Show the Tendermint supported versions')
    start_parser.add_argument('--experimental-parallel-validation', dest='experimental_parallel_validation', default=False, action='store_true', help='💀 EXPERIMENTAL: parallelize validation for better throughput 💀')
    return parser

def main():
    if False:
        i = 10
        return i + 15
    utils.start(create_parser(), sys.argv[1:], globals())