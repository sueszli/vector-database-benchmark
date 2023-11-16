"""
This script generates VMTests that are used to check EVM's Istanbul fork correctness.

### TO GENERATE ALL:

## Initialize env:
cd manticore/tests/ && mkdir -p  ethereum_vm/VMTests_concrete 
git clone https://github.com/ethereum/tests --depth=1

## Get help
python make_VMTest.py --help 

## Generate concrete tests:
for i in  tests/BlockchainTests/ValidBlocks/VMTests/*/*json; do python make_VMTest.py -i $i --fork Istanbul -o ethereum_vm/VMTests_concrete; done

"""
import argparse
import sys
import logging
import os
import json
import pyevmasm as EVMAsm
from binascii import unhexlify
total_count = 0
DEFAULT_FORK = 'Istanbul'
real_open = open

def fake_open(filename, mode='r', *args, **kwargs):
    if False:
        return 10
    'Replace normal global open with this for a wuick dry run'
    from io import StringIO
    logging.info('Fake openning %r', (filename, mode) + args)
    if os.path.exists(filename):
        return StringIO(real_open(filename, 'r').read())
    return StringIO()

def get_caller(nonce, price, gas, address, value, calldata, v, r, s):
    if False:
        return 10
    if address is None:
        to = b''
    else:
        to = unhexlify('%040x' % address)
    from eth.vm.forks.frontier.transactions import FrontierTransaction
    t = FrontierTransaction(nonce=nonce, gas_price=price, gas=gas, to=to, value=value, data=calldata, v=v, r=r, s=s)
    return int.from_bytes(t.sender, 'big')

def gen_header(testcases):
    if False:
        while True:
            i = 10
    header = f'"""DO NOT MODIFY: Tests generated from `tests/` with {sys.argv[0]}"""\nimport unittest\nfrom binascii import unhexlify\nfrom manticore import ManticoreEVM, Plugin\nfrom manticore.utils import config\n'
    if any(('logs' in testcase for testcase in testcases.values())):
        body += "\nimport sha3\nimport rlp\nfrom rlp.sedes import (\n    CountableList,\n    BigEndianInt,\n    Binary,\n)\nclass Log(rlp.Serializable):\n    fields = [\n        ('address', Binary.fixed_length(20, allow_empty=True)),\n        ('topics', CountableList(BigEndianInt(32))),\n        ('data', Binary())\n    ]\n"
    header += "consts = config.get_group('core')\nconsts.mprocessing = consts.mprocessing.single\nconsts = config.get_group('evm')\nconsts.oog = 'pedantic'\n\nclass EVMTest(unittest.TestCase):\n    # https://nose.readthedocs.io/en/latest/doc_tests/test_multiprocess/multiprocess.html#controlling-distribution\n    _multiprocess_can_split_ = True\n    # https://docs.python.org/3.7/library/unittest.html#unittest.TestCase.maxDiff\n    maxDiff = None\n\n"
    return header

def gen_footer(testcase):
    if False:
        return 10
    footer = "\n\nif __name__ == '__main__':\n    unittest.main()"
    return footer

def gen_body(name, testcase):
    if False:
        i = 10
        return i + 15
    body = f'''\n    def test_{name}(self):\n        """\n        Testcase taken from https://github.com/ethereum/tests\n        Source: {testcase['_info']['source']} \n        """\n        class UsedGas(Plugin):\n            @property\n            def used_gas(self):\n                with self.locked_context() as ctx:\n                    return ctx['test_used_gas']\n            @used_gas.setter\n            def used_gas(self, value):\n                with self.locked_context() as ctx:\n                    ctx['test_used_gas']=value\n\n            def did_close_transaction_callback(self, state, tx):\n                if tx.is_human:\n                    self.used_gas = tx.used_gas\n    \n        used_gas_plugin = UsedGas()\n        m = ManticoreEVM(workspace_url="mem:", plugins=(used_gas_plugin,))\n\n'''
    for (address, account) in testcase['pre'].items():
        account_address = int(address, 0)
        account_code = account['code'][2:]
        account_nonce = int(account['nonce'], 0)
        account_balance = int(account['balance'], 0)
        disassembly = EVMAsm.disassemble(unhexlify(account_code), fork=DEFAULT_FORK.lower())
        disassembly = '\n        """' + '\n            ' + '\n            '.join(disassembly.split('\n')) + '\n        """'
        body += f"""\n        {(disassembly if account_code else '')}\n        m.create_account(address={hex(account_address)},\n                         balance={account_balance}, \n                         code={("unhexlify('" + account_code + "')" if account_code else "b''")}, \n                         nonce={account_nonce})"""
        if 'storage' in account and account['storage']:
            body += '\n        for state in m.all_states:\n            world = state.platform'
        for (key, value) in account['storage'].items():
            body += f'\n            world.set_storage_data({hex(account_address)}, {key}, {value})'
    coinbases = set()
    for block in testcase['blocks']:
        blockheader = block['blockHeader']
        coinbases.add(blockheader['coinbase'])
    for coinbase in coinbases:
        body += f"\n        #coinbase\n        m.create_account(address={coinbase},\n                         balance=0, \n                         code=b'', \n                         nonce=0)\n        "
    for block in testcase['blocks']:
        blockheader = block['blockHeader']
        body += f"\n        # Start a block\n        self.assertEqual(m.count_all_states(), 1)\n        m.start_block(blocknumber={blockheader['number']},\n                      timestamp={blockheader['timestamp']},\n                      difficulty={blockheader['difficulty']},\n                      coinbase={blockheader['coinbase']},\n                      gaslimit={hex(int(blockheader['gasLimit'], 0))})\n\n        #VMtest Transaction\n"
        for transaction in block['transactions']:
            address = None if transaction['to'] == '' else int(transaction['to'], 16)
            calldata = unhexlify(transaction['data'][2:])
            gas = int(transaction['gasLimit'], 0)
            price = int(transaction['gasPrice'], 0)
            nonce = int(transaction['nonce'], 0)
            value = 0 if transaction['value'] == '0x' else int(transaction['value'], 0)
            r = int(transaction['r'], 0)
            s = int(transaction['s'], 0)
            v = int(transaction['v'], 0)
            caller = get_caller(nonce, price, gas, address, value, calldata, v, r, s)
            body += f'\n\n        m.transaction(caller={hex(caller)},\n                      address={hex(address)},\n                      value={value},\n                      data={calldata},\n                      gas={gas},\n                      price={price})'
    body += f"\n        for state in m.all_states:\n            world = state.platform\n            self.assertEqual(used_gas_plugin.used_gas, {blockheader['gasUsed']})\n            \n            world.end_block()"
    for (account_address, account) in testcase['postState'].items():
        body += f"""\n            # Add post checks for account {account_address}\n            # check nonce, balance, code and storage values\n            self.assertEqual(world.get_nonce({account_address}), {account['nonce']})\n            self.assertEqual(world.get_balance({account_address}), {account['balance']})\n            self.assertEqual(world.get_code({account_address}), {("unhexlify('" + account['code'][2:] + "')" if account['code'][2:] else "b''")})"""
        if account['storage']:
            body += '\n            # check storage'
            for (key, value) in account['storage'].items():
                body += f'\n            self.assertEqual(world.get_storage_data({account_address}, {key}), {value})'
    if 'logs' in testcase:
        print(testcase['logs'])
        body += f"\n            # check logs\n            logs = [Log(unhexlify('{'{'}:040x{'}'}'.format(l.address)), l.topics, solve(l.memlog)) for l in world.logs]\n            data = rlp.encode(logs)\n            self.assertEqual(sha3.keccak_256(data).hexdigest(), '{testcase['logs'][2:]}')"
    return body

def gen_testfile(testcases, fork):
    if False:
        print('Hello World!')
    global total_count
    output = gen_header(testcases)
    for (name, testcase) in testcases.items():
        if testcase['network'] != fork:
            logging.warning(f"Skipping testcase {name}. Wrong fork: {testcase['network']} != {fork}")
            continue
        total_count += 1
        output += gen_body(name.replace('-', '_'), testcase)
    output += gen_footer(testcases)
    return output
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manticore test generator for Ethereum BlockchainTests')
    parser.add_argument('-v', action='count', default=0, help='Specify verbosity level from -v to -vvvv')
    parser.add_argument('-f', '--fork', '--flavor', default=DEFAULT_FORK, type=str, help='Fork, default: byzantium. Possible: Byzantium, Constantinople, EIP150, EIP158, Frontier, Homestead, Istanbul.Also an unsigned block number is accepted to select the fork.')
    parser.add_argument('-d', '--dry-run', default=False, action='store_true', help='Do not generate any file')
    parser.add_argument('-i', '--input-path', nargs='?', help='Path to Ethereum tests', required=True)
    parser.add_argument('-r', '--filter-regex', type=str, help='Filter by regex')
    parser.add_argument('-o', '--output-path', nargs='?', default='!inplace', help='Output path, by default this generates a .py file in the same folder as the json input')
    parser.add_argument('-x', '--force', default=False, action='store_true', help='Overwrite any existing file')
    args = parser.parse_args(sys.argv[1:])
    loglevel = (logging.CRITICAL, logging.ERROR, logging.INFO, logging.DEBUG)
    logging.basicConfig(level=loglevel[min(args.v, 3)], format='%(message)s')
    accepted_forks = ['Byzantium', 'Constantinople', 'EIP150', 'EIP158', 'Frontier', 'Homestead', 'Istanbul']
    args.fork = args.fork.title()
    if args.fork not in accepted_forks:
        logging.error('Wrong fork name. Please provide one of %s.\n' % accepted_forks)
        sys.exit(1)
    if args.dry_run:
        open = fake_open
    if not os.path.isfile(args.input_path):
        logging.error('Wrong json test file (%s). Please provide one.\n' % args.input_path)
        sys.exit(1)
    with open(args.input_path) as fp:
        if not os.path.isfile(args.input_path) or not args.input_path.endswith('.json'):
            logging.debug('Input file args.input_path looks odd. Expecting a .json file.')
        testcases = dict(json.loads(fp.read()))
    logging.info(f'Loaded {len(testcases)} testcases from {args.input_path}')
    if args.output_path == '!inplace':
        (stem, ext) = os.path.splitext(args.input_path)
        args.output_path = stem + '.py'
    elif os.path.isdir(args.output_path):
        (stem, ext) = os.path.splitext(os.path.basename(args.input_path))
        output_path = os.path.join(args.output_path, f'test_{stem}.py')
        if os.path.exists(output_path):
            folders = args.input_path.split(os.sep)
            if len(folders) >= 2:
                output_path = os.path.join(args.output_path, f'test_{folders[-2]}_{stem}.py')
        args.output_path = output_path
    if os.path.exists(args.output_path):
        if not args.force:
            logging.error(f'File {args.output_path} already exists. Consider adding --force')
            if not args.dry_run:
                sys.exit(1)
            logging.error(f'Continuing because it is a dry run. ')
        else:
            logging.info(f'File {args.output_path} already exists. Overwritting.')
    with open(args.output_path, 'w') as fp:
        fp.write(gen_testfile(testcases, args.fork))
    logging.info(f'{total_count} unittests generated in {args.output_path}')