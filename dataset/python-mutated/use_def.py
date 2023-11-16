from manticore.core.plugin import Plugin
from manticore.ethereum import ManticoreEVM
m = ManticoreEVM()
source_code = '\npragma solidity ^0.4;\ncontract C {\n    uint c;\n    bool enabled;\n    bool i;\n    function C() public {\n        c =0;\n        enabled = false;\n        i = false;\n\n    }\n    function f1() public {\n        c+=1;\n    }\n    function f2() public {\n        if(c>100)\n            enabled=true;\n\n    }\n    function f3() public{\n        if (!enabled)\n            return;\n        i = true;\n\n    }\n}\n'
print(source_code)

class EVMUseDef(Plugin):

    def did_evm_write_storage_callback(self, state, address, offset, value):
        if False:
            return 10
        m = self.manticore
        world = state.platform
        tx = world.all_transactions[-1]
        md = m.get_metadata(tx.address)
        if md:
            offsets = state.solve_n(offset, 3000)
            with self.locked_context('storage_writes', dict) as storage_writes:
                contract_function = (md.name, md.get_func_name(state.solve_one(tx.data[0:4])))
                if contract_function not in storage_writes:
                    storage_writes[contract_function] = set()
                for off in offsets:
                    storage_writes[contract_function].add(off)

    def did_evm_read_storage_callback(self, state, address, offset, value):
        if False:
            while True:
                i = 10
        m = self.manticore
        world = state.platform
        tx = world.all_transactions[-1]
        md = m.get_metadata(tx.address)
        if md:
            offsets = state.solve_n(offset, 3000)
            with self.locked_context('storage_reads', dict) as storage_reads:
                contract_function = (md.name, md.get_func_name(state.solve_one(tx.data[0:4])))
                if contract_function not in storage_reads:
                    storage_reads[contract_function] = set()
                for off in offsets:
                    storage_reads[contract_function].add(off)
p = EVMUseDef()
m.register_plugin(p)
user_account = m.create_account(balance=1000)
contract_account = m.solidity_create_contract(source_code, owner=user_account)
symbolic_data = m.make_symbolic_buffer(320)
symbolic_value = m.make_symbolic_value()
m.transaction(caller=user_account, address=contract_account, value=symbolic_value, data=symbolic_data)
print('READS', p.context['storage_reads'])
print('WRITES', p.context['storage_writes'])
print('It makes no sense to try f3() after 1 tx')
m.finalize()
print(f'[+] Look for results in {m.workspace}')