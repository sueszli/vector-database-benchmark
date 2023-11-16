from manticore.ethereum import Detector, ManticoreEVM
m = ManticoreEVM()
m.verbosity(3)
source_code = '\npragma solidity ^0.4.21;\n\ncontract MappingChallenge {\n    bool public isComplete;\n    uint256[] map;\n\n    function set(uint256 key, uint256 value) public payable {\n        // Expand dynamic array as needed\n        if (map.length <= key) {\n            map.length = key + 1;\n        }\n\n        map[key] = value;\n    }\n}\n'
print('Source code:\n', source_code)

class StopAtDepth(Detector):
    """This just aborts explorations that are too deep"""

    def will_run_callback(self, *args):
        if False:
            for i in range(10):
                print('nop')
        with self.manticore.locked_context('seen_rep', dict) as reps:
            reps.clear()

    def will_decode_instruction_callback(self, state, pc):
        if False:
            return 10
        world = state.platform
        with self.manticore.locked_context('seen_rep', dict) as reps:
            item = (world.current_transaction.sort == 'CREATE', world.current_transaction.address, pc)
            if not item in reps:
                reps[item] = 0
            reps[item] += 1
            if reps[item] > 2:
                state.abandon()
m.register_plugin(StopAtDepth())
owner_account = m.create_account(balance=1000)
user_account = m.create_account(balance=1000)
target_account = m.create_account(balance=1000)
contract_account = m.solidity_create_contract(source_code, owner=user_account)
contract_account.set(m.make_symbolic_value(name='A'), 1)
contract_account.set(m.make_symbolic_value(name='B'), 1)
for st in m.all_states:
    flag_storage_slot = 0
    flag_value = st.platform.get_storage_data(contract_account.address, flag_storage_slot)
    if st.can_be_true(flag_value != 0):
        print('Flag Found! Check ', m.workspace)
        st.constraints.add(flag_value != 0)
        m.generate_testcase(st, 'Flag Found')