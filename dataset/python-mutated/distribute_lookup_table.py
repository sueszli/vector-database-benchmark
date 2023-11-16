LOOKUP_TABLE_TYPE = 'lookup_table'

def find_distributed_lookup_table_inputs(program, table_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find input variable of distribute lookup table in program.\n    We only support one distribute table now.\n    Args:\n    program(Program): given program, locate distributed lookup table\n    table_name(str): given table name that is found beforehand\n    Returns:\n    inputs\n    '
    local_vars = program.current_block().vars
    inputs = []
    for op in program.global_block().ops:
        if op.type == LOOKUP_TABLE_TYPE:
            if table_name == op.input('W')[0]:
                inputs.extend([local_vars[name] for name in op.input('Ids')])
    return inputs

def find_distributed_lookup_table_outputs(program, table_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find output variable of distribute lookup table in program.\n    We only support one distribute table now.\n    Args:\n    program(Program): given program, locate distributed lookup table\n    table_name(str): given table name that is found beforehand\n    Returns:\n    outputs\n    '
    local_vars = program.current_block().vars
    outputs = []
    for op in program.global_block().ops:
        if op.type == LOOKUP_TABLE_TYPE:
            if table_name == op.input('W')[0]:
                outputs.extend([local_vars[name] for name in op.output('Out')])
    return outputs

def find_distributed_lookup_table(program):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find distribute lookup table in program.\n    We only support one distribute table now.\n    Args:\n    program(Program): given program, locate distributed lookup table\n    Returns:\n    table_name or None\n    '
    table_name = None
    for op in program.global_block().ops:
        if op.type == LOOKUP_TABLE_TYPE:
            if op.attr('is_distributed') is True:
                if table_name is None:
                    table_name = op.input('W')[0]
                if table_name != op.input('W')[0]:
                    raise RuntimeError('all distributed lookup_table_ops should have only one table')
            elif table_name is not None:
                assert op.input('W')[0] != table_name
    return table_name