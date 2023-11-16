"""OpenQASM parser."""
import os
import shutil
import tempfile
import numpy as np
from ply import yacc
from . import node
from .exceptions import QasmError
from .qasmlexer import QasmLexer

class QasmParser:
    """OPENQASM Parser."""

    def __init__(self, filename):
        if False:
            for i in range(10):
                print('nop')
        'Create the parser.'
        if filename is None:
            filename = ''
        self.lexer = QasmLexer(filename)
        self.tokens = self.lexer.tokens
        self.parse_dir = tempfile.mkdtemp(prefix='qiskit')
        self.precedence = (('left', '+', '-'), ('left', '*', '/'), ('left', 'negative', 'positive'), ('right', '^'))
        self.parser = yacc.yacc(module=self, debug=False, outputdir=self.parse_dir)
        self.qasm = None
        self.parse_deb = False
        self.global_symtab = {}
        self.current_symtab = self.global_symtab
        self.symbols = []
        self.external_functions = ['sin', 'cos', 'tan', 'exp', 'ln', 'sqrt', 'acos', 'atan', 'asin']

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, *args):
        if False:
            return 10
        if os.path.exists(self.parse_dir):
            shutil.rmtree(self.parse_dir)

    def update_symtab(self, obj):
        if False:
            print('Hello World!')
        'Update a node in the symbol table.\n\n        Everything in the symtab must be a node with these attributes:\n        name - the string name of the object\n        type - the string type of the object\n        line - the source line where the type was first found\n        file - the source file where the type was first found\n        '
        if obj.name in self.current_symtab:
            prev = self.current_symtab[obj.name]
            raise QasmError('Duplicate declaration for', obj.type + " '" + obj.name + "' at line", str(obj.line) + ', file', obj.file + '.\nPrevious occurrence at line', str(prev.line) + ', file', prev.file)
        self.current_symtab[obj.name] = obj

    def verify_declared_bit(self, obj):
        if False:
            return 10
        'Verify a qubit id against the gate prototype.'
        if obj.name not in self.current_symtab:
            raise QasmError("Cannot find symbol '" + obj.name + "' in argument list for gate, line", str(obj.line), 'file', obj.file)
        sym = self.current_symtab[obj.name]
        if not (sym.type == 'id' and sym.is_bit):
            raise QasmError('Bit', obj.name, 'is not declared as a bit in the gate.')

    def verify_bit_list(self, obj):
        if False:
            i = 10
            return i + 15
        'Verify each qubit in a list of ids.'
        for children in obj.children:
            self.verify_declared_bit(children)

    def verify_exp_list(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'Verify each expression in a list.'
        if obj.children is not None:
            for children in obj.children:
                if isinstance(children, node.Id):
                    if children.name in self.external_functions:
                        continue
                    if children.name not in self.current_symtab:
                        raise QasmError("Argument '" + children.name + "' in expression cannot be " + 'found, line', str(children.line), 'file', children.file)
                elif hasattr(children, 'children'):
                    self.verify_exp_list(children)

    def verify_as_gate(self, obj, bitlist, arglist=None):
        if False:
            for i in range(10):
                print('nop')
        'Verify a user defined gate call.'
        if obj.name not in self.global_symtab:
            raise QasmError("Cannot find gate definition for '" + obj.name + "', line", str(obj.line), 'file', obj.file)
        g_sym = self.global_symtab[obj.name]
        if g_sym.type not in ('gate', 'opaque'):
            raise QasmError("'" + obj.name + "' is used as a gate " + 'or opaque call but the symbol is neither;' + " it is a '" + g_sym.type + "' line", str(obj.line), 'file', obj.file)
        if g_sym.n_bits() != bitlist.size():
            raise QasmError("Gate or opaque call to '" + obj.name + "' uses", str(bitlist.size()), 'qubits but is declared for', str(g_sym.n_bits()), 'qubits', 'line', str(obj.line), 'file', obj.file)
        if arglist:
            if g_sym.n_args() != arglist.size():
                raise QasmError("Gate or opaque call to '" + obj.name + "' uses", str(arglist.size()), 'qubits but is declared for', str(g_sym.n_args()), 'qubits', 'line', str(obj.line), 'file', obj.file)
        elif g_sym.n_args() > 0:
            raise QasmError("Gate or opaque call to '" + obj.name + "' has no arguments but is declared for", str(g_sym.n_args()), 'qubits', 'line', str(obj.line), 'file', obj.file)

    def verify_reg(self, obj, object_type):
        if False:
            i = 10
            return i + 15
        'Verify a register.'
        if obj.name not in self.global_symtab:
            raise QasmError('Cannot find definition for', object_type, "'" + obj.name + "'", 'at line', str(obj.line), 'file', obj.file)
        g_sym = self.global_symtab[obj.name]
        if g_sym.type != object_type:
            raise QasmError("Type for '" + g_sym.name + "' should be '" + object_type + "' but was found to be '" + g_sym.type + "'", 'line', str(obj.line), 'file', obj.file)
        if obj.type == 'indexed_id':
            bound = g_sym.index
            ndx = obj.index
            if ndx < 0 or ndx >= bound:
                raise QasmError("Register index for '" + g_sym.name + "' out of bounds. Index is", str(ndx), 'bound is 0 <= index <', str(bound), 'at line', str(obj.line), 'file', obj.file)

    def verify_reg_list(self, obj, object_type):
        if False:
            for i in range(10):
                print('nop')
        'Verify a list of registers.'
        for children in obj.children:
            self.verify_reg(children, object_type)

    def id_tuple_list(self, id_node):
        if False:
            return 10
        'Return a list of (name, index) tuples for this id node.'
        if id_node.type != 'id':
            raise QasmError('internal error, id_tuple_list')
        bit_list = []
        try:
            g_sym = self.current_symtab[id_node.name]
        except KeyError:
            g_sym = self.global_symtab[id_node.name]
        if g_sym.type in ('qreg', 'creg'):
            for idx in range(g_sym.index):
                bit_list.append((id_node.name, idx))
        else:
            bit_list.append((id_node.name, -1))
        return bit_list

    def verify_distinct(self, list_of_nodes):
        if False:
            i = 10
            return i + 15
        "Check that objects in list_of_nodes represent distinct (qu)bits.\n\n        list_of_nodes is a list containing nodes of type id, indexed_id,\n        primary_list, or id_list. We assume these are all the same type\n        'qreg' or 'creg'.\n        This method raises an exception if list_of_nodes refers to the\n        same object more than once.\n        "
        bit_list = []
        line_number = -1
        filename = ''
        for node_ in list_of_nodes:
            if node_.type == 'id':
                bit_list.extend(self.id_tuple_list(node_))
                line_number = node_.line
                filename = node_.file
            elif node_.type == 'indexed_id':
                bit_list.append((node_.name, node_.index))
                line_number = node_.line
                filename = node_.file
            elif node_.type == 'primary_list':
                for child in node_.children:
                    if child.type == 'id':
                        bit_list.extend(self.id_tuple_list(child))
                    else:
                        bit_list.append((child.name, child.index))
                    line_number = child.line
                    filename = child.file
            elif node_.type == 'id_list':
                for child in node_.children:
                    bit_list.extend(self.id_tuple_list(child))
                    line_number = child.line
                    filename = child.file
            else:
                raise QasmError('internal error, verify_distinct')
        if len(bit_list) != len(set(bit_list)):
            raise QasmError('duplicate identifiers at line %d file %s' % (line_number, filename))

    def pop_scope(self):
        if False:
            return 10
        'Return to the previous scope.'
        self.current_symtab = self.symbols.pop()

    def push_scope(self):
        if False:
            print('Hello World!')
        'Enter a new scope.'
        self.symbols.append(self.current_symtab)
        self.current_symtab = {}
    start = 'main'

    def p_main(self, program):
        if False:
            while True:
                i = 10
        '\n        main : program\n        '
        self.qasm = program[1]

    def p_program_0(self, program):
        if False:
            while True:
                i = 10
        '\n        program : statement\n        '
        program[0] = node.Program([program[1]])

    def p_program_1(self, program):
        if False:
            i = 10
            return i + 15
        '\n        program : program statement\n        '
        program[0] = program[1]
        program[0].add_child(program[2])

    def p_statement(self, program):
        if False:
            return 10
        "\n        statement : decl\n                  | quantum_op ';'\n                  | format ';'\n                  | ignore\n                  | quantum_op error\n                  | format error\n        "
        if len(program) > 2:
            if program[2] != ';':
                raise QasmError("Missing ';' at end of statement; " + 'received', str(program[2].value))
        program[0] = program[1]

    def p_format(self, program):
        if False:
            for i in range(10):
                print('nop')
        '\n        format : FORMAT\n        '
        version = node.Format(program[1])
        if version.majorversion != '2' or version.minorversion != '0':
            provided_version = f'{version.majorversion}.{version.minorversion}'
            raise QasmError(f"Invalid version: '{provided_version}'. This module supports OpenQASM 2.0 only.")
        program[0] = version

    def p_id(self, program):
        if False:
            for i in range(10):
                print('nop')
        '\n        id : ID\n        '
        program[0] = program[1]

    def p_id_e(self, program):
        if False:
            print('Hello World!')
        '\n        id : error\n        '
        raise QasmError("Expected an ID, received '" + str(program[1].value) + "'")

    def p_indexed_id(self, program):
        if False:
            return 10
        "\n        indexed_id : id '[' NNINTEGER ']'\n                   | id '[' NNINTEGER error\n                   | id '[' error\n        "
        if len(program) == 4:
            raise QasmError('Expecting an integer index; received', str(program[3].value))
        if program[4] != ']':
            raise QasmError("Missing ']' in indexed ID; received", str(program[4].value))
        program[0] = node.IndexedId([program[1], node.Int(program[3])])

    def p_primary(self, program):
        if False:
            return 10
        '\n        primary : id\n                | indexed_id\n        '
        program[0] = program[1]

    def p_id_list_0(self, program):
        if False:
            print('Hello World!')
        '\n        id_list : id\n        '
        program[0] = node.IdList([program[1]])

    def p_id_list_1(self, program):
        if False:
            while True:
                i = 10
        "\n        id_list : id_list ',' id\n        "
        program[0] = program[1]
        program[0].add_child(program[3])

    def p_gate_id_list_0(self, program):
        if False:
            while True:
                i = 10
        '\n        gate_id_list : id\n        '
        program[0] = node.IdList([program[1]])
        self.update_symtab(program[1])

    def p_gate_id_list_1(self, program):
        if False:
            while True:
                i = 10
        "\n        gate_id_list : gate_id_list ',' id\n        "
        program[0] = program[1]
        program[0].add_child(program[3])
        self.update_symtab(program[3])

    def p_bit_list_0(self, program):
        if False:
            print('Hello World!')
        '\n        bit_list : id\n        '
        program[0] = node.IdList([program[1]])
        program[1].is_bit = True
        self.update_symtab(program[1])

    def p_bit_list_1(self, program):
        if False:
            i = 10
            return i + 15
        "\n        bit_list : bit_list ',' id\n        "
        program[0] = program[1]
        program[0].add_child(program[3])
        program[3].is_bit = True
        self.update_symtab(program[3])

    def p_primary_list_0(self, program):
        if False:
            print('Hello World!')
        '\n        primary_list : primary\n        '
        program[0] = node.PrimaryList([program[1]])

    def p_primary_list_1(self, program):
        if False:
            return 10
        "\n        primary_list : primary_list ',' primary\n        "
        program[0] = program[1]
        program[1].add_child(program[3])

    def p_decl(self, program):
        if False:
            print('Hello World!')
        "\n        decl : qreg_decl ';'\n             | creg_decl ';'\n             | qreg_decl error\n             | creg_decl error\n             | gate_decl\n        "
        if len(program) > 2:
            if program[2] != ';':
                raise QasmError("Missing ';' in qreg or creg declaration. Instead received '" + program[2].value + "'")
        program[0] = program[1]

    def p_qreg_decl(self, program):
        if False:
            print('Hello World!')
        '\n        qreg_decl : QREG indexed_id\n        '
        program[0] = node.Qreg([program[2]])
        if program[2].name in self.external_functions:
            raise QasmError('QREG names cannot be reserved words. ' + "Received '" + program[2].name + "'")
        if program[2].index == 0:
            raise QasmError('QREG size must be positive')
        self.update_symtab(program[0])

    def p_qreg_decl_e(self, program):
        if False:
            while True:
                i = 10
        '\n        qreg_decl : QREG error\n        '
        raise QasmError('Expecting indexed id (ID[int]) in QREG' + ' declaration; received', program[2].value)

    def p_creg_decl(self, program):
        if False:
            return 10
        '\n        creg_decl : CREG indexed_id\n        '
        program[0] = node.Creg([program[2]])
        if program[2].name in self.external_functions:
            raise QasmError('CREG names cannot be reserved words. ' + "Received '" + program[2].name + "'")
        if program[2].index == 0:
            raise QasmError('CREG size must be positive')
        self.update_symtab(program[0])

    def p_creg_decl_e(self, program):
        if False:
            while True:
                i = 10
        '\n        creg_decl : CREG error\n        '
        raise QasmError('Expecting indexed id (ID[int]) in CREG' + ' declaration; received', program[2].value)

    def p_gate_decl_0(self, program):
        if False:
            while True:
                i = 10
        '\n        gate_decl : GATE id gate_scope bit_list gate_body\n        '
        program[0] = node.Gate([program[2], program[4], program[5]])
        if program[2].name in self.external_functions:
            raise QasmError('GATE names cannot be reserved words. ' + "Received '" + program[2].name + "'")
        self.pop_scope()
        self.update_symtab(program[0])

    def p_gate_decl_1(self, program):
        if False:
            while True:
                i = 10
        "\n        gate_decl : GATE id gate_scope '(' ')' bit_list gate_body\n        "
        program[0] = node.Gate([program[2], program[6], program[7]])
        if program[2].name in self.external_functions:
            raise QasmError('GATE names cannot be reserved words. ' + "Received '" + program[2].name + "'")
        self.pop_scope()
        self.update_symtab(program[0])

    def p_gate_decl_2(self, program):
        if False:
            return 10
        "\n        gate_decl : GATE id gate_scope '(' gate_id_list ')' bit_list gate_body\n        "
        program[0] = node.Gate([program[2], program[5], program[7], program[8]])
        if program[2].name in self.external_functions:
            raise QasmError('GATE names cannot be reserved words. ' + "Received '" + program[2].name + "'")
        self.pop_scope()
        self.update_symtab(program[0])

    def p_gate_scope(self, _):
        if False:
            for i in range(10):
                print('nop')
        '\n        gate_scope :\n        '
        self.push_scope()

    def p_gate_body_0(self, program):
        if False:
            return 10
        "\n        gate_body : '{' '}'\n        "
        if program[2] != '}':
            raise QasmError("Missing '}' in gate definition; received'" + str(program[2].value) + "'")
        program[0] = node.GateBody(None)

    def p_gate_body_1(self, program):
        if False:
            return 10
        "\n        gate_body : '{' gate_op_list '}'\n        "
        program[0] = node.GateBody(program[2])

    def p_gate_op_list_0(self, program):
        if False:
            return 10
        '\n        gate_op_list : gate_op\n        '
        program[0] = [program[1]]

    def p_gate_op_list_1(self, program):
        if False:
            return 10
        '\n        gate_op_list : gate_op_list gate_op\n        '
        program[0] = program[1]
        program[0].append(program[2])

    def p_unitary_op_0(self, program):
        if False:
            while True:
                i = 10
        "\n        unitary_op : U '(' exp_list ')' primary\n        "
        program[0] = node.UniversalUnitary([program[3], program[5]])
        self.verify_reg(program[5], 'qreg')
        self.verify_exp_list(program[3])

    def p_unitary_op_1(self, program):
        if False:
            while True:
                i = 10
        "\n        unitary_op : CX primary ',' primary\n        "
        program[0] = node.Cnot([program[2], program[4]])
        self.verify_reg(program[2], 'qreg')
        self.verify_reg(program[4], 'qreg')
        self.verify_distinct([program[2], program[4]])

    def p_unitary_op_2(self, program):
        if False:
            while True:
                i = 10
        '\n        unitary_op : id primary_list\n        '
        program[0] = node.CustomUnitary([program[1], program[2]])
        self.verify_as_gate(program[1], program[2])
        self.verify_reg_list(program[2], 'qreg')
        self.verify_distinct([program[2]])

    def p_unitary_op_3(self, program):
        if False:
            for i in range(10):
                print('nop')
        "\n        unitary_op : id '(' ')' primary_list\n        "
        program[0] = node.CustomUnitary([program[1], program[4]])
        self.verify_as_gate(program[1], program[4])
        self.verify_reg_list(program[4], 'qreg')
        self.verify_distinct([program[4]])

    def p_unitary_op_4(self, program):
        if False:
            return 10
        "\n        unitary_op : id '(' exp_list ')' primary_list\n        "
        program[0] = node.CustomUnitary([program[1], program[3], program[5]])
        self.verify_as_gate(program[1], program[5], arglist=program[3])
        self.verify_reg_list(program[5], 'qreg')
        self.verify_exp_list(program[3])
        self.verify_distinct([program[5]])

    def p_gate_op_0(self, program):
        if False:
            return 10
        "\n        gate_op : U '(' exp_list ')' id ';'\n        "
        program[0] = node.UniversalUnitary([program[3], program[5]])
        self.verify_declared_bit(program[5])
        self.verify_exp_list(program[3])

    def p_gate_op_0e1(self, p):
        if False:
            while True:
                i = 10
        "\n        gate_op : U '(' exp_list ')' error\n        "
        raise QasmError('Invalid U inside gate definition. ' + "Missing bit id or ';'")

    def p_gate_op_0e2(self, _):
        if False:
            print('Hello World!')
        "\n        gate_op : U '(' exp_list error\n        "
        raise QasmError("Missing ')' in U invocation in gate definition.")

    def p_gate_op_1(self, program):
        if False:
            return 10
        "\n        gate_op : CX id ',' id ';'\n        "
        program[0] = node.Cnot([program[2], program[4]])
        self.verify_declared_bit(program[2])
        self.verify_declared_bit(program[4])
        self.verify_distinct([program[2], program[4]])

    def p_gate_op_1e1(self, program):
        if False:
            print('Hello World!')
        '\n        gate_op : CX error\n        '
        raise QasmError('Invalid CX inside gate definition. ' + "Expected an ID or ',', received '" + str(program[2].value) + "'")

    def p_gate_op_1e2(self, program):
        if False:
            return 10
        "\n        gate_op : CX id ',' error\n        "
        raise QasmError('Invalid CX inside gate definition. ' + "Expected an ID or ';', received '" + str(program[4].value) + "'")

    def p_gate_op_2(self, program):
        if False:
            while True:
                i = 10
        "\n        gate_op : id id_list ';'\n        "
        program[0] = node.CustomUnitary([program[1], program[2]])
        self.verify_as_gate(program[1], program[2])
        self.verify_bit_list(program[2])
        self.verify_distinct([program[2]])

    def p_gate_op_2e(self, _):
        if False:
            for i in range(10):
                print('nop')
        '\n        gate_op : id  id_list error\n        '
        raise QasmError('Invalid gate invocation inside gate definition.')

    def p_gate_op_3(self, program):
        if False:
            while True:
                i = 10
        "\n        gate_op : id '(' ')' id_list ';'\n        "
        program[0] = node.CustomUnitary([program[1], program[4]])
        self.verify_as_gate(program[1], program[4])
        self.verify_bit_list(program[4])
        self.verify_distinct([program[4]])

    def p_gate_op_4(self, program):
        if False:
            for i in range(10):
                print('nop')
        "\n        gate_op : id '(' exp_list ')' id_list ';'\n        "
        program[0] = node.CustomUnitary([program[1], program[3], program[5]])
        self.verify_as_gate(program[1], program[5], arglist=program[3])
        self.verify_bit_list(program[5])
        self.verify_exp_list(program[3])
        self.verify_distinct([program[5]])

    def p_gate_op_4e0(self, _):
        if False:
            for i in range(10):
                print('nop')
        "\n        gate_op : id '(' ')'  error\n        "
        raise QasmError('Invalid bit list inside gate definition or' + " missing ';'")

    def p_gate_op_4e1(self, _):
        if False:
            while True:
                i = 10
        "\n        gate_op : id '('   error\n        "
        raise QasmError('Unmatched () for gate invocation inside gate' + ' invocation.')

    def p_gate_op_5(self, program):
        if False:
            print('Hello World!')
        "\n        gate_op : BARRIER id_list ';'\n        "
        program[0] = node.Barrier([program[2]])
        self.verify_bit_list(program[2])
        self.verify_distinct([program[2]])

    def p_gate_op_5e(self, _):
        if False:
            return 10
        '\n        gate_op : BARRIER error\n        '
        raise QasmError('Invalid barrier inside gate definition.')

    def p_opaque_0(self, program):
        if False:
            i = 10
            return i + 15
        '\n        opaque : OPAQUE id gate_scope bit_list\n        '
        program[0] = node.Opaque([program[2], program[4]])
        if program[2].name in self.external_functions:
            raise QasmError('OPAQUE names cannot be reserved words. ' + "Received '" + program[2].name + "'")
        self.pop_scope()
        self.update_symtab(program[0])

    def p_opaque_1(self, program):
        if False:
            return 10
        "\n        opaque : OPAQUE id gate_scope '(' ')' bit_list\n        "
        program[0] = node.Opaque([program[2], program[6]])
        self.pop_scope()
        self.update_symtab(program[0])

    def p_opaque_2(self, program):
        if False:
            i = 10
            return i + 15
        "\n        opaque : OPAQUE id gate_scope '(' gate_id_list ')' bit_list\n        "
        program[0] = node.Opaque([program[2], program[5], program[7]])
        if program[2].name in self.external_functions:
            raise QasmError('OPAQUE names cannot be reserved words. ' + "Received '" + program[2].name + "'")
        self.pop_scope()
        self.update_symtab(program[0])

    def p_opaque_1e(self, _):
        if False:
            while True:
                i = 10
        "\n        opaque : OPAQUE id gate_scope '(' error\n        "
        raise QasmError('Poorly formed OPAQUE statement.')

    def p_measure(self, program):
        if False:
            print('Hello World!')
        '\n        measure : MEASURE primary ASSIGN primary\n        '
        program[0] = node.Measure([program[2], program[4]])
        self.verify_reg(program[2], 'qreg')
        self.verify_reg(program[4], 'creg')

    def p_measure_e(self, program):
        if False:
            i = 10
            return i + 15
        '\n        measure : MEASURE primary error\n        '
        raise QasmError('Illegal measure statement.' + str(program[3].value))

    def p_barrier(self, program):
        if False:
            for i in range(10):
                print('nop')
        '\n        barrier : BARRIER primary_list\n        '
        program[0] = node.Barrier([program[2]])
        self.verify_reg_list(program[2], 'qreg')
        self.verify_distinct([program[2]])

    def p_reset(self, program):
        if False:
            while True:
                i = 10
        '\n        reset : RESET primary\n        '
        program[0] = node.Reset([program[2]])
        self.verify_reg(program[2], 'qreg')

    def p_if(self, program):
        if False:
            i = 10
            return i + 15
        "\n        if : IF '(' id MATCHES NNINTEGER ')' quantum_op\n        if : IF '(' id error\n        if : IF '(' id MATCHES error\n        if : IF '(' id MATCHES NNINTEGER error\n        if : IF error\n        "
        if len(program) == 3:
            raise QasmError('Ill-formed IF statement. Perhaps a' + " missing '('?")
        if len(program) == 5:
            raise QasmError("Ill-formed IF statement.  Expected '==', " + "received '" + str(program[4].value))
        if len(program) == 6:
            raise QasmError('Ill-formed IF statement.  Expected a number, ' + "received '" + str(program[5].value))
        if len(program) == 7:
            raise QasmError("Ill-formed IF statement, unmatched '('")
        if program[7].type == 'if':
            raise QasmError('Nested IF statements not allowed')
        if program[7].type == 'barrier':
            raise QasmError('barrier not permitted in IF statement')
        program[0] = node.If([program[3], node.Int(program[5]), program[7]])

    def p_quantum_op(self, program):
        if False:
            while True:
                i = 10
        '\n        quantum_op : unitary_op\n                   | opaque\n                   | measure\n                   | barrier\n                   | reset\n                   | if\n        '
        program[0] = program[1]

    def p_unary_0(self, program):
        if False:
            print('Hello World!')
        '\n        unary : NNINTEGER\n        '
        program[0] = node.Int(program[1])

    def p_unary_1(self, program):
        if False:
            return 10
        '\n        unary : REAL\n        '
        program[0] = node.Real(program[1])

    def p_unary_2(self, program):
        if False:
            while True:
                i = 10
        '\n        unary : PI\n        '
        program[0] = node.Real(np.pi)

    def p_unary_3(self, program):
        if False:
            return 10
        '\n        unary : id\n        '
        program[0] = program[1]

    def p_unary_4(self, program):
        if False:
            i = 10
            return i + 15
        "\n        unary : '(' expression ')'\n        "
        program[0] = program[2]

    def p_unary_6(self, program):
        if False:
            print('Hello World!')
        "\n        unary : id '(' expression ')'\n        "
        if program[1].name not in self.external_functions:
            raise QasmError('Illegal external function call: ', str(program[1].name))
        program[0] = node.External([program[1], program[3]])

    def p_expression_1(self, program):
        if False:
            return 10
        "\n        expression : '-' expression %prec negative\n                    | '+' expression %prec positive\n        "
        program[0] = node.Prefix([node.UnaryOperator(program[1]), program[2]])

    def p_expression_0(self, program):
        if False:
            return 10
        "\n        expression : expression '*' expression\n                    | expression '/' expression\n                    | expression '+' expression\n                    | expression '-' expression\n                    | expression '^' expression\n        "
        program[0] = node.BinaryOp([node.BinaryOperator(program[2]), program[1], program[3]])

    def p_expression_2(self, program):
        if False:
            return 10
        '\n        expression : unary\n        '
        program[0] = program[1]

    def p_exp_list_0(self, program):
        if False:
            i = 10
            return i + 15
        '\n        exp_list : expression\n        '
        program[0] = node.ExpressionList([program[1]])

    def p_exp_list_1(self, program):
        if False:
            for i in range(10):
                print('nop')
        "\n        exp_list : exp_list ',' expression\n        "
        program[0] = program[1]
        program[0].add_child(program[3])

    def p_ignore(self, _):
        if False:
            return 10
        '\n        ignore : STRING\n        '
        pass

    def p_error(self, program):
        if False:
            return 10
        if not program:
            raise QasmError('Error at end of file. ' + "Perhaps there is a missing ';'")
        col = self.find_column(self.lexer.data, program)
        print('Error near line', str(self.lexer.lineno), 'Column', col)

    def find_column(self, input_, token):
        if False:
            print('Hello World!')
        'Compute the column.\n\n        Input is the input text string.\n        token is a token instance.\n        '
        if token is None:
            return 0
        last_cr = input_.rfind('\n', 0, token.lexpos)
        last_cr = max(last_cr, 0)
        column = token.lexpos - last_cr + 1
        return column

    def read_tokens(self):
        if False:
            while True:
                i = 10
        'finds and reads the tokens.'
        try:
            while True:
                token = self.lexer.token()
                if not token:
                    break
                yield token
        except QasmError as e:
            print('Exception tokenizing qasm file:', e.msg)

    def parse_debug(self, val):
        if False:
            i = 10
            return i + 15
        'Set the parse_deb field.'
        if val is True:
            self.parse_deb = True
        elif val is False:
            self.parse_deb = False
        else:
            raise QasmError("Illegal debug value '" + str(val) + "' must be True or False.")

    def parse(self, data):
        if False:
            while True:
                i = 10
        'Parse some data.'
        self.parser.parse(data, lexer=self.lexer, debug=self.parse_deb)
        if self.qasm is None:
            raise QasmError('Uncaught exception in parser; ' + 'see previous messages for details.')
        return self.qasm

    def print_tree(self):
        if False:
            return 10
        'Print parsed OPENQASM.'
        if self.qasm is not None:
            self.qasm.to_string(0)
        else:
            print('No parsed qasm to print')

    def run(self, data):
        if False:
            print('Hello World!')
        'Parser runner.\n\n        To use this module stand-alone.\n        '
        ast = self.parser.parse(data, debug=True)
        self.parser.parse(data, debug=True)
        ast.to_string(0)