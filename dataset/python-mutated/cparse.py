import sys
import clex
import ply.yacc as yacc
tokens = clex.tokens

def p_translation_unit_1(t):
    if False:
        return 10
    'translation_unit : external_declaration'
    pass

def p_translation_unit_2(t):
    if False:
        for i in range(10):
            print('nop')
    'translation_unit : translation_unit external_declaration'
    pass

def p_external_declaration_1(t):
    if False:
        i = 10
        return i + 15
    'external_declaration : function_definition'
    pass

def p_external_declaration_2(t):
    if False:
        for i in range(10):
            print('nop')
    'external_declaration : declaration'
    pass

def p_function_definition_1(t):
    if False:
        for i in range(10):
            print('nop')
    'function_definition : declaration_specifiers declarator declaration_list compound_statement'
    pass

def p_function_definition_2(t):
    if False:
        i = 10
        return i + 15
    'function_definition : declarator declaration_list compound_statement'
    pass

def p_function_definition_3(t):
    if False:
        print('Hello World!')
    'function_definition : declarator compound_statement'
    pass

def p_function_definition_4(t):
    if False:
        return 10
    'function_definition : declaration_specifiers declarator compound_statement'
    pass

def p_declaration_1(t):
    if False:
        print('Hello World!')
    'declaration : declaration_specifiers init_declarator_list SEMI'
    pass

def p_declaration_2(t):
    if False:
        for i in range(10):
            print('nop')
    'declaration : declaration_specifiers SEMI'
    pass

def p_declaration_list_1(t):
    if False:
        return 10
    'declaration_list : declaration'
    pass

def p_declaration_list_2(t):
    if False:
        return 10
    'declaration_list : declaration_list declaration '
    pass

def p_declaration_specifiers_1(t):
    if False:
        print('Hello World!')
    'declaration_specifiers : storage_class_specifier declaration_specifiers'
    pass

def p_declaration_specifiers_2(t):
    if False:
        return 10
    'declaration_specifiers : type_specifier declaration_specifiers'
    pass

def p_declaration_specifiers_3(t):
    if False:
        while True:
            i = 10
    'declaration_specifiers : type_qualifier declaration_specifiers'
    pass

def p_declaration_specifiers_4(t):
    if False:
        for i in range(10):
            print('nop')
    'declaration_specifiers : storage_class_specifier'
    pass

def p_declaration_specifiers_5(t):
    if False:
        i = 10
        return i + 15
    'declaration_specifiers : type_specifier'
    pass

def p_declaration_specifiers_6(t):
    if False:
        print('Hello World!')
    'declaration_specifiers : type_qualifier'
    pass

def p_storage_class_specifier(t):
    if False:
        return 10
    'storage_class_specifier : AUTO\n                               | REGISTER\n                               | STATIC\n                               | EXTERN\n                               | TYPEDEF\n                               '
    pass

def p_type_specifier(t):
    if False:
        return 10
    'type_specifier : VOID\n                      | CHAR\n                      | SHORT\n                      | INT\n                      | LONG\n                      | FLOAT\n                      | DOUBLE\n                      | SIGNED\n                      | UNSIGNED\n                      | struct_or_union_specifier\n                      | enum_specifier\n                      | TYPEID\n                      '
    pass

def p_type_qualifier(t):
    if False:
        for i in range(10):
            print('nop')
    'type_qualifier : CONST\n                      | VOLATILE'
    pass

def p_struct_or_union_specifier_1(t):
    if False:
        while True:
            i = 10
    'struct_or_union_specifier : struct_or_union ID LBRACE struct_declaration_list RBRACE'
    pass

def p_struct_or_union_specifier_2(t):
    if False:
        return 10
    'struct_or_union_specifier : struct_or_union LBRACE struct_declaration_list RBRACE'
    pass

def p_struct_or_union_specifier_3(t):
    if False:
        return 10
    'struct_or_union_specifier : struct_or_union ID'
    pass

def p_struct_or_union(t):
    if False:
        for i in range(10):
            print('nop')
    'struct_or_union : STRUCT\n                       | UNION\n                       '
    pass

def p_struct_declaration_list_1(t):
    if False:
        print('Hello World!')
    'struct_declaration_list : struct_declaration'
    pass

def p_struct_declaration_list_2(t):
    if False:
        while True:
            i = 10
    'struct_declaration_list : struct_declaration_list struct_declaration'
    pass

def p_init_declarator_list_1(t):
    if False:
        i = 10
        return i + 15
    'init_declarator_list : init_declarator'
    pass

def p_init_declarator_list_2(t):
    if False:
        i = 10
        return i + 15
    'init_declarator_list : init_declarator_list COMMA init_declarator'
    pass

def p_init_declarator_1(t):
    if False:
        for i in range(10):
            print('nop')
    'init_declarator : declarator'
    pass

def p_init_declarator_2(t):
    if False:
        print('Hello World!')
    'init_declarator : declarator EQUALS initializer'
    pass

def p_struct_declaration(t):
    if False:
        i = 10
        return i + 15
    'struct_declaration : specifier_qualifier_list struct_declarator_list SEMI'
    pass

def p_specifier_qualifier_list_1(t):
    if False:
        i = 10
        return i + 15
    'specifier_qualifier_list : type_specifier specifier_qualifier_list'
    pass

def p_specifier_qualifier_list_2(t):
    if False:
        return 10
    'specifier_qualifier_list : type_specifier'
    pass

def p_specifier_qualifier_list_3(t):
    if False:
        for i in range(10):
            print('nop')
    'specifier_qualifier_list : type_qualifier specifier_qualifier_list'
    pass

def p_specifier_qualifier_list_4(t):
    if False:
        for i in range(10):
            print('nop')
    'specifier_qualifier_list : type_qualifier'
    pass

def p_struct_declarator_list_1(t):
    if False:
        while True:
            i = 10
    'struct_declarator_list : struct_declarator'
    pass

def p_struct_declarator_list_2(t):
    if False:
        i = 10
        return i + 15
    'struct_declarator_list : struct_declarator_list COMMA struct_declarator'
    pass

def p_struct_declarator_1(t):
    if False:
        i = 10
        return i + 15
    'struct_declarator : declarator'
    pass

def p_struct_declarator_2(t):
    if False:
        i = 10
        return i + 15
    'struct_declarator : declarator COLON constant_expression'
    pass

def p_struct_declarator_3(t):
    if False:
        return 10
    'struct_declarator : COLON constant_expression'
    pass

def p_enum_specifier_1(t):
    if False:
        return 10
    'enum_specifier : ENUM ID LBRACE enumerator_list RBRACE'
    pass

def p_enum_specifier_2(t):
    if False:
        while True:
            i = 10
    'enum_specifier : ENUM LBRACE enumerator_list RBRACE'
    pass

def p_enum_specifier_3(t):
    if False:
        for i in range(10):
            print('nop')
    'enum_specifier : ENUM ID'
    pass

def p_enumerator_list_1(t):
    if False:
        print('Hello World!')
    'enumerator_list : enumerator'
    pass

def p_enumerator_list_2(t):
    if False:
        for i in range(10):
            print('nop')
    'enumerator_list : enumerator_list COMMA enumerator'
    pass

def p_enumerator_1(t):
    if False:
        return 10
    'enumerator : ID'
    pass

def p_enumerator_2(t):
    if False:
        while True:
            i = 10
    'enumerator : ID EQUALS constant_expression'
    pass

def p_declarator_1(t):
    if False:
        while True:
            i = 10
    'declarator : pointer direct_declarator'
    pass

def p_declarator_2(t):
    if False:
        print('Hello World!')
    'declarator : direct_declarator'
    pass

def p_direct_declarator_1(t):
    if False:
        return 10
    'direct_declarator : ID'
    pass

def p_direct_declarator_2(t):
    if False:
        while True:
            i = 10
    'direct_declarator : LPAREN declarator RPAREN'
    pass

def p_direct_declarator_3(t):
    if False:
        i = 10
        return i + 15
    'direct_declarator : direct_declarator LBRACKET constant_expression_opt RBRACKET'
    pass

def p_direct_declarator_4(t):
    if False:
        while True:
            i = 10
    'direct_declarator : direct_declarator LPAREN parameter_type_list RPAREN '
    pass

def p_direct_declarator_5(t):
    if False:
        return 10
    'direct_declarator : direct_declarator LPAREN identifier_list RPAREN '
    pass

def p_direct_declarator_6(t):
    if False:
        for i in range(10):
            print('nop')
    'direct_declarator : direct_declarator LPAREN RPAREN '
    pass

def p_pointer_1(t):
    if False:
        while True:
            i = 10
    'pointer : TIMES type_qualifier_list'
    pass

def p_pointer_2(t):
    if False:
        i = 10
        return i + 15
    'pointer : TIMES'
    pass

def p_pointer_3(t):
    if False:
        while True:
            i = 10
    'pointer : TIMES type_qualifier_list pointer'
    pass

def p_pointer_4(t):
    if False:
        return 10
    'pointer : TIMES pointer'
    pass

def p_type_qualifier_list_1(t):
    if False:
        i = 10
        return i + 15
    'type_qualifier_list : type_qualifier'
    pass

def p_type_qualifier_list_2(t):
    if False:
        return 10
    'type_qualifier_list : type_qualifier_list type_qualifier'
    pass

def p_parameter_type_list_1(t):
    if False:
        while True:
            i = 10
    'parameter_type_list : parameter_list'
    pass

def p_parameter_type_list_2(t):
    if False:
        print('Hello World!')
    'parameter_type_list : parameter_list COMMA ELLIPSIS'
    pass

def p_parameter_list_1(t):
    if False:
        for i in range(10):
            print('nop')
    'parameter_list : parameter_declaration'
    pass

def p_parameter_list_2(t):
    if False:
        i = 10
        return i + 15
    'parameter_list : parameter_list COMMA parameter_declaration'
    pass

def p_parameter_declaration_1(t):
    if False:
        for i in range(10):
            print('nop')
    'parameter_declaration : declaration_specifiers declarator'
    pass

def p_parameter_declaration_2(t):
    if False:
        while True:
            i = 10
    'parameter_declaration : declaration_specifiers abstract_declarator_opt'
    pass

def p_identifier_list_1(t):
    if False:
        for i in range(10):
            print('nop')
    'identifier_list : ID'
    pass

def p_identifier_list_2(t):
    if False:
        return 10
    'identifier_list : identifier_list COMMA ID'
    pass

def p_initializer_1(t):
    if False:
        while True:
            i = 10
    'initializer : assignment_expression'
    pass

def p_initializer_2(t):
    if False:
        for i in range(10):
            print('nop')
    'initializer : LBRACE initializer_list RBRACE\n                   | LBRACE initializer_list COMMA RBRACE'
    pass

def p_initializer_list_1(t):
    if False:
        i = 10
        return i + 15
    'initializer_list : initializer'
    pass

def p_initializer_list_2(t):
    if False:
        for i in range(10):
            print('nop')
    'initializer_list : initializer_list COMMA initializer'
    pass

def p_type_name(t):
    if False:
        while True:
            i = 10
    'type_name : specifier_qualifier_list abstract_declarator_opt'
    pass

def p_abstract_declarator_opt_1(t):
    if False:
        while True:
            i = 10
    'abstract_declarator_opt : empty'
    pass

def p_abstract_declarator_opt_2(t):
    if False:
        print('Hello World!')
    'abstract_declarator_opt : abstract_declarator'
    pass

def p_abstract_declarator_1(t):
    if False:
        while True:
            i = 10
    'abstract_declarator : pointer '
    pass

def p_abstract_declarator_2(t):
    if False:
        while True:
            i = 10
    'abstract_declarator : pointer direct_abstract_declarator'
    pass

def p_abstract_declarator_3(t):
    if False:
        for i in range(10):
            print('nop')
    'abstract_declarator : direct_abstract_declarator'
    pass

def p_direct_abstract_declarator_1(t):
    if False:
        for i in range(10):
            print('nop')
    'direct_abstract_declarator : LPAREN abstract_declarator RPAREN'
    pass

def p_direct_abstract_declarator_2(t):
    if False:
        print('Hello World!')
    'direct_abstract_declarator : direct_abstract_declarator LBRACKET constant_expression_opt RBRACKET'
    pass

def p_direct_abstract_declarator_3(t):
    if False:
        print('Hello World!')
    'direct_abstract_declarator : LBRACKET constant_expression_opt RBRACKET'
    pass

def p_direct_abstract_declarator_4(t):
    if False:
        for i in range(10):
            print('nop')
    'direct_abstract_declarator : direct_abstract_declarator LPAREN parameter_type_list_opt RPAREN'
    pass

def p_direct_abstract_declarator_5(t):
    if False:
        print('Hello World!')
    'direct_abstract_declarator : LPAREN parameter_type_list_opt RPAREN'
    pass

def p_constant_expression_opt_1(t):
    if False:
        i = 10
        return i + 15
    'constant_expression_opt : empty'
    pass

def p_constant_expression_opt_2(t):
    if False:
        i = 10
        return i + 15
    'constant_expression_opt : constant_expression'
    pass

def p_parameter_type_list_opt_1(t):
    if False:
        i = 10
        return i + 15
    'parameter_type_list_opt : empty'
    pass

def p_parameter_type_list_opt_2(t):
    if False:
        return 10
    'parameter_type_list_opt : parameter_type_list'
    pass

def p_statement(t):
    if False:
        while True:
            i = 10
    '\n    statement : labeled_statement\n              | expression_statement\n              | compound_statement\n              | selection_statement\n              | iteration_statement\n              | jump_statement\n              '
    pass

def p_labeled_statement_1(t):
    if False:
        i = 10
        return i + 15
    'labeled_statement : ID COLON statement'
    pass

def p_labeled_statement_2(t):
    if False:
        i = 10
        return i + 15
    'labeled_statement : CASE constant_expression COLON statement'
    pass

def p_labeled_statement_3(t):
    if False:
        while True:
            i = 10
    'labeled_statement : DEFAULT COLON statement'
    pass

def p_expression_statement(t):
    if False:
        for i in range(10):
            print('nop')
    'expression_statement : expression_opt SEMI'
    pass

def p_compound_statement_1(t):
    if False:
        return 10
    'compound_statement : LBRACE declaration_list statement_list RBRACE'
    pass

def p_compound_statement_2(t):
    if False:
        return 10
    'compound_statement : LBRACE statement_list RBRACE'
    pass

def p_compound_statement_3(t):
    if False:
        for i in range(10):
            print('nop')
    'compound_statement : LBRACE declaration_list RBRACE'
    pass

def p_compound_statement_4(t):
    if False:
        return 10
    'compound_statement : LBRACE RBRACE'
    pass

def p_statement_list_1(t):
    if False:
        for i in range(10):
            print('nop')
    'statement_list : statement'
    pass

def p_statement_list_2(t):
    if False:
        i = 10
        return i + 15
    'statement_list : statement_list statement'
    pass

def p_selection_statement_1(t):
    if False:
        i = 10
        return i + 15
    'selection_statement : IF LPAREN expression RPAREN statement'
    pass

def p_selection_statement_2(t):
    if False:
        while True:
            i = 10
    'selection_statement : IF LPAREN expression RPAREN statement ELSE statement '
    pass

def p_selection_statement_3(t):
    if False:
        return 10
    'selection_statement : SWITCH LPAREN expression RPAREN statement '
    pass

def p_iteration_statement_1(t):
    if False:
        while True:
            i = 10
    'iteration_statement : WHILE LPAREN expression RPAREN statement'
    pass

def p_iteration_statement_2(t):
    if False:
        i = 10
        return i + 15
    'iteration_statement : FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN statement '
    pass

def p_iteration_statement_3(t):
    if False:
        print('Hello World!')
    'iteration_statement : DO statement WHILE LPAREN expression RPAREN SEMI'
    pass

def p_jump_statement_1(t):
    if False:
        while True:
            i = 10
    'jump_statement : GOTO ID SEMI'
    pass

def p_jump_statement_2(t):
    if False:
        i = 10
        return i + 15
    'jump_statement : CONTINUE SEMI'
    pass

def p_jump_statement_3(t):
    if False:
        return 10
    'jump_statement : BREAK SEMI'
    pass

def p_jump_statement_4(t):
    if False:
        print('Hello World!')
    'jump_statement : RETURN expression_opt SEMI'
    pass

def p_expression_opt_1(t):
    if False:
        print('Hello World!')
    'expression_opt : empty'
    pass

def p_expression_opt_2(t):
    if False:
        for i in range(10):
            print('nop')
    'expression_opt : expression'
    pass

def p_expression_1(t):
    if False:
        for i in range(10):
            print('nop')
    'expression : assignment_expression'
    pass

def p_expression_2(t):
    if False:
        return 10
    'expression : expression COMMA assignment_expression'
    pass

def p_assignment_expression_1(t):
    if False:
        print('Hello World!')
    'assignment_expression : conditional_expression'
    pass

def p_assignment_expression_2(t):
    if False:
        print('Hello World!')
    'assignment_expression : unary_expression assignment_operator assignment_expression'
    pass

def p_assignment_operator(t):
    if False:
        while True:
            i = 10
    '\n    assignment_operator : EQUALS\n                        | TIMESEQUAL\n                        | DIVEQUAL\n                        | MODEQUAL\n                        | PLUSEQUAL\n                        | MINUSEQUAL\n                        | LSHIFTEQUAL\n                        | RSHIFTEQUAL\n                        | ANDEQUAL\n                        | OREQUAL\n                        | XOREQUAL\n                        '
    pass

def p_conditional_expression_1(t):
    if False:
        i = 10
        return i + 15
    'conditional_expression : logical_or_expression'
    pass

def p_conditional_expression_2(t):
    if False:
        i = 10
        return i + 15
    'conditional_expression : logical_or_expression CONDOP expression COLON conditional_expression '
    pass

def p_constant_expression(t):
    if False:
        for i in range(10):
            print('nop')
    'constant_expression : conditional_expression'
    pass

def p_logical_or_expression_1(t):
    if False:
        print('Hello World!')
    'logical_or_expression : logical_and_expression'
    pass

def p_logical_or_expression_2(t):
    if False:
        while True:
            i = 10
    'logical_or_expression : logical_or_expression LOR logical_and_expression'
    pass

def p_logical_and_expression_1(t):
    if False:
        for i in range(10):
            print('nop')
    'logical_and_expression : inclusive_or_expression'
    pass

def p_logical_and_expression_2(t):
    if False:
        print('Hello World!')
    'logical_and_expression : logical_and_expression LAND inclusive_or_expression'
    pass

def p_inclusive_or_expression_1(t):
    if False:
        return 10
    'inclusive_or_expression : exclusive_or_expression'
    pass

def p_inclusive_or_expression_2(t):
    if False:
        i = 10
        return i + 15
    'inclusive_or_expression : inclusive_or_expression OR exclusive_or_expression'
    pass

def p_exclusive_or_expression_1(t):
    if False:
        while True:
            i = 10
    'exclusive_or_expression :  and_expression'
    pass

def p_exclusive_or_expression_2(t):
    if False:
        return 10
    'exclusive_or_expression :  exclusive_or_expression XOR and_expression'
    pass

def p_and_expression_1(t):
    if False:
        for i in range(10):
            print('nop')
    'and_expression : equality_expression'
    pass

def p_and_expression_2(t):
    if False:
        for i in range(10):
            print('nop')
    'and_expression : and_expression AND equality_expression'
    pass

def p_equality_expression_1(t):
    if False:
        print('Hello World!')
    'equality_expression : relational_expression'
    pass

def p_equality_expression_2(t):
    if False:
        return 10
    'equality_expression : equality_expression EQ relational_expression'
    pass

def p_equality_expression_3(t):
    if False:
        while True:
            i = 10
    'equality_expression : equality_expression NE relational_expression'
    pass

def p_relational_expression_1(t):
    if False:
        for i in range(10):
            print('nop')
    'relational_expression : shift_expression'
    pass

def p_relational_expression_2(t):
    if False:
        while True:
            i = 10
    'relational_expression : relational_expression LT shift_expression'
    pass

def p_relational_expression_3(t):
    if False:
        print('Hello World!')
    'relational_expression : relational_expression GT shift_expression'
    pass

def p_relational_expression_4(t):
    if False:
        while True:
            i = 10
    'relational_expression : relational_expression LE shift_expression'
    pass

def p_relational_expression_5(t):
    if False:
        for i in range(10):
            print('nop')
    'relational_expression : relational_expression GE shift_expression'
    pass

def p_shift_expression_1(t):
    if False:
        while True:
            i = 10
    'shift_expression : additive_expression'
    pass

def p_shift_expression_2(t):
    if False:
        while True:
            i = 10
    'shift_expression : shift_expression LSHIFT additive_expression'
    pass

def p_shift_expression_3(t):
    if False:
        while True:
            i = 10
    'shift_expression : shift_expression RSHIFT additive_expression'
    pass

def p_additive_expression_1(t):
    if False:
        for i in range(10):
            print('nop')
    'additive_expression : multiplicative_expression'
    pass

def p_additive_expression_2(t):
    if False:
        for i in range(10):
            print('nop')
    'additive_expression : additive_expression PLUS multiplicative_expression'
    pass

def p_additive_expression_3(t):
    if False:
        return 10
    'additive_expression : additive_expression MINUS multiplicative_expression'
    pass

def p_multiplicative_expression_1(t):
    if False:
        for i in range(10):
            print('nop')
    'multiplicative_expression : cast_expression'
    pass

def p_multiplicative_expression_2(t):
    if False:
        print('Hello World!')
    'multiplicative_expression : multiplicative_expression TIMES cast_expression'
    pass

def p_multiplicative_expression_3(t):
    if False:
        while True:
            i = 10
    'multiplicative_expression : multiplicative_expression DIVIDE cast_expression'
    pass

def p_multiplicative_expression_4(t):
    if False:
        return 10
    'multiplicative_expression : multiplicative_expression MOD cast_expression'
    pass

def p_cast_expression_1(t):
    if False:
        return 10
    'cast_expression : unary_expression'
    pass

def p_cast_expression_2(t):
    if False:
        while True:
            i = 10
    'cast_expression : LPAREN type_name RPAREN cast_expression'
    pass

def p_unary_expression_1(t):
    if False:
        for i in range(10):
            print('nop')
    'unary_expression : postfix_expression'
    pass

def p_unary_expression_2(t):
    if False:
        return 10
    'unary_expression : PLUSPLUS unary_expression'
    pass

def p_unary_expression_3(t):
    if False:
        while True:
            i = 10
    'unary_expression : MINUSMINUS unary_expression'
    pass

def p_unary_expression_4(t):
    if False:
        return 10
    'unary_expression : unary_operator cast_expression'
    pass

def p_unary_expression_5(t):
    if False:
        for i in range(10):
            print('nop')
    'unary_expression : SIZEOF unary_expression'
    pass

def p_unary_expression_6(t):
    if False:
        print('Hello World!')
    'unary_expression : SIZEOF LPAREN type_name RPAREN'
    pass

def p_unary_operator(t):
    if False:
        i = 10
        return i + 15
    'unary_operator : AND\n                    | TIMES\n                    | PLUS \n                    | MINUS\n                    | NOT\n                    | LNOT '
    pass

def p_postfix_expression_1(t):
    if False:
        return 10
    'postfix_expression : primary_expression'
    pass

def p_postfix_expression_2(t):
    if False:
        print('Hello World!')
    'postfix_expression : postfix_expression LBRACKET expression RBRACKET'
    pass

def p_postfix_expression_3(t):
    if False:
        while True:
            i = 10
    'postfix_expression : postfix_expression LPAREN argument_expression_list RPAREN'
    pass

def p_postfix_expression_4(t):
    if False:
        print('Hello World!')
    'postfix_expression : postfix_expression LPAREN RPAREN'
    pass

def p_postfix_expression_5(t):
    if False:
        i = 10
        return i + 15
    'postfix_expression : postfix_expression PERIOD ID'
    pass

def p_postfix_expression_6(t):
    if False:
        return 10
    'postfix_expression : postfix_expression ARROW ID'
    pass

def p_postfix_expression_7(t):
    if False:
        while True:
            i = 10
    'postfix_expression : postfix_expression PLUSPLUS'
    pass

def p_postfix_expression_8(t):
    if False:
        for i in range(10):
            print('nop')
    'postfix_expression : postfix_expression MINUSMINUS'
    pass

def p_primary_expression(t):
    if False:
        print('Hello World!')
    'primary_expression :  ID\n                        |  constant\n                        |  SCONST\n                        |  LPAREN expression RPAREN'
    pass

def p_argument_expression_list(t):
    if False:
        i = 10
        return i + 15
    'argument_expression_list :  assignment_expression\n                              |  argument_expression_list COMMA assignment_expression'
    pass

def p_constant(t):
    if False:
        return 10
    'constant : ICONST\n               | FCONST\n               | CCONST'
    pass

def p_empty(t):
    if False:
        while True:
            i = 10
    'empty : '
    pass

def p_error(t):
    if False:
        return 10
    print("Whoa. We're hosed")
import profile
yacc.yacc()