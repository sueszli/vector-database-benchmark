from antlr4 import *
if __name__ is not None and '.' in __name__:
    from .CParser import CParser
else:
    from CParser import CParser
import Ecc.CodeFragment as CodeFragment
import Ecc.FileProfile as FileProfile

class CListener(ParseTreeListener):

    def enterTranslation_unit(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitTranslation_unit(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterExternal_declaration(self, ctx):
        if False:
            return 10
        pass

    def exitExternal_declaration(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterFunction_definition(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitFunction_definition(self, ctx):
        if False:
            return 10
        pass

    def enterDeclaration_specifiers(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitDeclaration_specifiers(self, ctx):
        if False:
            return 10
        pass

    def enterDeclaration(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitDeclaration(self, ctx):
        if False:
            return 10
        pass

    def enterInit_declarator_list(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitInit_declarator_list(self, ctx):
        if False:
            return 10
        pass

    def enterInit_declarator(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitInit_declarator(self, ctx):
        if False:
            return 10
        pass

    def enterStorage_class_specifier(self, ctx):
        if False:
            return 10
        pass

    def exitStorage_class_specifier(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterType_specifier(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitType_specifier(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterType_id(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitType_id(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterStruct_or_union_specifier(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitStruct_or_union_specifier(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterStruct_or_union(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitStruct_or_union(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterStruct_declaration_list(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitStruct_declaration_list(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterStruct_declaration(self, ctx):
        if False:
            return 10
        pass

    def exitStruct_declaration(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterSpecifier_qualifier_list(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitSpecifier_qualifier_list(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterStruct_declarator_list(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitStruct_declarator_list(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterStruct_declarator(self, ctx):
        if False:
            return 10
        pass

    def exitStruct_declarator(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterEnum_specifier(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitEnum_specifier(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterEnumerator_list(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitEnumerator_list(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterEnumerator(self, ctx):
        if False:
            return 10
        pass

    def exitEnumerator(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterType_qualifier(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitType_qualifier(self, ctx):
        if False:
            return 10
        pass

    def enterDeclarator(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitDeclarator(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterDirect_declarator(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitDirect_declarator(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterDeclarator_suffix(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitDeclarator_suffix(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterPointer(self, ctx):
        if False:
            return 10
        pass

    def exitPointer(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterParameter_type_list(self, ctx):
        if False:
            return 10
        pass

    def exitParameter_type_list(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterParameter_list(self, ctx):
        if False:
            return 10
        pass

    def exitParameter_list(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterParameter_declaration(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitParameter_declaration(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterIdentifier_list(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitIdentifier_list(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterType_name(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitType_name(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterAbstract_declarator(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitAbstract_declarator(self, ctx):
        if False:
            return 10
        pass

    def enterDirect_abstract_declarator(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitDirect_abstract_declarator(self, ctx):
        if False:
            return 10
        pass

    def enterAbstract_declarator_suffix(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitAbstract_declarator_suffix(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterInitializer(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitInitializer(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterInitializer_list(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitInitializer_list(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterArgument_expression_list(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitArgument_expression_list(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterAdditive_expression(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitAdditive_expression(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterMultiplicative_expression(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitMultiplicative_expression(self, ctx):
        if False:
            return 10
        pass

    def enterCast_expression(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitCast_expression(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterUnary_expression(self, ctx):
        if False:
            return 10
        pass

    def exitUnary_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterPostfix_expression(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitPostfix_expression(self, ctx):
        if False:
            return 10
        pass

    def enterMacro_parameter_list(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitMacro_parameter_list(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterUnary_operator(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitUnary_operator(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterPrimary_expression(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitPrimary_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterConstant(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitConstant(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterExpression(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitExpression(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterConstant_expression(self, ctx):
        if False:
            return 10
        pass

    def exitConstant_expression(self, ctx):
        if False:
            return 10
        pass

    def enterAssignment_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitAssignment_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterLvalue(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitLvalue(self, ctx):
        if False:
            return 10
        pass

    def enterAssignment_operator(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitAssignment_operator(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterConditional_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitConditional_expression(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterLogical_or_expression(self, ctx):
        if False:
            return 10
        pass

    def exitLogical_or_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterLogical_and_expression(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitLogical_and_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterInclusive_or_expression(self, ctx):
        if False:
            return 10
        pass

    def exitInclusive_or_expression(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterExclusive_or_expression(self, ctx):
        if False:
            return 10
        pass

    def exitExclusive_or_expression(self, ctx):
        if False:
            return 10
        pass

    def enterAnd_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitAnd_expression(self, ctx):
        if False:
            return 10
        pass

    def enterEquality_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitEquality_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterRelational_expression(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitRelational_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterShift_expression(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitShift_expression(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterStatement(self, ctx):
        if False:
            return 10
        pass

    def exitStatement(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterAsm2_statement(self, ctx):
        if False:
            return 10
        pass

    def exitAsm2_statement(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterAsm1_statement(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitAsm1_statement(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def enterAsm_statement(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitAsm_statement(self, ctx):
        if False:
            print('Hello World!')
        pass

    def enterMacro_statement(self, ctx):
        if False:
            i = 10
            return i + 15
        pass

    def exitMacro_statement(self, ctx):
        if False:
            return 10
        pass

    def enterLabeled_statement(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitLabeled_statement(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterCompound_statement(self, ctx):
        if False:
            return 10
        pass

    def exitCompound_statement(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterStatement_list(self, ctx):
        if False:
            return 10
        pass

    def exitStatement_list(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def enterExpression_statement(self, ctx):
        if False:
            print('Hello World!')
        pass

    def exitExpression_statement(self, ctx):
        if False:
            return 10
        pass

    def enterSelection_statement(self, ctx):
        if False:
            return 10
        pass

    def exitSelection_statement(self, ctx):
        if False:
            return 10
        pass

    def enterIteration_statement(self, ctx):
        if False:
            return 10
        pass

    def exitIteration_statement(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterJump_statement(self, ctx):
        if False:
            while True:
                i = 10
        pass

    def exitJump_statement(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        pass