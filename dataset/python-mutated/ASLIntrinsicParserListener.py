from antlr4 import *
if '.' in __name__:
    from .ASLIntrinsicParser import ASLIntrinsicParser
else:
    from ASLIntrinsicParser import ASLIntrinsicParser

class ASLIntrinsicParserListener(ParseTreeListener):

    def enterFunc_decl(self, ctx: ASLIntrinsicParser.Func_declContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitFunc_decl(self, ctx: ASLIntrinsicParser.Func_declContext):
        if False:
            i = 10
            return i + 15
        pass

    def enterStates_func_decl(self, ctx: ASLIntrinsicParser.States_func_declContext):
        if False:
            return 10
        pass

    def exitStates_func_decl(self, ctx: ASLIntrinsicParser.States_func_declContext):
        if False:
            i = 10
            return i + 15
        pass

    def enterState_fun_name(self, ctx: ASLIntrinsicParser.State_fun_nameContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitState_fun_name(self, ctx: ASLIntrinsicParser.State_fun_nameContext):
        if False:
            while True:
                i = 10
        pass

    def enterFunc_arg_list(self, ctx: ASLIntrinsicParser.Func_arg_listContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitFunc_arg_list(self, ctx: ASLIntrinsicParser.Func_arg_listContext):
        if False:
            return 10
        pass

    def enterFunc_arg_string(self, ctx: ASLIntrinsicParser.Func_arg_stringContext):
        if False:
            print('Hello World!')
        pass

    def exitFunc_arg_string(self, ctx: ASLIntrinsicParser.Func_arg_stringContext):
        if False:
            print('Hello World!')
        pass

    def enterFunc_arg_int(self, ctx: ASLIntrinsicParser.Func_arg_intContext):
        if False:
            return 10
        pass

    def exitFunc_arg_int(self, ctx: ASLIntrinsicParser.Func_arg_intContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterFunc_arg_float(self, ctx: ASLIntrinsicParser.Func_arg_floatContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitFunc_arg_float(self, ctx: ASLIntrinsicParser.Func_arg_floatContext):
        if False:
            print('Hello World!')
        pass

    def enterFunc_arg_bool(self, ctx: ASLIntrinsicParser.Func_arg_boolContext):
        if False:
            print('Hello World!')
        pass

    def exitFunc_arg_bool(self, ctx: ASLIntrinsicParser.Func_arg_boolContext):
        if False:
            while True:
                i = 10
        pass

    def enterFunc_arg_json_path(self, ctx: ASLIntrinsicParser.Func_arg_json_pathContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitFunc_arg_json_path(self, ctx: ASLIntrinsicParser.Func_arg_json_pathContext):
        if False:
            return 10
        pass

    def enterFunc_arg_func_decl(self, ctx: ASLIntrinsicParser.Func_arg_func_declContext):
        if False:
            return 10
        pass

    def exitFunc_arg_func_decl(self, ctx: ASLIntrinsicParser.Func_arg_func_declContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterJson_path(self, ctx: ASLIntrinsicParser.Json_pathContext):
        if False:
            print('Hello World!')
        pass

    def exitJson_path(self, ctx: ASLIntrinsicParser.Json_pathContext):
        if False:
            print('Hello World!')
        pass

    def enterJson_path_part(self, ctx: ASLIntrinsicParser.Json_path_partContext):
        if False:
            print('Hello World!')
        pass

    def exitJson_path_part(self, ctx: ASLIntrinsicParser.Json_path_partContext):
        if False:
            i = 10
            return i + 15
        pass

    def enterJson_path_iden(self, ctx: ASLIntrinsicParser.Json_path_idenContext):
        if False:
            return 10
        pass

    def exitJson_path_iden(self, ctx: ASLIntrinsicParser.Json_path_idenContext):
        if False:
            while True:
                i = 10
        pass

    def enterJson_path_iden_qual(self, ctx: ASLIntrinsicParser.Json_path_iden_qualContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitJson_path_iden_qual(self, ctx: ASLIntrinsicParser.Json_path_iden_qualContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterJson_path_qual_void(self, ctx: ASLIntrinsicParser.Json_path_qual_voidContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitJson_path_qual_void(self, ctx: ASLIntrinsicParser.Json_path_qual_voidContext):
        if False:
            print('Hello World!')
        pass

    def enterJson_path_qual_idx(self, ctx: ASLIntrinsicParser.Json_path_qual_idxContext):
        if False:
            print('Hello World!')
        pass

    def exitJson_path_qual_idx(self, ctx: ASLIntrinsicParser.Json_path_qual_idxContext):
        if False:
            print('Hello World!')
        pass

    def enterJson_path_qual_query(self, ctx: ASLIntrinsicParser.Json_path_qual_queryContext):
        if False:
            while True:
                i = 10
        pass

    def exitJson_path_qual_query(self, ctx: ASLIntrinsicParser.Json_path_qual_queryContext):
        if False:
            print('Hello World!')
        pass

    def enterJson_path_query_cmp(self, ctx: ASLIntrinsicParser.Json_path_query_cmpContext):
        if False:
            print('Hello World!')
        pass

    def exitJson_path_query_cmp(self, ctx: ASLIntrinsicParser.Json_path_query_cmpContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterJson_path_query_length(self, ctx: ASLIntrinsicParser.Json_path_query_lengthContext):
        if False:
            return 10
        pass

    def exitJson_path_query_length(self, ctx: ASLIntrinsicParser.Json_path_query_lengthContext):
        if False:
            print('Hello World!')
        pass

    def enterJson_path_query_binary(self, ctx: ASLIntrinsicParser.Json_path_query_binaryContext):
        if False:
            print('Hello World!')
        pass

    def exitJson_path_query_binary(self, ctx: ASLIntrinsicParser.Json_path_query_binaryContext):
        if False:
            while True:
                i = 10
        pass

    def enterJson_path_query_star(self, ctx: ASLIntrinsicParser.Json_path_query_starContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitJson_path_query_star(self, ctx: ASLIntrinsicParser.Json_path_query_starContext):
        if False:
            return 10
        pass

    def enterIdentifier(self, ctx: ASLIntrinsicParser.IdentifierContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitIdentifier(self, ctx: ASLIntrinsicParser.IdentifierContext):
        if False:
            while True:
                i = 10
        pass
del ASLIntrinsicParser