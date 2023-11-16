from antlr4 import *
if '.' in __name__:
    from .ASLIntrinsicParser import ASLIntrinsicParser
else:
    from ASLIntrinsicParser import ASLIntrinsicParser

class ASLIntrinsicParserVisitor(ParseTreeVisitor):

    def visitFunc_decl(self, ctx: ASLIntrinsicParser.Func_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitStates_func_decl(self, ctx: ASLIntrinsicParser.States_func_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitState_fun_name(self, ctx: ASLIntrinsicParser.State_fun_nameContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitFunc_arg_list(self, ctx: ASLIntrinsicParser.Func_arg_listContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitFunc_arg_string(self, ctx: ASLIntrinsicParser.Func_arg_stringContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitFunc_arg_int(self, ctx: ASLIntrinsicParser.Func_arg_intContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitFunc_arg_float(self, ctx: ASLIntrinsicParser.Func_arg_floatContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitFunc_arg_bool(self, ctx: ASLIntrinsicParser.Func_arg_boolContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitFunc_arg_json_path(self, ctx: ASLIntrinsicParser.Func_arg_json_pathContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitFunc_arg_func_decl(self, ctx: ASLIntrinsicParser.Func_arg_func_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitJson_path(self, ctx: ASLIntrinsicParser.Json_pathContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitJson_path_part(self, ctx: ASLIntrinsicParser.Json_path_partContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitJson_path_iden(self, ctx: ASLIntrinsicParser.Json_path_idenContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitJson_path_iden_qual(self, ctx: ASLIntrinsicParser.Json_path_iden_qualContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitJson_path_qual_void(self, ctx: ASLIntrinsicParser.Json_path_qual_voidContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitJson_path_qual_idx(self, ctx: ASLIntrinsicParser.Json_path_qual_idxContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitJson_path_qual_query(self, ctx: ASLIntrinsicParser.Json_path_qual_queryContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitJson_path_query_cmp(self, ctx: ASLIntrinsicParser.Json_path_query_cmpContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitJson_path_query_length(self, ctx: ASLIntrinsicParser.Json_path_query_lengthContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitJson_path_query_binary(self, ctx: ASLIntrinsicParser.Json_path_query_binaryContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitJson_path_query_star(self, ctx: ASLIntrinsicParser.Json_path_query_starContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitIdentifier(self, ctx: ASLIntrinsicParser.IdentifierContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)
del ASLIntrinsicParser