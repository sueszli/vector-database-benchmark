from antlr4 import *
if '.' in __name__:
    from .ASLParser import ASLParser
else:
    from ASLParser import ASLParser

class ASLParserVisitor(ParseTreeVisitor):

    def visitProgram_decl(self, ctx: ASLParser.Program_declContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitTop_layer_stmt(self, ctx: ASLParser.Top_layer_stmtContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitStartat_decl(self, ctx: ASLParser.Startat_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitComment_decl(self, ctx: ASLParser.Comment_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitState_stmt(self, ctx: ASLParser.State_stmtContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitStates_decl(self, ctx: ASLParser.States_declContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitState_name(self, ctx: ASLParser.State_nameContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitState_decl(self, ctx: ASLParser.State_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitState_decl_body(self, ctx: ASLParser.State_decl_bodyContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitType_decl(self, ctx: ASLParser.Type_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitNext_decl(self, ctx: ASLParser.Next_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitResource_decl(self, ctx: ASLParser.Resource_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitInput_path_decl(self, ctx: ASLParser.Input_path_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitResult_decl(self, ctx: ASLParser.Result_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitResult_path_decl(self, ctx: ASLParser.Result_path_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitOutput_path_decl(self, ctx: ASLParser.Output_path_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitEnd_decl(self, ctx: ASLParser.End_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitDefault_decl(self, ctx: ASLParser.Default_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitError_decl(self, ctx: ASLParser.Error_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitCause_decl(self, ctx: ASLParser.Cause_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitSeconds_decl(self, ctx: ASLParser.Seconds_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitSeconds_path_decl(self, ctx: ASLParser.Seconds_path_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitTimestamp_decl(self, ctx: ASLParser.Timestamp_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitTimestamp_path_decl(self, ctx: ASLParser.Timestamp_path_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitItems_path_decl(self, ctx: ASLParser.Items_path_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitMax_concurrency_decl(self, ctx: ASLParser.Max_concurrency_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitParameters_decl(self, ctx: ASLParser.Parameters_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitTimeout_seconds_decl(self, ctx: ASLParser.Timeout_seconds_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitTimeout_seconds_path_decl(self, ctx: ASLParser.Timeout_seconds_path_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitHeartbeat_seconds_decl(self, ctx: ASLParser.Heartbeat_seconds_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitHeartbeat_seconds_path_decl(self, ctx: ASLParser.Heartbeat_seconds_path_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitPayload_tmpl_decl(self, ctx: ASLParser.Payload_tmpl_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitPayload_binding_path(self, ctx: ASLParser.Payload_binding_pathContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitPayload_binding_path_context_obj(self, ctx: ASLParser.Payload_binding_path_context_objContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitPayload_binding_intrinsic_func(self, ctx: ASLParser.Payload_binding_intrinsic_funcContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitPayload_binding_value(self, ctx: ASLParser.Payload_binding_valueContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitIntrinsic_func(self, ctx: ASLParser.Intrinsic_funcContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitPayload_arr_decl(self, ctx: ASLParser.Payload_arr_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitPayload_value_decl(self, ctx: ASLParser.Payload_value_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitPayload_value_float(self, ctx: ASLParser.Payload_value_floatContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitPayload_value_int(self, ctx: ASLParser.Payload_value_intContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitPayload_value_bool(self, ctx: ASLParser.Payload_value_boolContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitPayload_value_null(self, ctx: ASLParser.Payload_value_nullContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitPayload_value_str(self, ctx: ASLParser.Payload_value_strContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitResult_selector_decl(self, ctx: ASLParser.Result_selector_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitState_type(self, ctx: ASLParser.State_typeContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitChoices_decl(self, ctx: ASLParser.Choices_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitChoice_rule_comparison_variable(self, ctx: ASLParser.Choice_rule_comparison_variableContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitChoice_rule_comparison_composite(self, ctx: ASLParser.Choice_rule_comparison_compositeContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitComparison_variable_stmt(self, ctx: ASLParser.Comparison_variable_stmtContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitComparison_composite_stmt(self, ctx: ASLParser.Comparison_composite_stmtContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitComparison_composite(self, ctx: ASLParser.Comparison_compositeContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitVariable_decl(self, ctx: ASLParser.Variable_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitComparison_func(self, ctx: ASLParser.Comparison_funcContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitBranches_decl(self, ctx: ASLParser.Branches_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitItem_processor_decl(self, ctx: ASLParser.Item_processor_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitItem_processor_item(self, ctx: ASLParser.Item_processor_itemContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitProcessor_config_decl(self, ctx: ASLParser.Processor_config_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitProcessor_config_field(self, ctx: ASLParser.Processor_config_fieldContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitMode_decl(self, ctx: ASLParser.Mode_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitMode_type(self, ctx: ASLParser.Mode_typeContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitExecution_decl(self, ctx: ASLParser.Execution_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitExecution_type(self, ctx: ASLParser.Execution_typeContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitIterator_decl(self, ctx: ASLParser.Iterator_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitIterator_decl_item(self, ctx: ASLParser.Iterator_decl_itemContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitItem_selector_decl(self, ctx: ASLParser.Item_selector_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitItem_reader_decl(self, ctx: ASLParser.Item_reader_declContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitItems_reader_field(self, ctx: ASLParser.Items_reader_fieldContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitReader_config_decl(self, ctx: ASLParser.Reader_config_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitReader_config_field(self, ctx: ASLParser.Reader_config_fieldContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitInput_type_decl(self, ctx: ASLParser.Input_type_declContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitCsv_header_location_decl(self, ctx: ASLParser.Csv_header_location_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitCsv_headers_decl(self, ctx: ASLParser.Csv_headers_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitMax_items_decl(self, ctx: ASLParser.Max_items_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitMax_items_path_decl(self, ctx: ASLParser.Max_items_path_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitRetry_decl(self, ctx: ASLParser.Retry_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitRetrier_decl(self, ctx: ASLParser.Retrier_declContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitRetrier_stmt(self, ctx: ASLParser.Retrier_stmtContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitError_equals_decl(self, ctx: ASLParser.Error_equals_declContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitInterval_seconds_decl(self, ctx: ASLParser.Interval_seconds_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitMax_attempts_decl(self, ctx: ASLParser.Max_attempts_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitBackoff_rate_decl(self, ctx: ASLParser.Backoff_rate_declContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitCatch_decl(self, ctx: ASLParser.Catch_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitCatcher_decl(self, ctx: ASLParser.Catcher_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitCatcher_stmt(self, ctx: ASLParser.Catcher_stmtContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitComparison_op(self, ctx: ASLParser.Comparison_opContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitChoice_operator(self, ctx: ASLParser.Choice_operatorContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitStates_error_name(self, ctx: ASLParser.States_error_nameContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitError_name(self, ctx: ASLParser.Error_nameContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitJson_obj_decl(self, ctx: ASLParser.Json_obj_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitJson_binding(self, ctx: ASLParser.Json_bindingContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitJson_arr_decl(self, ctx: ASLParser.Json_arr_declContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitJson_value_decl(self, ctx: ASLParser.Json_value_declContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitKeyword_or_string(self, ctx: ASLParser.Keyword_or_stringContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)
del ASLParser