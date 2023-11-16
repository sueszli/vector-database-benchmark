from antlr4 import *
if '.' in __name__:
    from .HogQLParser import HogQLParser
else:
    from HogQLParser import HogQLParser

class HogQLParserVisitor(ParseTreeVisitor):

    def visitSelect(self, ctx: HogQLParser.SelectContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitSelectUnionStmt(self, ctx: HogQLParser.SelectUnionStmtContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitSelectStmtWithParens(self, ctx: HogQLParser.SelectStmtWithParensContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitSelectStmt(self, ctx: HogQLParser.SelectStmtContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitWithClause(self, ctx: HogQLParser.WithClauseContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitTopClause(self, ctx: HogQLParser.TopClauseContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitFromClause(self, ctx: HogQLParser.FromClauseContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitArrayJoinClause(self, ctx: HogQLParser.ArrayJoinClauseContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitWindowClause(self, ctx: HogQLParser.WindowClauseContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitPrewhereClause(self, ctx: HogQLParser.PrewhereClauseContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitWhereClause(self, ctx: HogQLParser.WhereClauseContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitGroupByClause(self, ctx: HogQLParser.GroupByClauseContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitHavingClause(self, ctx: HogQLParser.HavingClauseContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitOrderByClause(self, ctx: HogQLParser.OrderByClauseContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitProjectionOrderByClause(self, ctx: HogQLParser.ProjectionOrderByClauseContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitLimitAndOffsetClause(self, ctx: HogQLParser.LimitAndOffsetClauseContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitOffsetOnlyClause(self, ctx: HogQLParser.OffsetOnlyClauseContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitSettingsClause(self, ctx: HogQLParser.SettingsClauseContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitJoinExprOp(self, ctx: HogQLParser.JoinExprOpContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitJoinExprTable(self, ctx: HogQLParser.JoinExprTableContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitJoinExprParens(self, ctx: HogQLParser.JoinExprParensContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitJoinExprCrossOp(self, ctx: HogQLParser.JoinExprCrossOpContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitJoinOpInner(self, ctx: HogQLParser.JoinOpInnerContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitJoinOpLeftRight(self, ctx: HogQLParser.JoinOpLeftRightContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitJoinOpFull(self, ctx: HogQLParser.JoinOpFullContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitJoinOpCross(self, ctx: HogQLParser.JoinOpCrossContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitJoinConstraintClause(self, ctx: HogQLParser.JoinConstraintClauseContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitSampleClause(self, ctx: HogQLParser.SampleClauseContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitOrderExprList(self, ctx: HogQLParser.OrderExprListContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitOrderExpr(self, ctx: HogQLParser.OrderExprContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitRatioExpr(self, ctx: HogQLParser.RatioExprContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitSettingExprList(self, ctx: HogQLParser.SettingExprListContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitSettingExpr(self, ctx: HogQLParser.SettingExprContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitWindowExpr(self, ctx: HogQLParser.WindowExprContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitWinPartitionByClause(self, ctx: HogQLParser.WinPartitionByClauseContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitWinOrderByClause(self, ctx: HogQLParser.WinOrderByClauseContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitWinFrameClause(self, ctx: HogQLParser.WinFrameClauseContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitFrameStart(self, ctx: HogQLParser.FrameStartContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitFrameBetween(self, ctx: HogQLParser.FrameBetweenContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitWinFrameBound(self, ctx: HogQLParser.WinFrameBoundContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitExpr(self, ctx: HogQLParser.ExprContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitColumnTypeExprSimple(self, ctx: HogQLParser.ColumnTypeExprSimpleContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitColumnTypeExprNested(self, ctx: HogQLParser.ColumnTypeExprNestedContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitColumnTypeExprEnum(self, ctx: HogQLParser.ColumnTypeExprEnumContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitColumnTypeExprComplex(self, ctx: HogQLParser.ColumnTypeExprComplexContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitColumnTypeExprParam(self, ctx: HogQLParser.ColumnTypeExprParamContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitColumnExprList(self, ctx: HogQLParser.ColumnExprListContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnExprTernaryOp(self, ctx: HogQLParser.ColumnExprTernaryOpContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitColumnExprAlias(self, ctx: HogQLParser.ColumnExprAliasContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnExprExtract(self, ctx: HogQLParser.ColumnExprExtractContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitColumnExprNegate(self, ctx: HogQLParser.ColumnExprNegateContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnExprSubquery(self, ctx: HogQLParser.ColumnExprSubqueryContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitColumnExprLiteral(self, ctx: HogQLParser.ColumnExprLiteralContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitColumnExprArray(self, ctx: HogQLParser.ColumnExprArrayContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitColumnExprSubstring(self, ctx: HogQLParser.ColumnExprSubstringContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnExprCast(self, ctx: HogQLParser.ColumnExprCastContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnExprOr(self, ctx: HogQLParser.ColumnExprOrContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitColumnExprPrecedence1(self, ctx: HogQLParser.ColumnExprPrecedence1Context):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitColumnExprPrecedence2(self, ctx: HogQLParser.ColumnExprPrecedence2Context):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitColumnExprPrecedence3(self, ctx: HogQLParser.ColumnExprPrecedence3Context):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnExprInterval(self, ctx: HogQLParser.ColumnExprIntervalContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnExprIsNull(self, ctx: HogQLParser.ColumnExprIsNullContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnExprWinFunctionTarget(self, ctx: HogQLParser.ColumnExprWinFunctionTargetContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnExprTrim(self, ctx: HogQLParser.ColumnExprTrimContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnExprTagElement(self, ctx: HogQLParser.ColumnExprTagElementContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnExprTuple(self, ctx: HogQLParser.ColumnExprTupleContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnExprArrayAccess(self, ctx: HogQLParser.ColumnExprArrayAccessContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnExprBetween(self, ctx: HogQLParser.ColumnExprBetweenContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitColumnExprPropertyAccess(self, ctx: HogQLParser.ColumnExprPropertyAccessContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitColumnExprParens(self, ctx: HogQLParser.ColumnExprParensContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitColumnExprTimestamp(self, ctx: HogQLParser.ColumnExprTimestampContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnExprNullish(self, ctx: HogQLParser.ColumnExprNullishContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitColumnExprAnd(self, ctx: HogQLParser.ColumnExprAndContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnExprTupleAccess(self, ctx: HogQLParser.ColumnExprTupleAccessContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitColumnExprCase(self, ctx: HogQLParser.ColumnExprCaseContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnExprDate(self, ctx: HogQLParser.ColumnExprDateContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitColumnExprNot(self, ctx: HogQLParser.ColumnExprNotContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnExprWinFunction(self, ctx: HogQLParser.ColumnExprWinFunctionContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnExprIdentifier(self, ctx: HogQLParser.ColumnExprIdentifierContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitColumnExprFunction(self, ctx: HogQLParser.ColumnExprFunctionContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitColumnExprAsterisk(self, ctx: HogQLParser.ColumnExprAsteriskContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitColumnArgList(self, ctx: HogQLParser.ColumnArgListContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitColumnArgExpr(self, ctx: HogQLParser.ColumnArgExprContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnLambdaExpr(self, ctx: HogQLParser.ColumnLambdaExprContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitHogqlxTagElementClosed(self, ctx: HogQLParser.HogqlxTagElementClosedContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitHogqlxTagElementNested(self, ctx: HogQLParser.HogqlxTagElementNestedContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitHogqlxTagAttribute(self, ctx: HogQLParser.HogqlxTagAttributeContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitWithExprList(self, ctx: HogQLParser.WithExprListContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitWithExprSubquery(self, ctx: HogQLParser.WithExprSubqueryContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitWithExprColumn(self, ctx: HogQLParser.WithExprColumnContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitColumnIdentifier(self, ctx: HogQLParser.ColumnIdentifierContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitNestedIdentifier(self, ctx: HogQLParser.NestedIdentifierContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitTableExprTag(self, ctx: HogQLParser.TableExprTagContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitTableExprIdentifier(self, ctx: HogQLParser.TableExprIdentifierContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitTableExprPlaceholder(self, ctx: HogQLParser.TableExprPlaceholderContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitTableExprSubquery(self, ctx: HogQLParser.TableExprSubqueryContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitTableExprAlias(self, ctx: HogQLParser.TableExprAliasContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitTableExprFunction(self, ctx: HogQLParser.TableExprFunctionContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitTableFunctionExpr(self, ctx: HogQLParser.TableFunctionExprContext):
        if False:
            i = 10
            return i + 15
        return self.visitChildren(ctx)

    def visitTableIdentifier(self, ctx: HogQLParser.TableIdentifierContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitTableArgList(self, ctx: HogQLParser.TableArgListContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitDatabaseIdentifier(self, ctx: HogQLParser.DatabaseIdentifierContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitFloatingLiteral(self, ctx: HogQLParser.FloatingLiteralContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitNumberLiteral(self, ctx: HogQLParser.NumberLiteralContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitLiteral(self, ctx: HogQLParser.LiteralContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitInterval(self, ctx: HogQLParser.IntervalContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitKeyword(self, ctx: HogQLParser.KeywordContext):
        if False:
            print('Hello World!')
        return self.visitChildren(ctx)

    def visitKeywordForAlias(self, ctx: HogQLParser.KeywordForAliasContext):
        if False:
            return 10
        return self.visitChildren(ctx)

    def visitAlias(self, ctx: HogQLParser.AliasContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)

    def visitIdentifier(self, ctx: HogQLParser.IdentifierContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitEnumValue(self, ctx: HogQLParser.EnumValueContext):
        if False:
            for i in range(10):
                print('nop')
        return self.visitChildren(ctx)

    def visitPlaceholder(self, ctx: HogQLParser.PlaceholderContext):
        if False:
            while True:
                i = 10
        return self.visitChildren(ctx)
del HogQLParser