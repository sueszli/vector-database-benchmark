from antlr4 import *
if __name__ is not None and '.' in __name__:
    from .autolevparser import AutolevParser
else:
    from autolevparser import AutolevParser

class AutolevListener(ParseTreeListener):

    def enterProg(self, ctx: AutolevParser.ProgContext):
        if False:
            return 10
        pass

    def exitProg(self, ctx: AutolevParser.ProgContext):
        if False:
            i = 10
            return i + 15
        pass

    def enterStat(self, ctx: AutolevParser.StatContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitStat(self, ctx: AutolevParser.StatContext):
        if False:
            while True:
                i = 10
        pass

    def enterVecAssign(self, ctx: AutolevParser.VecAssignContext):
        if False:
            return 10
        pass

    def exitVecAssign(self, ctx: AutolevParser.VecAssignContext):
        if False:
            while True:
                i = 10
        pass

    def enterIndexAssign(self, ctx: AutolevParser.IndexAssignContext):
        if False:
            while True:
                i = 10
        pass

    def exitIndexAssign(self, ctx: AutolevParser.IndexAssignContext):
        if False:
            i = 10
            return i + 15
        pass

    def enterRegularAssign(self, ctx: AutolevParser.RegularAssignContext):
        if False:
            print('Hello World!')
        pass

    def exitRegularAssign(self, ctx: AutolevParser.RegularAssignContext):
        if False:
            while True:
                i = 10
        pass

    def enterEquals(self, ctx: AutolevParser.EqualsContext):
        if False:
            while True:
                i = 10
        pass

    def exitEquals(self, ctx: AutolevParser.EqualsContext):
        if False:
            i = 10
            return i + 15
        pass

    def enterIndex(self, ctx: AutolevParser.IndexContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitIndex(self, ctx: AutolevParser.IndexContext):
        if False:
            while True:
                i = 10
        pass

    def enterDiff(self, ctx: AutolevParser.DiffContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitDiff(self, ctx: AutolevParser.DiffContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterFunctionCall(self, ctx: AutolevParser.FunctionCallContext):
        if False:
            return 10
        pass

    def exitFunctionCall(self, ctx: AutolevParser.FunctionCallContext):
        if False:
            while True:
                i = 10
        pass

    def enterVarDecl(self, ctx: AutolevParser.VarDeclContext):
        if False:
            while True:
                i = 10
        pass

    def exitVarDecl(self, ctx: AutolevParser.VarDeclContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterVarType(self, ctx: AutolevParser.VarTypeContext):
        if False:
            print('Hello World!')
        pass

    def exitVarType(self, ctx: AutolevParser.VarTypeContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterVarDecl2(self, ctx: AutolevParser.VarDecl2Context):
        if False:
            i = 10
            return i + 15
        pass

    def exitVarDecl2(self, ctx: AutolevParser.VarDecl2Context):
        if False:
            return 10
        pass

    def enterRanges(self, ctx: AutolevParser.RangesContext):
        if False:
            return 10
        pass

    def exitRanges(self, ctx: AutolevParser.RangesContext):
        if False:
            while True:
                i = 10
        pass

    def enterMassDecl(self, ctx: AutolevParser.MassDeclContext):
        if False:
            while True:
                i = 10
        pass

    def exitMassDecl(self, ctx: AutolevParser.MassDeclContext):
        if False:
            return 10
        pass

    def enterMassDecl2(self, ctx: AutolevParser.MassDecl2Context):
        if False:
            while True:
                i = 10
        pass

    def exitMassDecl2(self, ctx: AutolevParser.MassDecl2Context):
        if False:
            print('Hello World!')
        pass

    def enterInertiaDecl(self, ctx: AutolevParser.InertiaDeclContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitInertiaDecl(self, ctx: AutolevParser.InertiaDeclContext):
        if False:
            return 10
        pass

    def enterMatrix(self, ctx: AutolevParser.MatrixContext):
        if False:
            return 10
        pass

    def exitMatrix(self, ctx: AutolevParser.MatrixContext):
        if False:
            while True:
                i = 10
        pass

    def enterMatrixInOutput(self, ctx: AutolevParser.MatrixInOutputContext):
        if False:
            print('Hello World!')
        pass

    def exitMatrixInOutput(self, ctx: AutolevParser.MatrixInOutputContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterCodeCommands(self, ctx: AutolevParser.CodeCommandsContext):
        if False:
            while True:
                i = 10
        pass

    def exitCodeCommands(self, ctx: AutolevParser.CodeCommandsContext):
        if False:
            return 10
        pass

    def enterSettings(self, ctx: AutolevParser.SettingsContext):
        if False:
            print('Hello World!')
        pass

    def exitSettings(self, ctx: AutolevParser.SettingsContext):
        if False:
            while True:
                i = 10
        pass

    def enterUnits(self, ctx: AutolevParser.UnitsContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitUnits(self, ctx: AutolevParser.UnitsContext):
        if False:
            return 10
        pass

    def enterInputs(self, ctx: AutolevParser.InputsContext):
        if False:
            return 10
        pass

    def exitInputs(self, ctx: AutolevParser.InputsContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterId_diff(self, ctx: AutolevParser.Id_diffContext):
        if False:
            return 10
        pass

    def exitId_diff(self, ctx: AutolevParser.Id_diffContext):
        if False:
            return 10
        pass

    def enterInputs2(self, ctx: AutolevParser.Inputs2Context):
        if False:
            while True:
                i = 10
        pass

    def exitInputs2(self, ctx: AutolevParser.Inputs2Context):
        if False:
            print('Hello World!')
        pass

    def enterOutputs(self, ctx: AutolevParser.OutputsContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitOutputs(self, ctx: AutolevParser.OutputsContext):
        if False:
            i = 10
            return i + 15
        pass

    def enterOutputs2(self, ctx: AutolevParser.Outputs2Context):
        if False:
            i = 10
            return i + 15
        pass

    def exitOutputs2(self, ctx: AutolevParser.Outputs2Context):
        if False:
            return 10
        pass

    def enterCodegen(self, ctx: AutolevParser.CodegenContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitCodegen(self, ctx: AutolevParser.CodegenContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterCommands(self, ctx: AutolevParser.CommandsContext):
        if False:
            print('Hello World!')
        pass

    def exitCommands(self, ctx: AutolevParser.CommandsContext):
        if False:
            while True:
                i = 10
        pass

    def enterVec(self, ctx: AutolevParser.VecContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitVec(self, ctx: AutolevParser.VecContext):
        if False:
            i = 10
            return i + 15
        pass

    def enterParens(self, ctx: AutolevParser.ParensContext):
        if False:
            return 10
        pass

    def exitParens(self, ctx: AutolevParser.ParensContext):
        if False:
            while True:
                i = 10
        pass

    def enterVectorOrDyadic(self, ctx: AutolevParser.VectorOrDyadicContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitVectorOrDyadic(self, ctx: AutolevParser.VectorOrDyadicContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterExponent(self, ctx: AutolevParser.ExponentContext):
        if False:
            print('Hello World!')
        pass

    def exitExponent(self, ctx: AutolevParser.ExponentContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def enterMulDiv(self, ctx: AutolevParser.MulDivContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitMulDiv(self, ctx: AutolevParser.MulDivContext):
        if False:
            return 10
        pass

    def enterAddSub(self, ctx: AutolevParser.AddSubContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitAddSub(self, ctx: AutolevParser.AddSubContext):
        if False:
            while True:
                i = 10
        pass

    def enterFloat(self, ctx: AutolevParser.FloatContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitFloat(self, ctx: AutolevParser.FloatContext):
        if False:
            while True:
                i = 10
        pass

    def enterInt(self, ctx: AutolevParser.IntContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitInt(self, ctx: AutolevParser.IntContext):
        if False:
            return 10
        pass

    def enterIdEqualsExpr(self, ctx: AutolevParser.IdEqualsExprContext):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exitIdEqualsExpr(self, ctx: AutolevParser.IdEqualsExprContext):
        if False:
            while True:
                i = 10
        pass

    def enterNegativeOne(self, ctx: AutolevParser.NegativeOneContext):
        if False:
            return 10
        pass

    def exitNegativeOne(self, ctx: AutolevParser.NegativeOneContext):
        if False:
            i = 10
            return i + 15
        pass

    def enterFunction(self, ctx: AutolevParser.FunctionContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitFunction(self, ctx: AutolevParser.FunctionContext):
        if False:
            print('Hello World!')
        pass

    def enterRangess(self, ctx: AutolevParser.RangessContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitRangess(self, ctx: AutolevParser.RangessContext):
        if False:
            while True:
                i = 10
        pass

    def enterColon(self, ctx: AutolevParser.ColonContext):
        if False:
            while True:
                i = 10
        pass

    def exitColon(self, ctx: AutolevParser.ColonContext):
        if False:
            while True:
                i = 10
        pass

    def enterId(self, ctx: AutolevParser.IdContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitId(self, ctx: AutolevParser.IdContext):
        if False:
            return 10
        pass

    def enterExp(self, ctx: AutolevParser.ExpContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitExp(self, ctx: AutolevParser.ExpContext):
        if False:
            return 10
        pass

    def enterMatrices(self, ctx: AutolevParser.MatricesContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitMatrices(self, ctx: AutolevParser.MatricesContext):
        if False:
            while True:
                i = 10
        pass

    def enterIndexing(self, ctx: AutolevParser.IndexingContext):
        if False:
            i = 10
            return i + 15
        pass

    def exitIndexing(self, ctx: AutolevParser.IndexingContext):
        if False:
            while True:
                i = 10
        pass
del AutolevParser