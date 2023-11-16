class Comment:

    def __init__(self, Str, Begin, End, CommentType):
        if False:
            for i in range(10):
                print('nop')
        self.Content = Str
        self.StartPos = Begin
        self.EndPos = End
        self.Type = CommentType

class PP_Directive:

    def __init__(self, Str, Begin, End):
        if False:
            print('Hello World!')
        self.Content = Str
        self.StartPos = Begin
        self.EndPos = End

class AssignmentExpression:

    def __init__(self, Lvalue, Op, Exp, Begin, End):
        if False:
            print('Hello World!')
        self.Name = Lvalue
        self.Operator = Op
        self.Value = Exp
        self.StartPos = Begin
        self.EndPos = End

class PredicateExpression:

    def __init__(self, Str, Begin, End):
        if False:
            i = 10
            return i + 15
        self.Content = Str
        self.StartPos = Begin
        self.EndPos = End

class FunctionDefinition:

    def __init__(self, ModifierStr, DeclStr, Begin, End, LBPos, NamePos):
        if False:
            while True:
                i = 10
        self.Modifier = ModifierStr
        self.Declarator = DeclStr
        self.StartPos = Begin
        self.EndPos = End
        self.LeftBracePos = LBPos
        self.NamePos = NamePos

class VariableDeclaration:

    def __init__(self, ModifierStr, DeclStr, Begin, End):
        if False:
            i = 10
            return i + 15
        self.Modifier = ModifierStr
        self.Declarator = DeclStr
        self.StartPos = Begin
        self.EndPos = End

class EnumerationDefinition:

    def __init__(self, Str, Begin, End):
        if False:
            return 10
        self.Content = Str
        self.StartPos = Begin
        self.EndPos = End

class StructUnionDefinition:

    def __init__(self, Str, Begin, End):
        if False:
            print('Hello World!')
        self.Content = Str
        self.StartPos = Begin
        self.EndPos = End

class TypedefDefinition:

    def __init__(self, FromStr, ToStr, Begin, End):
        if False:
            print('Hello World!')
        self.FromType = FromStr
        self.ToType = ToStr
        self.StartPos = Begin
        self.EndPos = End

class FunctionCalling:

    def __init__(self, Name, Param, Begin, End):
        if False:
            print('Hello World!')
        self.FuncName = Name
        self.ParamList = Param
        self.StartPos = Begin
        self.EndPos = End