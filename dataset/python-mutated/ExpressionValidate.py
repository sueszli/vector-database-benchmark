"""
ExpressionValidate
"""
from __future__ import print_function
import re
from Logger import StringTable as ST

def IsValidBareCString(String):
    if False:
        i = 10
        return i + 15
    EscapeList = ['n', 't', 'f', 'r', 'b', '0', '\\', '"']
    PreChar = ''
    LastChar = ''
    for Char in String:
        LastChar = Char
        if PreChar == '\\':
            if Char not in EscapeList:
                return False
            if Char == '\\':
                PreChar = ''
                continue
        else:
            IntChar = ord(Char)
            if IntChar != 32 and IntChar != 9 and (IntChar != 33) and (IntChar < 35 or IntChar > 126):
                return False
        PreChar = Char
    if LastChar == '\\' and PreChar == LastChar:
        return False
    return True

def _ValidateToken(Token):
    if False:
        print('Hello World!')
    Token = Token.strip()
    Index = Token.find('"')
    if Index != -1:
        return IsValidBareCString(Token[Index + 1:-1])
    return True

class _ExprError(Exception):

    def __init__(self, Error=''):
        if False:
            return 10
        Exception.__init__(self)
        self.Error = Error

class _ExprBase:
    HEX_PATTERN = '[\t\\s]*0[xX][a-fA-F0-9]+'
    INT_PATTERN = '[\t\\s]*[0-9]+'
    MACRO_PATTERN = '[\t\\s]*\\$\\(([A-Z][_A-Z0-9]*)\\)'
    PCD_PATTERN = '[\t\\s]*[_a-zA-Z][a-zA-Z0-9_]*[\t\\s]*\\.[\t\\s]*[_a-zA-Z][a-zA-Z0-9_]*'
    QUOTED_PATTERN = '[\t\\s]*L?"[^"]*"'
    BOOL_PATTERN = '[\t\\s]*(true|True|TRUE|false|False|FALSE)'

    def __init__(self, Token):
        if False:
            return 10
        self.Token = Token
        self.Index = 0
        self.Len = len(Token)

    def SkipWhitespace(self):
        if False:
            while True:
                i = 10
        for Char in self.Token[self.Index:]:
            if Char not in ' \t':
                break
            self.Index += 1

    def IsCurrentOp(self, OpList):
        if False:
            i = 10
            return i + 15
        self.SkipWhitespace()
        LetterOp = ['EQ', 'NE', 'GE', 'LE', 'GT', 'LT', 'NOT', 'and', 'AND', 'or', 'OR', 'XOR']
        OpMap = {'|': '|', '&': '&', '!': '=', '>': '=', '<': '='}
        for Operator in OpList:
            if not self.Token[self.Index:].startswith(Operator):
                continue
            self.Index += len(Operator)
            Char = self.Token[self.Index:self.Index + 1]
            if Operator in LetterOp and (Char == '_' or Char.isalnum()) or (Operator in OpMap and OpMap[Operator] == Char):
                self.Index -= len(Operator)
                break
            return True
        return False

class _LogicalExpressionParser(_ExprBase):
    STRINGITEM = -1
    LOGICAL = 0
    REALLOGICAL = 2
    ARITH = 1

    def __init__(self, Token):
        if False:
            for i in range(10):
                print('nop')
        _ExprBase.__init__(self, Token)
        self.Parens = 0

    def _CheckToken(self, MatchList):
        if False:
            while True:
                i = 10
        for Match in MatchList:
            if Match and Match.start() == 0:
                if not _ValidateToken(self.Token[self.Index:self.Index + Match.end()]):
                    return False
                self.Index += Match.end()
                if self.Token[self.Index - 1] == '"':
                    return True
                if self.Token[self.Index:self.Index + 1] == '_' or self.Token[self.Index:self.Index + 1].isalnum():
                    self.Index -= Match.end()
                    return False
                Token = self.Token[self.Index - Match.end():self.Index]
                if Token.strip() in ['EQ', 'NE', 'GE', 'LE', 'GT', 'LT', 'NOT', 'and', 'AND', 'or', 'OR', 'XOR']:
                    self.Index -= Match.end()
                    return False
                return True
        return False

    def IsAtomicNumVal(self):
        if False:
            for i in range(10):
                print('nop')
        Match1 = re.compile(self.HEX_PATTERN).match(self.Token[self.Index:])
        Match2 = re.compile(self.INT_PATTERN).match(self.Token[self.Index:])
        Match3 = re.compile(self.MACRO_PATTERN).match(self.Token[self.Index:])
        Match4 = re.compile(self.PCD_PATTERN).match(self.Token[self.Index:])
        return self._CheckToken([Match1, Match2, Match3, Match4])

    def IsAtomicItem(self):
        if False:
            while True:
                i = 10
        Match1 = re.compile(self.MACRO_PATTERN).match(self.Token[self.Index:])
        Match2 = re.compile(self.PCD_PATTERN).match(self.Token[self.Index:])
        Match3 = re.compile(self.QUOTED_PATTERN).match(self.Token[self.Index:].replace('\\\\', '//').replace('\\"', "\\'"))
        return self._CheckToken([Match1, Match2, Match3])

    def LogicalExpression(self):
        if False:
            print('Hello World!')
        Ret = self.SpecNot()
        while self.IsCurrentOp(['||', 'OR', 'or', '&&', 'AND', 'and', 'XOR', 'xor', '^']):
            if self.Token[self.Index - 1] == '|' and self.Parens <= 0:
                raise _ExprError(ST.ERR_EXPR_OR % self.Token)
            if Ret not in [self.ARITH, self.LOGICAL, self.REALLOGICAL, self.STRINGITEM]:
                raise _ExprError(ST.ERR_EXPR_LOGICAL % self.Token)
            Ret = self.SpecNot()
            if Ret not in [self.ARITH, self.LOGICAL, self.REALLOGICAL, self.STRINGITEM]:
                raise _ExprError(ST.ERR_EXPR_LOGICAL % self.Token)
            Ret = self.REALLOGICAL
        return Ret

    def SpecNot(self):
        if False:
            i = 10
            return i + 15
        if self.IsCurrentOp(['NOT', '!', 'not']):
            return self.SpecNot()
        return self.Rel()

    def Rel(self):
        if False:
            while True:
                i = 10
        Ret = self.Expr()
        if self.IsCurrentOp(['<=', '>=', '>', '<', 'GT', 'LT', 'GE', 'LE', '==', 'EQ', '!=', 'NE']):
            if Ret == self.STRINGITEM:
                raise _ExprError(ST.ERR_EXPR_LOGICAL % self.Token)
            Ret = self.Expr()
            if Ret == self.REALLOGICAL:
                raise _ExprError(ST.ERR_EXPR_LOGICAL % self.Token)
            Ret = self.REALLOGICAL
        return Ret

    def Expr(self):
        if False:
            for i in range(10):
                print('nop')
        Ret = self.Factor()
        while self.IsCurrentOp(['+', '-', '&', '|', '^', 'XOR', 'xor']):
            if self.Token[self.Index - 1] == '|' and self.Parens <= 0:
                raise _ExprError(ST.ERR_EXPR_OR)
            if Ret == self.STRINGITEM or Ret == self.REALLOGICAL:
                raise _ExprError(ST.ERR_EXPR_LOGICAL % self.Token)
            Ret = self.Factor()
            if Ret == self.STRINGITEM or Ret == self.REALLOGICAL:
                raise _ExprError(ST.ERR_EXPR_LOGICAL % self.Token)
            Ret = self.ARITH
        return Ret

    def Factor(self):
        if False:
            return 10
        if self.IsCurrentOp(['(']):
            self.Parens += 1
            Ret = self.LogicalExpression()
            if not self.IsCurrentOp([')']):
                raise _ExprError(ST.ERR_EXPR_RIGHT_PAREN % (self.Token, self.Token[self.Index:]))
            self.Parens -= 1
            return Ret
        if self.IsAtomicItem():
            if self.Token[self.Index - 1] == '"':
                return self.STRINGITEM
            return self.LOGICAL
        elif self.IsAtomicNumVal():
            return self.ARITH
        else:
            raise _ExprError(ST.ERR_EXPR_FACTOR % (self.Token[self.Index:], self.Token))

    def IsValidLogicalExpression(self):
        if False:
            for i in range(10):
                print('nop')
        if self.Len == 0:
            return (False, ST.ERR_EXPRESS_EMPTY)
        try:
            if self.LogicalExpression() not in [self.ARITH, self.LOGICAL, self.REALLOGICAL, self.STRINGITEM]:
                return (False, ST.ERR_EXPR_LOGICAL % self.Token)
        except _ExprError as XExcept:
            return (False, XExcept.Error)
        self.SkipWhitespace()
        if self.Index != self.Len:
            return (False, ST.ERR_EXPR_BOOLEAN % (self.Token[self.Index:], self.Token))
        return (True, '')

class _ValidRangeExpressionParser(_ExprBase):
    INT_RANGE_PATTERN = '[\t\\s]*[0-9]+[\t\\s]*-[\t\\s]*[0-9]+'
    HEX_RANGE_PATTERN = '[\t\\s]*0[xX][a-fA-F0-9]+[\t\\s]*-[\t\\s]*0[xX][a-fA-F0-9]+'

    def __init__(self, Token):
        if False:
            while True:
                i = 10
        _ExprBase.__init__(self, Token)
        self.Parens = 0
        self.HEX = 1
        self.INT = 2
        self.IsParenHappen = False
        self.IsLogicalOpHappen = False

    def IsValidRangeExpression(self):
        if False:
            i = 10
            return i + 15
        if self.Len == 0:
            return (False, ST.ERR_EXPR_RANGE_EMPTY)
        try:
            if self.RangeExpression() not in [self.HEX, self.INT]:
                return (False, ST.ERR_EXPR_RANGE % self.Token)
        except _ExprError as XExcept:
            return (False, XExcept.Error)
        self.SkipWhitespace()
        if self.Index != self.Len:
            return (False, ST.ERR_EXPR_RANGE % self.Token)
        return (True, '')

    def RangeExpression(self):
        if False:
            print('Hello World!')
        Ret = self.Unary()
        while self.IsCurrentOp(['OR', 'AND', 'and', 'or']):
            self.IsLogicalOpHappen = True
            if not self.IsParenHappen:
                raise _ExprError(ST.ERR_PAREN_NOT_USED % self.Token)
            self.IsParenHappen = False
            Ret = self.Unary()
        if self.IsCurrentOp(['XOR']):
            Ret = self.Unary()
        return Ret

    def Unary(self):
        if False:
            i = 10
            return i + 15
        if self.IsCurrentOp(['NOT']):
            return self.Unary()
        return self.ValidRange()

    def ValidRange(self):
        if False:
            while True:
                i = 10
        Ret = -1
        if self.IsCurrentOp(['(']):
            self.IsLogicalOpHappen = False
            self.IsParenHappen = True
            self.Parens += 1
            if self.Parens > 1:
                raise _ExprError(ST.ERR_EXPR_RANGE_DOUBLE_PAREN_NESTED % self.Token)
            Ret = self.RangeExpression()
            if not self.IsCurrentOp([')']):
                raise _ExprError(ST.ERR_EXPR_RIGHT_PAREN % self.Token)
            self.Parens -= 1
            return Ret
        if self.IsLogicalOpHappen:
            raise _ExprError(ST.ERR_PAREN_NOT_USED % self.Token)
        if self.IsCurrentOp(['LT', 'GT', 'LE', 'GE', 'EQ', 'XOR']):
            IntMatch = re.compile(self.INT_PATTERN).match(self.Token[self.Index:])
            HexMatch = re.compile(self.HEX_PATTERN).match(self.Token[self.Index:])
            if HexMatch and HexMatch.start() == 0:
                self.Index += HexMatch.end()
                Ret = self.HEX
            elif IntMatch and IntMatch.start() == 0:
                self.Index += IntMatch.end()
                Ret = self.INT
            else:
                raise _ExprError(ST.ERR_EXPR_RANGE_FACTOR % (self.Token[self.Index:], self.Token))
        else:
            IntRangeMatch = re.compile(self.INT_RANGE_PATTERN).match(self.Token[self.Index:])
            HexRangeMatch = re.compile(self.HEX_RANGE_PATTERN).match(self.Token[self.Index:])
            if HexRangeMatch and HexRangeMatch.start() == 0:
                self.Index += HexRangeMatch.end()
                Ret = self.HEX
            elif IntRangeMatch and IntRangeMatch.start() == 0:
                self.Index += IntRangeMatch.end()
                Ret = self.INT
            else:
                raise _ExprError(ST.ERR_EXPR_RANGE % self.Token)
        return Ret

class _ValidListExpressionParser(_ExprBase):
    VALID_LIST_PATTERN = '(0[xX][0-9a-fA-F]+|[0-9]+)([\t\\s]*,[\t\\s]*(0[xX][0-9a-fA-F]+|[0-9]+))*'

    def __init__(self, Token):
        if False:
            print('Hello World!')
        _ExprBase.__init__(self, Token)
        self.NUM = 1

    def IsValidListExpression(self):
        if False:
            i = 10
            return i + 15
        if self.Len == 0:
            return (False, ST.ERR_EXPR_LIST_EMPTY)
        try:
            if self.ListExpression() not in [self.NUM]:
                return (False, ST.ERR_EXPR_LIST % self.Token)
        except _ExprError as XExcept:
            return (False, XExcept.Error)
        self.SkipWhitespace()
        if self.Index != self.Len:
            return (False, ST.ERR_EXPR_LIST % self.Token)
        return (True, '')

    def ListExpression(self):
        if False:
            while True:
                i = 10
        Ret = -1
        self.SkipWhitespace()
        ListMatch = re.compile(self.VALID_LIST_PATTERN).match(self.Token[self.Index:])
        if ListMatch and ListMatch.start() == 0:
            self.Index += ListMatch.end()
            Ret = self.NUM
        else:
            raise _ExprError(ST.ERR_EXPR_LIST % self.Token)
        return Ret

class _StringTestParser(_ExprBase):

    def __init__(self, Token):
        if False:
            return 10
        _ExprBase.__init__(self, Token)

    def IsValidStringTest(self):
        if False:
            print('Hello World!')
        if self.Len == 0:
            return (False, ST.ERR_EXPR_EMPTY)
        try:
            self.StringTest()
        except _ExprError as XExcept:
            return (False, XExcept.Error)
        return (True, '')

    def StringItem(self):
        if False:
            print('Hello World!')
        Match1 = re.compile(self.QUOTED_PATTERN).match(self.Token[self.Index:].replace('\\\\', '//').replace('\\"', "\\'"))
        Match2 = re.compile(self.MACRO_PATTERN).match(self.Token[self.Index:])
        Match3 = re.compile(self.PCD_PATTERN).match(self.Token[self.Index:])
        MatchList = [Match1, Match2, Match3]
        for Match in MatchList:
            if Match and Match.start() == 0:
                if not _ValidateToken(self.Token[self.Index:self.Index + Match.end()]):
                    raise _ExprError(ST.ERR_EXPR_STRING_ITEM % (self.Token, self.Token[self.Index:]))
                self.Index += Match.end()
                Token = self.Token[self.Index - Match.end():self.Index]
                if Token.strip() in ['EQ', 'NE']:
                    raise _ExprError(ST.ERR_EXPR_STRING_ITEM % (self.Token, self.Token[self.Index:]))
                return
        else:
            raise _ExprError(ST.ERR_EXPR_STRING_ITEM % (self.Token, self.Token[self.Index:]))

    def StringTest(self):
        if False:
            while True:
                i = 10
        self.StringItem()
        if not self.IsCurrentOp(['==', 'EQ', '!=', 'NE']):
            raise _ExprError(ST.ERR_EXPR_EQUALITY % (self.Token[self.Index:], self.Token))
        self.StringItem()
        if self.Index != self.Len:
            raise _ExprError(ST.ERR_EXPR_BOOLEAN % (self.Token[self.Index:], self.Token))

def IsValidStringTest(Token, Flag=False):
    if False:
        while True:
            i = 10
    if not Flag:
        return (True, '')
    return _StringTestParser(Token).IsValidStringTest()

def IsValidLogicalExpr(Token, Flag=False):
    if False:
        return 10
    if not Flag:
        return (True, '')
    return _LogicalExpressionParser(Token).IsValidLogicalExpression()

def IsValidRangeExpr(Token):
    if False:
        i = 10
        return i + 15
    return _ValidRangeExpressionParser(Token).IsValidRangeExpression()

def IsValidListExpr(Token):
    if False:
        print('Hello World!')
    return _ValidListExpressionParser(Token).IsValidListExpression()

def IsValidFeatureFlagExp(Token, Flag=False):
    if False:
        for i in range(10):
            print('nop')
    if not Flag:
        return (True, '', Token)
    else:
        if Token in ['TRUE', 'FALSE', 'true', 'false', 'True', 'False', '0x1', '0x01', '0x0', '0x00']:
            return (True, '')
        (Valid, Cause) = IsValidStringTest(Token, Flag)
        if not Valid:
            (Valid, Cause) = IsValidLogicalExpr(Token, Flag)
        if not Valid:
            return (False, Cause)
        return (True, '')
if __name__ == '__main__':
    print(_LogicalExpressionParser('gCrownBayTokenSpaceGuid.PcdPciDevice1BridgeAddressLE0').IsValidLogicalExpression())