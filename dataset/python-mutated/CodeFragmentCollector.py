from __future__ import print_function
from __future__ import absolute_import
import re
import Common.LongFilePathOs as os
import sys
if sys.version_info.major == 3:
    import antlr4 as antlr
    from Eot.CParser4.CLexer import CLexer
    from Eot.CParser4.CParser import CParser
else:
    import antlr3 as antlr
    antlr.InputStream = antlr.StringStream
    from Eot.CParser3.CLexer import CLexer
    from Eot.CParser3.CParser import CParser
from Eot import FileProfile
from Eot.CodeFragment import PP_Directive
from Eot.ParserWarning import Warning
(T_CHAR_SPACE, T_CHAR_NULL, T_CHAR_CR, T_CHAR_TAB, T_CHAR_LF, T_CHAR_SLASH, T_CHAR_BACKSLASH, T_CHAR_DOUBLE_QUOTE, T_CHAR_SINGLE_QUOTE, T_CHAR_STAR, T_CHAR_HASH) = (' ', '\x00', '\r', '\t', '\n', '/', '\\', '"', "'", '*', '#')
SEPERATOR_TUPLE = ('=', '|', ',', '{', '}')
(T_COMMENT_TWO_SLASH, T_COMMENT_SLASH_STAR) = (0, 1)
(T_PP_INCLUDE, T_PP_DEFINE, T_PP_OTHERS) = (0, 1, 2)

class CodeFragmentCollector:

    def __init__(self, FileName):
        if False:
            while True:
                i = 10
        self.Profile = FileProfile.FileProfile(FileName)
        self.Profile.FileLinesList.append(T_CHAR_LF)
        self.FileName = FileName
        self.CurrentLineNumber = 1
        self.CurrentOffsetWithinLine = 0
        self.__Token = ''
        self.__SkippedChars = ''

    def __EndOfFile(self):
        if False:
            print('Hello World!')
        NumberOfLines = len(self.Profile.FileLinesList)
        SizeOfLastLine = len(self.Profile.FileLinesList[-1])
        if self.CurrentLineNumber == NumberOfLines and self.CurrentOffsetWithinLine >= SizeOfLastLine - 1:
            return True
        elif self.CurrentLineNumber > NumberOfLines:
            return True
        else:
            return False

    def __EndOfLine(self):
        if False:
            print('Hello World!')
        SizeOfCurrentLine = len(self.Profile.FileLinesList[self.CurrentLineNumber - 1])
        if self.CurrentOffsetWithinLine >= SizeOfCurrentLine - 1:
            return True
        else:
            return False

    def Rewind(self):
        if False:
            i = 10
            return i + 15
        self.CurrentLineNumber = 1
        self.CurrentOffsetWithinLine = 0

    def __UndoOneChar(self):
        if False:
            print('Hello World!')
        if self.CurrentLineNumber == 1 and self.CurrentOffsetWithinLine == 0:
            return False
        elif self.CurrentOffsetWithinLine == 0:
            self.CurrentLineNumber -= 1
            self.CurrentOffsetWithinLine = len(self.__CurrentLine()) - 1
        else:
            self.CurrentOffsetWithinLine -= 1
        return True

    def __GetOneChar(self):
        if False:
            for i in range(10):
                print('nop')
        if self.CurrentOffsetWithinLine == len(self.Profile.FileLinesList[self.CurrentLineNumber - 1]) - 1:
            self.CurrentLineNumber += 1
            self.CurrentOffsetWithinLine = 0
        else:
            self.CurrentOffsetWithinLine += 1

    def __CurrentChar(self):
        if False:
            return 10
        CurrentChar = self.Profile.FileLinesList[self.CurrentLineNumber - 1][self.CurrentOffsetWithinLine]
        return CurrentChar

    def __NextChar(self):
        if False:
            while True:
                i = 10
        if self.CurrentOffsetWithinLine == len(self.Profile.FileLinesList[self.CurrentLineNumber - 1]) - 1:
            return self.Profile.FileLinesList[self.CurrentLineNumber][0]
        else:
            return self.Profile.FileLinesList[self.CurrentLineNumber - 1][self.CurrentOffsetWithinLine + 1]

    def __SetCurrentCharValue(self, Value):
        if False:
            while True:
                i = 10
        self.Profile.FileLinesList[self.CurrentLineNumber - 1][self.CurrentOffsetWithinLine] = Value

    def __SetCharValue(self, Line, Offset, Value):
        if False:
            while True:
                i = 10
        self.Profile.FileLinesList[Line - 1][Offset] = Value

    def __CurrentLine(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Profile.FileLinesList[self.CurrentLineNumber - 1]

    def __InsertComma(self, Line):
        if False:
            print('Hello World!')
        if self.Profile.FileLinesList[Line - 1][0] != T_CHAR_HASH:
            BeforeHashPart = str(self.Profile.FileLinesList[Line - 1]).split(T_CHAR_HASH)[0]
            if BeforeHashPart.rstrip().endswith(T_CHAR_COMMA) or BeforeHashPart.rstrip().endswith(';'):
                return
        if Line - 2 >= 0 and str(self.Profile.FileLinesList[Line - 2]).rstrip().endswith(','):
            return
        if Line - 2 >= 0 and str(self.Profile.FileLinesList[Line - 2]).rstrip().endswith(';'):
            return
        if str(self.Profile.FileLinesList[Line]).lstrip().startswith(',') or str(self.Profile.FileLinesList[Line]).lstrip().startswith(';'):
            return
        self.Profile.FileLinesList[Line - 1].insert(self.CurrentOffsetWithinLine, ',')

    def PreprocessFileWithClear(self):
        if False:
            i = 10
            return i + 15
        self.Rewind()
        InComment = False
        DoubleSlashComment = False
        HashComment = False
        PPExtend = False
        PPDirectiveObj = None
        InString = False
        InCharLiteral = False
        self.Profile.FileLinesList = [list(s) for s in self.Profile.FileLinesListFromFile]
        while not self.__EndOfFile():
            if not InComment and self.__CurrentChar() == T_CHAR_DOUBLE_QUOTE:
                InString = not InString
            if not InComment and self.__CurrentChar() == T_CHAR_SINGLE_QUOTE:
                InCharLiteral = not InCharLiteral
            if self.__CurrentChar() == T_CHAR_LF:
                if HashComment and PPDirectiveObj is not None:
                    if PPDirectiveObj.Content.rstrip(T_CHAR_CR).endswith(T_CHAR_BACKSLASH):
                        PPDirectiveObj.Content += T_CHAR_LF
                        PPExtend = True
                    else:
                        PPExtend = False
                EndLinePos = (self.CurrentLineNumber, self.CurrentOffsetWithinLine)
                if InComment and DoubleSlashComment:
                    InComment = False
                    DoubleSlashComment = False
                if InComment and HashComment and (not PPExtend):
                    InComment = False
                    HashComment = False
                    PPDirectiveObj.Content += T_CHAR_LF
                    PPDirectiveObj.EndPos = EndLinePos
                    FileProfile.PPDirectiveList.append(PPDirectiveObj)
                    PPDirectiveObj = None
                if InString or InCharLiteral:
                    CurrentLine = ''.join(self.__CurrentLine())
                    if CurrentLine.rstrip(T_CHAR_LF).rstrip(T_CHAR_CR).endswith(T_CHAR_BACKSLASH):
                        SlashIndex = CurrentLine.rindex(T_CHAR_BACKSLASH)
                        self.__SetCharValue(self.CurrentLineNumber, SlashIndex, T_CHAR_SPACE)
                self.CurrentLineNumber += 1
                self.CurrentOffsetWithinLine = 0
            elif InComment and (not DoubleSlashComment) and (not HashComment) and (self.__CurrentChar() == T_CHAR_STAR) and (self.__NextChar() == T_CHAR_SLASH):
                self.__SetCurrentCharValue(T_CHAR_SPACE)
                self.__GetOneChar()
                self.__SetCurrentCharValue(T_CHAR_SPACE)
                self.__GetOneChar()
                InComment = False
            elif InComment:
                if HashComment:
                    if self.__CurrentChar() == T_CHAR_SLASH and self.__NextChar() == T_CHAR_SLASH:
                        InComment = False
                        HashComment = False
                        PPDirectiveObj.EndPos = (self.CurrentLineNumber, self.CurrentOffsetWithinLine - 1)
                        FileProfile.PPDirectiveList.append(PPDirectiveObj)
                        PPDirectiveObj = None
                        continue
                    else:
                        PPDirectiveObj.Content += self.__CurrentChar()
                self.__SetCurrentCharValue(T_CHAR_SPACE)
                self.__GetOneChar()
            elif self.__CurrentChar() == T_CHAR_SLASH and self.__NextChar() == T_CHAR_SLASH:
                InComment = True
                DoubleSlashComment = True
            elif self.__CurrentChar() == T_CHAR_HASH and (not InString) and (not InCharLiteral):
                InComment = True
                HashComment = True
                PPDirectiveObj = PP_Directive('', (self.CurrentLineNumber, self.CurrentOffsetWithinLine), None)
            elif self.__CurrentChar() == T_CHAR_SLASH and self.__NextChar() == T_CHAR_STAR:
                self.__SetCurrentCharValue(T_CHAR_SPACE)
                self.__GetOneChar()
                self.__SetCurrentCharValue(T_CHAR_SPACE)
                self.__GetOneChar()
                InComment = True
            else:
                self.__GetOneChar()
        EndLinePos = (self.CurrentLineNumber, self.CurrentOffsetWithinLine)
        if InComment and HashComment and (not PPExtend):
            PPDirectiveObj.EndPos = EndLinePos
            FileProfile.PPDirectiveList.append(PPDirectiveObj)
        self.Rewind()

    def ParseFile(self):
        if False:
            return 10
        self.PreprocessFileWithClear()
        self.Profile.FileLinesList = [''.join(list) for list in self.Profile.FileLinesList]
        FileStringContents = ''
        for fileLine in self.Profile.FileLinesList:
            FileStringContents += fileLine
        cStream = antlr.InputStream(FileStringContents)
        lexer = CLexer(cStream)
        tStream = antlr.CommonTokenStream(lexer)
        parser = CParser(tStream)
        parser.translation_unit()

    def CleanFileProfileBuffer(self):
        if False:
            print('Hello World!')
        FileProfile.PPDirectiveList = []
        FileProfile.AssignmentExpressionList = []
        FileProfile.FunctionDefinitionList = []
        FileProfile.VariableDeclarationList = []
        FileProfile.EnumerationDefinitionList = []
        FileProfile.StructUnionDefinitionList = []
        FileProfile.TypedefDefinitionList = []
        FileProfile.FunctionCallingList = []

    def PrintFragments(self):
        if False:
            return 10
        print('################# ' + self.FileName + '#####################')
        print('/****************************************/')
        print('/************** ASSIGNMENTS *************/')
        print('/****************************************/')
        for assign in FileProfile.AssignmentExpressionList:
            print(str(assign.StartPos) + assign.Name + assign.Operator + assign.Value)
        print('/****************************************/')
        print('/********* PREPROCESS DIRECTIVES ********/')
        print('/****************************************/')
        for pp in FileProfile.PPDirectiveList:
            print(str(pp.StartPos) + pp.Content)
        print('/****************************************/')
        print('/********* VARIABLE DECLARATIONS ********/')
        print('/****************************************/')
        for var in FileProfile.VariableDeclarationList:
            print(str(var.StartPos) + var.Modifier + ' ' + var.Declarator)
        print('/****************************************/')
        print('/********* FUNCTION DEFINITIONS *********/')
        print('/****************************************/')
        for func in FileProfile.FunctionDefinitionList:
            print(str(func.StartPos) + func.Modifier + ' ' + func.Declarator + ' ' + str(func.NamePos))
        print('/****************************************/')
        print('/************ ENUMERATIONS **************/')
        print('/****************************************/')
        for enum in FileProfile.EnumerationDefinitionList:
            print(str(enum.StartPos) + enum.Content)
        print('/****************************************/')
        print('/*********** STRUCTS/UNIONS *************/')
        print('/****************************************/')
        for su in FileProfile.StructUnionDefinitionList:
            print(str(su.StartPos) + su.Content)
        print('/****************************************/')
        print('/************** TYPEDEFS ****************/')
        print('/****************************************/')
        for typedef in FileProfile.TypedefDefinitionList:
            print(str(typedef.StartPos) + typedef.ToType)
if __name__ == '__main__':
    print('For Test.')