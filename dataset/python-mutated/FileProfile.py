from __future__ import absolute_import
import re
import Common.LongFilePathOs as os
from .ParserWarning import Warning
from Common.LongFilePathSupport import OpenLongFilePath as open
PPDirectiveList = []
AssignmentExpressionList = []
PredicateExpressionList = []
FunctionDefinitionList = []
VariableDeclarationList = []
EnumerationDefinitionList = []
StructUnionDefinitionList = []
TypedefDefinitionList = []
FunctionCallingList = []

class FileProfile:

    def __init__(self, FileName):
        if False:
            print('Hello World!')
        self.FileLinesList = []
        self.FileLinesListFromFile = []
        try:
            fsock = open(FileName, 'rb', 0)
            try:
                self.FileLinesListFromFile = fsock.readlines()
            finally:
                fsock.close()
        except IOError:
            raise Warning('Error when opening file %s' % FileName)