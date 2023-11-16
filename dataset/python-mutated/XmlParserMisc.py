"""
XmlParserMisc
"""
from Object.POM.CommonObject import TextObject
from Logger.StringTable import ERR_XML_PARSER_REQUIRED_ITEM_MISSING
from Logger.ToolError import PARSER_ERROR
import Logger.Log as Logger

def ConvertVariableName(VariableName):
    if False:
        return 10
    VariableName = VariableName.strip()
    if VariableName.startswith('L"') and VariableName.endswith('"'):
        return VariableName
    ValueList = VariableName.split(' ')
    if len(ValueList) % 2 == 1:
        return None
    TransferedStr = ''
    Index = 0
    while Index < len(ValueList):
        FirstByte = int(ValueList[Index], 16)
        SecondByte = int(ValueList[Index + 1], 16)
        if SecondByte != 0:
            return None
        if FirstByte not in range(32, 127):
            return None
        TransferedStr += '%c' % FirstByte
        Index = Index + 2
    return 'L"' + TransferedStr + '"'

def IsRequiredItemListNull(ItemDict, XmlTreeLevel):
    if False:
        while True:
            i = 10
    for Key in ItemDict:
        if not ItemDict[Key]:
            Msg = '->'.join((Node for Node in XmlTreeLevel))
            ErrorMsg = ERR_XML_PARSER_REQUIRED_ITEM_MISSING % (Key, Msg)
            Logger.Error('\nUPT', PARSER_ERROR, ErrorMsg, RaiseError=True)

def GetHelpTextList(HelpText):
    if False:
        return 10
    HelpTextList = []
    for HelT in HelpText:
        HelpTextObj = TextObject()
        HelpTextObj.SetLang(HelT.Lang)
        HelpTextObj.SetString(HelT.HelpText)
        HelpTextList.append(HelpTextObj)
    return HelpTextList

def GetPromptList(Prompt):
    if False:
        return 10
    PromptList = []
    for SubPrompt in Prompt:
        PromptObj = TextObject()
        PromptObj.SetLang(SubPrompt.Lang)
        PromptObj.SetString(SubPrompt.Prompt)
        PromptList.append(PromptObj)
    return PromptList