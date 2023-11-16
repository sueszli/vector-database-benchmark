"""
XmlRoutines
"""
import xml.dom.minidom
import re
import codecs
from Logger.ToolError import PARSER_ERROR
import Logger.Log as Logger

def CreateXmlElement(Name, String, NodeList, AttributeList):
    if False:
        for i in range(10):
            print('nop')
    Doc = xml.dom.minidom.Document()
    Element = Doc.createElement(Name)
    if String != '' and String is not None:
        Element.appendChild(Doc.createTextNode(String))
    for Item in NodeList:
        if isinstance(Item, type([])):
            Key = Item[0]
            Value = Item[1]
            if Key != '' and Key is not None and (Value != '') and (Value is not None):
                Node = Doc.createElement(Key)
                Node.appendChild(Doc.createTextNode(Value))
                Element.appendChild(Node)
        else:
            Element.appendChild(Item)
    for Item in AttributeList:
        Key = Item[0]
        Value = Item[1]
        if Key != '' and Key is not None and (Value != '') and (Value is not None):
            Element.setAttribute(Key, Value)
    return Element

def XmlList(Dom, String):
    if False:
        i = 10
        return i + 15
    if String is None or String == '' or Dom is None or (Dom == ''):
        return []
    if Dom.nodeType == Dom.DOCUMENT_NODE:
        Dom = Dom.documentElement
    if String[0] == '/':
        String = String[1:]
    TagList = String.split('/')
    Nodes = [Dom]
    Index = 0
    End = len(TagList) - 1
    while Index <= End:
        ChildNodes = []
        for Node in Nodes:
            if Node.nodeType == Node.ELEMENT_NODE and Node.tagName == TagList[Index]:
                if Index < End:
                    ChildNodes.extend(Node.childNodes)
                else:
                    ChildNodes.append(Node)
        Nodes = ChildNodes
        ChildNodes = []
        Index += 1
    return Nodes

def XmlNode(Dom, String):
    if False:
        for i in range(10):
            print('nop')
    if String is None or String == '' or Dom is None or (Dom == ''):
        return None
    if Dom.nodeType == Dom.DOCUMENT_NODE:
        Dom = Dom.documentElement
    if String[0] == '/':
        String = String[1:]
    TagList = String.split('/')
    Index = 0
    End = len(TagList) - 1
    ChildNodes = [Dom]
    while Index <= End:
        for Node in ChildNodes:
            if Node.nodeType == Node.ELEMENT_NODE and Node.tagName == TagList[Index]:
                if Index < End:
                    ChildNodes = Node.childNodes
                else:
                    return Node
                break
        Index += 1
    return None

def XmlElement(Dom, String):
    if False:
        return 10
    try:
        return XmlNode(Dom, String).firstChild.data.strip()
    except BaseException:
        return ''

def XmlElement2(Dom, String):
    if False:
        print('Hello World!')
    try:
        HelpStr = XmlNode(Dom, String).firstChild.data
        gRemovePrettyRe = re.compile('(?:(\\n *)  )(.*)\\1', re.DOTALL)
        HelpStr = re.sub(gRemovePrettyRe, '\\2', HelpStr)
        return HelpStr
    except BaseException:
        return ''

def XmlElementData(Dom):
    if False:
        i = 10
        return i + 15
    try:
        return Dom.firstChild.data.strip()
    except BaseException:
        return ''

def XmlElementList(Dom, String):
    if False:
        for i in range(10):
            print('nop')
    return list(map(XmlElementData, XmlList(Dom, String)))

def XmlAttribute(Dom, Attribute):
    if False:
        print('Hello World!')
    try:
        return Dom.getAttribute(Attribute)
    except BaseException:
        return ''

def XmlNodeName(Dom):
    if False:
        while True:
            i = 10
    try:
        return Dom.nodeName.strip()
    except BaseException:
        return ''

def XmlParseFile(FileName):
    if False:
        i = 10
        return i + 15
    try:
        XmlFile = codecs.open(FileName, 'rb')
        Dom = xml.dom.minidom.parse(XmlFile)
        XmlFile.close()
        return Dom
    except BaseException as XExcept:
        XmlFile.close()
        Logger.Error('\nUPT', PARSER_ERROR, XExcept, File=FileName, RaiseError=True)