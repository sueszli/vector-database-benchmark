import re
from xml.etree.ElementTree import Element, ElementTree
from os import path
import string
from enum import Enum
startingPatt = re.compile('^STARTING: ([\\w\\.\\-]+)$')
skippingPatt = re.compile('^SKIPPING: ([\\w\\.\\-]+)\\s*(\\(([\\w\\.\\-\\ \\,]+)\\))?\\s*$')
exitCodePatt = re.compile('^EXIT CODE: (\\d+)$')
folderPatt = re.compile('^FOLDER: ([\\w\\.\\-]+)$')
timePatt = re.compile('^real\\s+([\\d\\.ms]+)$')
linePatt = re.compile('^' + '-' * 80 + '$')

def getFileBaseName(filePathName):
    if False:
        for i in range(10):
            print('nop')
    return path.splitext(path.basename(filePathName))[0]

def makeTestCaseElement(attrDict):
    if False:
        while True:
            i = 10
    return Element('testcase', attrib=attrDict)

def makeSystemOutElement(outputLines):
    if False:
        while True:
            i = 10
    e = Element('system-out')
    e.text = ''.join(filter(lambda c: c in string.printable, outputLines))
    return e

def makeFailureElement(outputLines):
    if False:
        print('Hello World!')
    e = Element('failure', message='failed')
    e.text = ''.join(filter(lambda c: c in string.printable, outputLines))
    return e

def setFileNameAttr(attrDict, fileName):
    if False:
        i = 10
        return i + 15
    attrDict.update(file=fileName, classname='', line='', name='', time='')

def setClassNameAttr(attrDict, className):
    if False:
        print('Hello World!')
    attrDict['classname'] = className

def setTestNameAttr(attrDict, testName):
    if False:
        return 10
    attrDict['name'] = testName

def setTimeAttr(attrDict, timeVal):
    if False:
        return 10
    (mins, seconds) = timeVal.split('m')
    seconds = float(seconds.strip('s')) + 60 * int(mins)
    attrDict['time'] = str(seconds)

def incrNumAttr(element, attr):
    if False:
        return 10
    newVal = int(element.attrib.get(attr)) + 1
    element.attrib[attr] = str(newVal)

def parseLog(logFile, testSuiteElement):
    if False:
        while True:
            i = 10
    with open(logFile) as lf:
        testSuiteElement.attrib['tests'] = '0'
        testSuiteElement.attrib['errors'] = '0'
        testSuiteElement.attrib['failures'] = '0'
        testSuiteElement.attrib['skipped'] = '0'
        testSuiteElement.attrib['time'] = '0'
        testSuiteElement.attrib['timestamp'] = ''
        attrDict = {}
        setFileNameAttr(attrDict, 'nbtest')
        parserStateEnum = Enum('parserStateEnum', 'newTest startingLine finishLine exitCode')
        parserState = parserStateEnum.newTest
        testOutput = ''
        for line in lf.readlines():
            if parserState == parserStateEnum.newTest:
                m = folderPatt.match(line)
                if m:
                    setClassNameAttr(attrDict, m.group(1))
                    continue
                m = skippingPatt.match(line)
                if m:
                    setTestNameAttr(attrDict, getFileBaseName(m.group(1)))
                    setTimeAttr(attrDict, '0m0s')
                    skippedElement = makeTestCaseElement(attrDict)
                    message = m.group(3) or ''
                    skippedElement.append(Element('skipped', message=message, type=''))
                    testSuiteElement.append(skippedElement)
                    incrNumAttr(testSuiteElement, 'skipped')
                    incrNumAttr(testSuiteElement, 'tests')
                    continue
                m = startingPatt.match(line)
                if m:
                    parserState = parserStateEnum.startingLine
                    testOutput = ''
                    setTestNameAttr(attrDict, m.group(1))
                    setTimeAttr(attrDict, '0m0s')
                    continue
                continue
            elif parserState == parserStateEnum.startingLine:
                if linePatt.match(line):
                    parserState = parserStateEnum.finishLine
                    testOutput = ''
                continue
            elif parserState == parserStateEnum.finishLine:
                if linePatt.match(line):
                    parserState = parserStateEnum.exitCode
                else:
                    testOutput += line
                continue
            elif parserState == parserStateEnum.exitCode:
                m = exitCodePatt.match(line)
                if m:
                    testCaseElement = makeTestCaseElement(attrDict)
                    if m.group(1) != '0':
                        failureElement = makeFailureElement(testOutput)
                        testCaseElement.append(failureElement)
                        incrNumAttr(testSuiteElement, 'failures')
                    else:
                        systemOutElement = makeSystemOutElement(testOutput)
                        testCaseElement.append(systemOutElement)
                    testSuiteElement.append(testCaseElement)
                    parserState = parserStateEnum.newTest
                    testOutput = ''
                    incrNumAttr(testSuiteElement, 'tests')
                    continue
                m = timePatt.match(line)
                if m:
                    setTimeAttr(attrDict, m.group(1))
                    continue
                continue
if __name__ == '__main__':
    import sys
    testSuitesElement = Element('testsuites')
    testSuiteElement = Element('testsuite', name='nbtest', hostname='')
    parseLog(sys.argv[1], testSuiteElement)
    testSuitesElement.append(testSuiteElement)
    ElementTree(testSuitesElement).write(sys.argv[1] + '.xml', xml_declaration=True)