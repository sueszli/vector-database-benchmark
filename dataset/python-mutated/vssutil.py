import time
import traceback
import pythoncom
import win32com.client
import win32com.client.gencache
import win32con
constants = win32com.client.constants
win32com.client.gencache.EnsureModule('{783CD4E0-9D54-11CF-B8EE-00608CC9A71F}', 0, 5, 0)
error = 'vssutil error'

def GetSS():
    if False:
        while True:
            i = 10
    ss = win32com.client.Dispatch('SourceSafe')
    ss.Open(pythoncom.Missing, pythoncom.Missing, pythoncom.Missing)
    return ss

def test(projectName):
    if False:
        while True:
            i = 10
    ss = GetSS()
    project = ss.VSSItem(projectName)
    for item in project.GetVersions(constants.VSSFLAG_RECURSYES):
        print(item.VSSItem.Name, item.VersionNumber, item.Action)

def SubstituteInString(inString, evalEnv):
    if False:
        print('Hello World!')
    substChar = '$'
    fields = inString.split(substChar)
    newFields = []
    for i in range(len(fields)):
        didSubst = 0
        strVal = fields[i]
        if i % 2 != 0:
            try:
                strVal = eval(strVal, evalEnv[0], evalEnv[1])
                newFields.append(strVal)
                didSubst = 1
            except:
                traceback.print_exc()
                print('Could not substitute', strVal)
        if not didSubst:
            newFields.append(strVal)
    return ''.join(map(str, newFields))

def SubstituteInFile(inName, outName, evalEnv):
    if False:
        i = 10
        return i + 15
    inFile = open(inName, 'r')
    try:
        outFile = open(outName, 'w')
        try:
            while 1:
                line = inFile.read()
                if not line:
                    break
                outFile.write(SubstituteInString(line, evalEnv))
        finally:
            outFile.close()
    finally:
        inFile.close()

def VssLog(project, linePrefix='', noLabels=5, maxItems=150):
    if False:
        return 10
    lines = []
    num = 0
    labelNum = 0
    for i in project.GetVersions(constants.VSSFLAG_RECURSYES):
        num = num + 1
        if num > maxItems:
            break
        commentDesc = itemDesc = ''
        if i.Action[:5] == 'Added':
            continue
        if len(i.Label):
            labelNum = labelNum + 1
            itemDesc = i.Action
        else:
            itemDesc = i.VSSItem.Name
            if str(itemDesc[-4:]) == '.dsp':
                continue
        if i.Comment:
            commentDesc = f'\n{linePrefix}\t{i.Comment}'
        lines.append('{}{}\t{}{}'.format(linePrefix, time.asctime(time.localtime(int(i.Date))), itemDesc, commentDesc))
        if labelNum > noLabels:
            break
    return '\n'.join(lines)

def SubstituteVSSInFile(projectName, inName, outName):
    if False:
        for i in range(10):
            print('nop')
    import win32api
    if win32api.GetFullPathName(inName) == win32api.GetFullPathName(outName):
        raise RuntimeError('The input and output filenames can not be the same')
    sourceSafe = GetSS()
    project = sourceSafe.VSSItem(projectName)
    label = None
    for version in project.Versions:
        if version.Label:
            break
    else:
        print('Couldnt find a label in the sourcesafe project!')
        return
    vss_label = version.Label
    vss_date = time.asctime(time.localtime(int(version.Date)))
    now = time.asctime(time.localtime(time.time()))
    SubstituteInFile(inName, outName, (locals(), globals()))

def CountCheckouts(item):
    if False:
        i = 10
        return i + 15
    num = 0
    if item.Type == constants.VSSITEM_PROJECT:
        for sub in item.Items:
            num = num + CountCheckouts(sub)
    elif item.IsCheckedOut:
        num = num + 1
    return num

def GetLastBuildNo(project):
    if False:
        print('Hello World!')
    i = GetSS().VSSItem(project)
    lab = None
    for version in i.Versions:
        lab = str(version.Label)
        if lab:
            return lab
    return None

def MakeNewBuildNo(project, buildDesc=None, auto=0, bRebrand=0):
    if False:
        print('Hello World!')
    if buildDesc is None:
        buildDesc = 'Created by Python'
    ss = GetSS()
    i = ss.VSSItem(project)
    num = CountCheckouts(i)
    if num > 0:
        msg = 'This project has %d items checked out\r\n\r\nDo you still want to continue?' % num
        import win32ui
        if win32ui.MessageBox(msg, project, win32con.MB_YESNO) != win32con.IDYES:
            return
    oldBuild = buildNo = GetLastBuildNo(project)
    if buildNo is None:
        buildNo = '1'
        oldBuild = '<None>'
    else:
        try:
            buildNo = int(buildNo)
            if not bRebrand:
                buildNo = buildNo + 1
            buildNo = str(buildNo)
        except ValueError:
            raise error('The previous label could not be incremented: %s' % oldBuild)
    if not auto:
        from pywin.mfc import dialog
        buildNo = dialog.GetSimpleInput('Enter new build number', buildNo, f'{project} - Prev: {oldBuild}')
        if buildNo is None:
            return
    i.Label(buildNo, f'Build {buildNo}: {buildDesc}')
    if auto:
        print(f'Branded project {project} with label {buildNo}')
    return buildNo
if __name__ == '__main__':
    tp = '\\Python\\Python Win32 Extensions'
    SubstituteVSSInFile(tp, 'd:\\src\\pythonex\\win32\\win32.txt', 'd:\\temp\\win32.txt')