from __future__ import annotations
from ansible.plugins.shell.powershell import _parse_clixml, ShellModule

def test_parse_clixml_empty():
    if False:
        while True:
            i = 10
    empty = b'#< CLIXML\r\n<Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04"></Objs>'
    expected = b''
    actual = _parse_clixml(empty)
    assert actual == expected

def test_parse_clixml_with_progress():
    if False:
        print('Hello World!')
    progress = b'#< CLIXML\r\n<Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04"><Obj S="progress" RefId="0"><TN RefId="0"><T>System.Management.Automation.PSCustomObject</T><T>System.Object</T></TN><MS><I64 N="SourceId">1</I64><PR N="Record"><AV>Preparing modules for first use.</AV><AI>0</AI><Nil /><PI>-1</PI><PC>-1</PC><T>Completed</T><SR>-1</SR><SD> </SD></PR></MS></Obj></Objs>'
    expected = b''
    actual = _parse_clixml(progress)
    assert actual == expected

def test_parse_clixml_single_stream():
    if False:
        while True:
            i = 10
    single_stream = b'#< CLIXML\r\n<Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04"><S S="Error">fake : The term \'fake\' is not recognized as the name of a cmdlet. Check _x000D__x000A_</S><S S="Error">the spelling of the name, or if a path was included._x000D__x000A_</S><S S="Error">At line:1 char:1_x000D__x000A_</S><S S="Error">+ fake cmdlet_x000D__x000A_</S><S S="Error">+ ~~~~_x000D__x000A_</S><S S="Error">    + CategoryInfo          : ObjectNotFound: (fake:String) [], CommandNotFoundException_x000D__x000A_</S><S S="Error">    + FullyQualifiedErrorId : CommandNotFoundException_x000D__x000A_</S><S S="Error"> _x000D__x000A_</S></Objs>'
    expected = b"fake : The term 'fake' is not recognized as the name of a cmdlet. Check \r\nthe spelling of the name, or if a path was included.\r\nAt line:1 char:1\r\n+ fake cmdlet\r\n+ ~~~~\r\n    + CategoryInfo          : ObjectNotFound: (fake:String) [], CommandNotFoundException\r\n    + FullyQualifiedErrorId : CommandNotFoundException\r\n "
    actual = _parse_clixml(single_stream)
    assert actual == expected

def test_parse_clixml_multiple_streams():
    if False:
        while True:
            i = 10
    multiple_stream = b'#< CLIXML\r\n<Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04"><S S="Error">fake : The term \'fake\' is not recognized as the name of a cmdlet. Check _x000D__x000A_</S><S S="Error">the spelling of the name, or if a path was included._x000D__x000A_</S><S S="Error">At line:1 char:1_x000D__x000A_</S><S S="Error">+ fake cmdlet_x000D__x000A_</S><S S="Error">+ ~~~~_x000D__x000A_</S><S S="Error">    + CategoryInfo          : ObjectNotFound: (fake:String) [], CommandNotFoundException_x000D__x000A_</S><S S="Error">    + FullyQualifiedErrorId : CommandNotFoundException_x000D__x000A_</S><S S="Error"> _x000D__x000A_</S><S S="Info">hi info</S></Objs>'
    expected = b'hi info'
    actual = _parse_clixml(multiple_stream, stream='Info')
    assert actual == expected

def test_parse_clixml_multiple_elements():
    if False:
        i = 10
        return i + 15
    multiple_elements = b'#< CLIXML\r\n#< CLIXML\r\n<Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04"><Obj S="progress" RefId="0"><TN RefId="0"><T>System.Management.Automation.PSCustomObject</T><T>System.Object</T></TN><MS><I64 N="SourceId">1</I64><PR N="Record"><AV>Preparing modules for first use.</AV><AI>0</AI><Nil /><PI>-1</PI><PC>-1</PC><T>Completed</T><SR>-1</SR><SD> </SD></PR></MS></Obj><S S="Error">Error 1</S></Objs><Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04"><Obj S="progress" RefId="0"><TN RefId="0"><T>System.Management.Automation.PSCustomObject</T><T>System.Object</T></TN><MS><I64 N="SourceId">1</I64><PR N="Record"><AV>Preparing modules for first use.</AV><AI>0</AI><Nil /><PI>-1</PI><PC>-1</PC><T>Completed</T><SR>-1</SR><SD> </SD></PR></MS></Obj><Obj S="progress" RefId="1"><TNRef RefId="0" /><MS><I64 N="SourceId">2</I64><PR N="Record"><AV>Preparing modules for first use.</AV><AI>0</AI><Nil /><PI>-1</PI><PC>-1</PC><T>Completed</T><SR>-1</SR><SD> </SD></PR></MS></Obj><S S="Error">Error 2</S></Objs>'
    expected = b'Error 1\r\nError 2'
    actual = _parse_clixml(multiple_elements)
    assert actual == expected

def test_join_path_unc():
    if False:
        while True:
            i = 10
    pwsh = ShellModule()
    unc_path_parts = ['\\\\host\\share\\dir1\\\\dir2\\', '\\dir3/dir4', 'dir5', 'dir6\\']
    expected = '\\\\host\\share\\dir1\\dir2\\dir3\\dir4\\dir5\\dir6'
    actual = pwsh.join_path(*unc_path_parts)
    assert actual == expected