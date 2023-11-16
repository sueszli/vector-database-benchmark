"""
Topic: 临时文件和目录
Desc : 
"""
from tempfile import TemporaryFile
from tempfile import TemporaryDirectory
from tempfile import NamedTemporaryFile
import tempfile

def temp_file():
    if False:
        while True:
            i = 10
    with TemporaryFile('w+t') as f:
        f.write('Hello World\n')
        f.write('Testing\n')
        f.seek(0)
        data = f.read()
        print(data)
    with NamedTemporaryFile('w+t') as f:
        print('filename is:', f.name)
    with TemporaryDirectory() as dirname:
        print('dirname is:', dirname)
    print(tempfile.mkstemp())
    print(tempfile.mkdtemp())
    print(tempfile.gettempdir())
if __name__ == '__main__':
    temp_file()