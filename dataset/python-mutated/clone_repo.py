from os import getenv
from e2b import Sandbox
E2B_API_KEY = getenv('E2B_API_KEY')

def print_out(output):
    if False:
        for i in range(10):
            print('nop')
    print(output.line)

def main():
    if False:
        while True:
            i = 10
    sandbox = Sandbox(id='base', api_key=E2B_API_KEY)
    proc = sandbox.process.start(cmd='git clone https://github.com/cruip/open-react-template.git /code/open-react-template', on_stdout=print_out, on_stderr=print_out)
    proc.wait()
    content = sandbox.filesystem.list('/code/open-react-template')
    print(content)
    print('Installing deps...')
    proc = sandbox.process.start(cmd='npm install', on_stdout=print_out, on_stderr=print_out, cwd='/code/open-react-template')
    proc.wait()
    sandbox.close()
main()