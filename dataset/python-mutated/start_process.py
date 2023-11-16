from os import getenv
from e2b import Sandbox
E2B_API_KEY = getenv('E2B_API_KEY')

def print_out(output):
    if False:
        while True:
            i = 10
    print(output.line)

def main():
    if False:
        while True:
            i = 10
    sandbox = Sandbox(id='base', api_key=E2B_API_KEY)
    proc = sandbox.process.start(cmd="ps aux | tr -s ' ' | cut -d ' ' -f 11", on_stdout=print_out, on_stderr=print_out)
    proc.wait()
    output = proc.output
    sandbox.close()
main()