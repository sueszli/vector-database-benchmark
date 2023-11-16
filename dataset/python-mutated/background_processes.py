import time
from os import getenv
from e2b import Sandbox
E2B_API_KEY = getenv('E2B_API_KEY')

def print_stdout(output):
    if False:
        for i in range(10):
            print('nop')
    print(output.line)

def main():
    if False:
        i = 10
        return i + 15
    sandbox = Sandbox(id='base', api_key=E2B_API_KEY)
    background_server = sandbox.process.start('python3 -m http.server 8000', on_stdout=print_stdout)
    time.sleep(1)
    server_request = sandbox.process.start('curl localhost:8000')
    request_output = server_request.wait()
    background_server.kill()
    server_output = background_server.output
    sandbox.close()
main()