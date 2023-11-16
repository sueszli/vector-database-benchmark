from e2b import Sandbox

def print_out(output):
    if False:
        for i in range(10):
            print('nop')
    print(output.line)
sandbox = Sandbox(id='base')
code = "\n  const fs = require('fs');\n  const dirContent = fs.readdirSync('/');\n  dirContent.forEach((item) => {\n    console.log('Root dir item inside playground:', item);\n  });\n"
sandbox.filesystem.write('/code/index.js', code)
proc = sandbox.process.start(cmd='node /code/index.js', on_stdout=print_out, on_stderr=print_out)
proc.wait()
output = proc.output
sandbox.close()