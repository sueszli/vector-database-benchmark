class PrintPythonExecutableArgs:
    pass

class PrintPythonExecutable:

    def run_action(self, _args):
        if False:
            for i in range(10):
                print('nop')
        import sys
        print(sys.executable)