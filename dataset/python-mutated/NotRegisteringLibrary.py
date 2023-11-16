from robot.libraries.BuiltIn import BuiltIn

def my_run_keyword(name, *args):
    if False:
        i = 10
        return i + 15
    return BuiltIn().run_keyword(name, *args)