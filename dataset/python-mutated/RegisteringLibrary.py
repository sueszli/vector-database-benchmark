from robot.libraries.BuiltIn import BuiltIn, register_run_keyword

def run_keyword_function(name, *args):
    if False:
        i = 10
        return i + 15
    return BuiltIn().run_keyword(name, *args)
register_run_keyword(__name__, 'run_keyword_function', 1)

def run_keyword_without_keyword(*args):
    if False:
        return 10
    return BuiltIn().run_keyword('\\\\Log Many', *args)
register_run_keyword(__name__, 'run_keyword_without_keyword', 0)