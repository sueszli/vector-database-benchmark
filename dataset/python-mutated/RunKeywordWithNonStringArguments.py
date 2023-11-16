from robot.libraries.BuiltIn import BuiltIn

def run_keyword_with_non_string_arguments():
    if False:
        i = 10
        return i + 15
    return BuiltIn().run_keyword('Create List', 1, 2, None)