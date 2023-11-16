def importing_robot_module_directly_fails():
    if False:
        return 10
    try:
        import running
    except ImportError:
        pass
    else:
        raise AssertionError("Importing 'running' directly succeeded!")

def importing_robot_module_through_robot_succeeds():
    if False:
        return 10
    from robot import running

def importing_standard_library_directly_fails():
    if False:
        print('Hello World!')
    try:
        import BuiltIn
    except ImportError:
        pass
    else:
        raise AssertionError("Importing 'BuiltIn' directly succeeded!")

def importing_standard_library_through_robot_libraries_succeeds():
    if False:
        for i in range(10):
            print('nop')
    from robot.libraries import BuiltIn
    BuiltIn.BuiltIn().set_test_variable('${SET BY LIBRARY}', 42)