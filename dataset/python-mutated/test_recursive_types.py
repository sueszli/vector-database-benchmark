"""Test if interop with recursive type inheritance works."""

def test_recursive_type_creation():
    if False:
        print('Hello World!')
    "Test that a recursive types don't crash with a\n    StackOverflowException"
    from Python.Test import RecursiveInheritance
    test_instance = RecursiveInheritance.SubClass()
    test_instance.SomeMethod()