"""Class Definition Syntax.

@see: https://docs.python.org/3/tutorial/classes.html#instance-objects
"""

def test_instance_objects():
    if False:
        i = 10
        return i + 15
    'Instance Objects.\n\n    Now what can we do with instance objects? The only operations understood by instance objects\n    are attribute references. There are two kinds of valid attribute names:\n    - data attributes\n    - methods.\n    '

    class DummyClass:
        """Dummy class"""
        pass
    dummy_instance = DummyClass()
    dummy_instance.temporary_attribute = 1
    assert dummy_instance.temporary_attribute == 1
    del dummy_instance.temporary_attribute