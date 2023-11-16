from maturin_starter.submodule import SubmoduleClass

def test_submodule_class() -> None:
    if False:
        print('Hello World!')
    submodule_class = SubmoduleClass()
    assert submodule_class.greeting() == 'Hello, world!'