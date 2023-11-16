from custom_components.hacs.utils.validate import Validate

def test_validate():
    if False:
        i = 10
        return i + 15
    validate = Validate()
    assert validate.success
    validate.errors.append('test')
    assert not validate.success