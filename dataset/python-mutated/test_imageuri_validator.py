import base64
import os
import pytest
from _plotly_utils.basevalidators import ImageUriValidator
from PIL import Image

@pytest.fixture()
def validator():
    if False:
        print('Hello World!')
    return ImageUriValidator('prop', 'parent')

@pytest.mark.parametrize('val', ['http://somewhere.com/images/image12.png', 'data:image/png;base64,iVBORw0KGgoAAAANSU'])
def test_validator_acceptance(val, validator):
    if False:
        return 10
    assert validator.validate_coerce(val) == val

def test_validator_coercion_PIL(validator):
    if False:
        return 10
    tests_dir = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(tests_dir, 'resources', '1x1-black.png')
    with open(img_path, 'rb') as f:
        hex_bytes = base64.b64encode(f.read()).decode('ascii')
        expected_uri = 'data:image/png;base64,' + hex_bytes
    img = Image.open(img_path)
    coerce_val = validator.validate_coerce(img)
    assert coerce_val == expected_uri

@pytest.mark.parametrize('val', [23, set(), []])
def test_rejection_by_type(val, validator):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as validation_failure:
        validator.validate_coerce(val)
    assert 'Invalid value' in str(validation_failure.value)