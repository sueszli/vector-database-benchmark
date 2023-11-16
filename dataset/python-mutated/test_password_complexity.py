from flask_appbuilder.exceptions import PasswordComplexityValidationError
from flask_appbuilder.validators import default_password_complexity
from parameterized import parameterized
from ..base import FABTestCase

class PasswordComplexityTestCase(FABTestCase):

    @parameterized.expand(['password', 'password1234', 'password123', 'PAssword123', 'PAssw12!', 'Password123!', 'PASSWOrd123!', 'PAssword3!!', 'PAssw3!!'])
    def test_default_complexity_validator_fail(self, password):
        if False:
            return 10
        with self.assertRaises(PasswordComplexityValidationError):
            default_password_complexity(password)

    @parameterized.expand(['PAssword12!', 'PAssword12!#', 'PAssword12!#>', 'PAssw!ord12', '!PAssword12', '!PAssw>ord12', 'ssw>ord12!PA', 'ssw>PAord12!'])
    def test_default_complexity_validator(self, password):
        if False:
            i = 10
            return i + 15
        default_password_complexity(password)