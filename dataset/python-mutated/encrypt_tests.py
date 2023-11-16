from typing import Any, Optional
from sqlalchemy import String, TypeDecorator
from sqlalchemy_utils import EncryptedType
from sqlalchemy_utils.types.encrypted.encrypted_type import StringEncryptedType
from superset.extensions import encrypted_field_factory
from superset.utils.encrypt import AbstractEncryptedFieldAdapter, SQLAlchemyUtilsAdapter
from tests.integration_tests.base_tests import SupersetTestCase

class CustomEncFieldAdapter(AbstractEncryptedFieldAdapter):

    def create(self, app_config: Optional[dict[str, Any]], *args: list[Any], **kwargs: Optional[dict[str, Any]]) -> TypeDecorator:
        if False:
            print('Hello World!')
        if app_config:
            return StringEncryptedType(*args, app_config['SECRET_KEY'], **kwargs)
        else:
            raise Exception('Missing app_config kwarg')

class EncryptedFieldTest(SupersetTestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.app.config['SQLALCHEMY_ENCRYPTED_FIELD_TYPE_ADAPTER'] = SQLAlchemyUtilsAdapter
        encrypted_field_factory.init_app(self.app)
        super().setUp()

    def test_create_field(self):
        if False:
            for i in range(10):
                print('nop')
        field = encrypted_field_factory.create(String(1024))
        self.assertTrue(isinstance(field, EncryptedType))
        self.assertEqual(self.app.config['SECRET_KEY'], field.key)

    def test_custom_adapter(self):
        if False:
            i = 10
            return i + 15
        self.app.config['SQLALCHEMY_ENCRYPTED_FIELD_TYPE_ADAPTER'] = CustomEncFieldAdapter
        encrypted_field_factory.init_app(self.app)
        field = encrypted_field_factory.create(String(1024))
        self.assertTrue(isinstance(field, StringEncryptedType))
        self.assertFalse(isinstance(field, EncryptedType))
        self.assertEqual(self.app.config['SECRET_KEY'], field.key)