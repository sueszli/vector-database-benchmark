import logging
import re
from typing import Any, Optional
from flask import current_app
from flask_babel import gettext as __
from superset.commands.base import BaseCommand
from superset.daos.database import DatabaseDAO
from superset.databases.commands.exceptions import DatabaseNotFoundError, NoValidatorConfigFoundError, NoValidatorFoundError, ValidatorSQL400Error, ValidatorSQLError, ValidatorSQLUnexpectedError
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.models.core import Database
from superset.sql_validators import get_validator_by_name
from superset.sql_validators.base import BaseSQLValidator
from superset.utils import core as utils
logger = logging.getLogger(__name__)

class ValidateSQLCommand(BaseCommand):

    def __init__(self, model_id: int, data: dict[str, Any]):
        if False:
            print('Hello World!')
        self._properties = data.copy()
        self._model_id = model_id
        self._model: Optional[Database] = None
        self._validator: Optional[type[BaseSQLValidator]] = None

    def run(self) -> list[dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        '\n        Validates a SQL statement\n\n        :return: A List of SQLValidationAnnotation\n        :raises: DatabaseNotFoundError, NoValidatorConfigFoundError\n          NoValidatorFoundError, ValidatorSQLUnexpectedError, ValidatorSQLError\n          ValidatorSQL400Error\n        '
        self.validate()
        if not self._validator or not self._model:
            raise ValidatorSQLUnexpectedError()
        sql = self._properties['sql']
        schema = self._properties.get('schema')
        try:
            timeout = current_app.config['SQLLAB_VALIDATION_TIMEOUT']
            timeout_msg = f'The query exceeded the {timeout} seconds timeout.'
            with utils.timeout(seconds=timeout, error_message=timeout_msg):
                errors = self._validator.validate(sql, schema, self._model)
            return [err.to_dict() for err in errors]
        except Exception as ex:
            logger.exception(ex)
            superset_error = SupersetError(message=__('%(validator)s was unable to check your query.\nPlease recheck your query.\nException: %(ex)s', validator=self._validator.name, ex=ex), error_type=SupersetErrorType.GENERIC_DB_ENGINE_ERROR, level=ErrorLevel.ERROR)
            if re.search('([\\W]|^)4\\d{2}([\\W]|$)', str(ex)):
                raise ValidatorSQL400Error(superset_error) from ex
            raise ValidatorSQLError(superset_error) from ex

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        self._model = DatabaseDAO.find_by_id(self._model_id)
        if not self._model:
            raise DatabaseNotFoundError()
        spec = self._model.db_engine_spec
        validators_by_engine = current_app.config['SQL_VALIDATORS_BY_ENGINE']
        if not validators_by_engine or spec.engine not in validators_by_engine:
            raise NoValidatorConfigFoundError(SupersetError(message=__(f'no SQL validator is configured for {spec.engine}'), error_type=SupersetErrorType.GENERIC_DB_ENGINE_ERROR, level=ErrorLevel.ERROR))
        validator_name = validators_by_engine[spec.engine]
        self._validator = get_validator_by_name(validator_name)
        if not self._validator:
            raise NoValidatorFoundError(SupersetError(message=__(f'No validator named {validator_name} found (configured for the {spec.engine} engine)'), error_type=SupersetErrorType.GENERIC_DB_ENGINE_ERROR, level=ErrorLevel.ERROR))