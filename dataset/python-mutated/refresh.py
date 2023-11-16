import logging
from typing import Optional
from flask_appbuilder.models.sqla import Model
from superset import security_manager
from superset.commands.base import BaseCommand
from superset.connectors.sqla.models import SqlaTable
from superset.daos.dataset import DatasetDAO
from superset.datasets.commands.exceptions import DatasetForbiddenError, DatasetNotFoundError, DatasetRefreshFailedError
from superset.exceptions import SupersetSecurityException
logger = logging.getLogger(__name__)

class RefreshDatasetCommand(BaseCommand):

    def __init__(self, model_id: int):
        if False:
            i = 10
            return i + 15
        self._model_id = model_id
        self._model: Optional[SqlaTable] = None

    def run(self) -> Model:
        if False:
            for i in range(10):
                print('nop')
        self.validate()
        if self._model:
            try:
                self._model.fetch_metadata()
                return self._model
            except Exception as ex:
                logger.exception(ex)
                raise DatasetRefreshFailedError() from ex
        raise DatasetRefreshFailedError()

    def validate(self) -> None:
        if False:
            return 10
        self._model = DatasetDAO.find_by_id(self._model_id)
        if not self._model:
            raise DatasetNotFoundError()
        try:
            security_manager.raise_for_ownership(self._model)
        except SupersetSecurityException as ex:
            raise DatasetForbiddenError() from ex