from __future__ import absolute_import
from st2common.models.db.pack import PackDB
from st2common.persistence.pack import Pack
from st2tests import DbTestCase
from tests.unit.base import BaseDBModelCRUDTestCase

class PackDBModelCRUDTestCase(BaseDBModelCRUDTestCase, DbTestCase):
    model_class = PackDB
    persistance_class = Pack
    model_class_kwargs = {'name': 'Yolo CI', 'ref': 'yolo_ci', 'description': 'YOLO CI pack', 'version': '0.1.0', 'author': 'Volkswagen', 'path': '/opt/stackstorm/packs/yolo_ci/'}
    update_attribute_name = 'author'

    def test_path_none(self):
        if False:
            i = 10
            return i + 15
        PackDBModelCRUDTestCase.model_class_kwargs = {'name': 'Yolo CI', 'ref': 'yolo_ci', 'description': 'YOLO CI pack', 'version': '0.1.0', 'author': 'Volkswagen'}
        super(PackDBModelCRUDTestCase, self).test_crud_operations()