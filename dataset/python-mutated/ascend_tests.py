from superset.db_engine_specs.ascend import AscendEngineSpec
from tests.integration_tests.db_engine_specs.base_tests import TestDbEngineSpec

class TestAscendDbEngineSpec(TestDbEngineSpec):

    def test_convert_dttm(self):
        if False:
            print('Hello World!')
        dttm = self.get_dttm()
        self.assertEqual(AscendEngineSpec.convert_dttm('DATE', dttm), "CAST('2019-01-02' AS DATE)")
        self.assertEqual(AscendEngineSpec.convert_dttm('TIMESTAMP', dttm), "CAST('2019-01-02T03:04:05.678900' AS TIMESTAMP)")