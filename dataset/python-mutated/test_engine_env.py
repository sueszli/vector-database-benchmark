import pytest
import os
from bigdl.dllib.utils.common import *

class TestEngineEnv:

    def setup_method(self, method):
        if False:
            return 10
        ' setup any state tied to the execution of the given method in a\n        class.  setup_method is invoked for every test method of a class.\n        '
        pass

    def teardown_method(self, method):
        if False:
            while True:
                i = 10
        ' teardown any state that was previously setup with a setup_method\n        call.\n        '
        pass

    def test___prepare_bigdl_env(self):
        if False:
            return 10
        from bigdl.dllib.utils.engine import prepare_env
        bigdl_jars_env_1 = os.environ.get('BIGDL_JARS', None)
        spark_class_path_1 = os.environ.get('SPARK_CLASSPATH', None)
        sys_path_1 = sys.path
        prepare_env()
        bigdl_jars_env_2 = os.environ.get('BIGDL_JARS', None)
        spark_class_path_2 = os.environ.get('SPARK_CLASSPATH', None)
        sys_path_2 = sys.path
        assert bigdl_jars_env_1 == bigdl_jars_env_2
        assert spark_class_path_1 == spark_class_path_2
        assert sys_path_1 == sys_path_2
if __name__ == '__main__':
    pytest.main()