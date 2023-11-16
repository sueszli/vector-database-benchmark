import os
import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pyspark.testing.utils import SPARK_HOME

class SparkSubmitTests(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.programDir = tempfile.mkdtemp()
        tmp_dir = tempfile.gettempdir()
        self.sparkSubmit = [os.path.join(SPARK_HOME, 'bin', 'spark-submit'), '--conf', 'spark.driver.extraJavaOptions=-Djava.io.tmpdir={0}'.format(tmp_dir), '--conf', 'spark.executor.extraJavaOptions=-Djava.io.tmpdir={0}'.format(tmp_dir)]

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.programDir)

    def createTempFile(self, name, content, dir=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a temp file with the given name and content and return its path.\n        Strips leading spaces from content up to the first '|' in each line.\n        "
        pattern = re.compile('^ *\\|', re.MULTILINE)
        content = re.sub(pattern, '', content.strip())
        if dir is None:
            path = os.path.join(self.programDir, name)
        else:
            os.makedirs(os.path.join(self.programDir, dir))
            path = os.path.join(self.programDir, dir, name)
        with open(path, 'w') as f:
            f.write(content)
        return path

    def createFileInZip(self, name, content, ext='.zip', dir=None, zip_name=None):
        if False:
            print('Hello World!')
        "\n        Create a zip archive containing a file with the given content and return its path.\n        Strips leading spaces from content up to the first '|' in each line.\n        "
        pattern = re.compile('^ *\\|', re.MULTILINE)
        content = re.sub(pattern, '', content.strip())
        if dir is None:
            path = os.path.join(self.programDir, name + ext)
        else:
            path = os.path.join(self.programDir, dir, zip_name + ext)
        zip = zipfile.ZipFile(path, 'w')
        zip.writestr(name, content)
        zip.close()
        return path

    def create_spark_package(self, artifact_name):
        if False:
            while True:
                i = 10
        (group_id, artifact_id, version) = artifact_name.split(':')
        self.createTempFile('%s-%s.pom' % (artifact_id, version), ('\n            |<?xml version="1.0" encoding="UTF-8"?>\n            |<project xmlns="http://maven.apache.org/POM/4.0.0"\n            |       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n            |       xsi:schemaLocation="http://maven.apache.org/POM/4.0.0\n            |       http://maven.apache.org/xsd/maven-4.0.0.xsd">\n            |   <modelVersion>4.0.0</modelVersion>\n            |   <groupId>%s</groupId>\n            |   <artifactId>%s</artifactId>\n            |   <version>%s</version>\n            |</project>\n            ' % (group_id, artifact_id, version)).lstrip(), os.path.join(group_id, artifact_id, version))
        self.createFileInZip('%s.py' % artifact_id, '\n            |def myfunc(x):\n            |    return x + 1\n            ', '.jar', os.path.join(group_id, artifact_id, version), '%s-%s' % (artifact_id, version))

    def test_single_script(self):
        if False:
            print('Hello World!')
        'Submit and test a single script file'
        script = self.createTempFile('test.py', '\n            |from pyspark import SparkContext\n            |\n            |sc = SparkContext()\n            |print(sc.parallelize([1, 2, 3]).map(lambda x: x * 2).collect())\n            ')
        proc = subprocess.Popen(self.sparkSubmit + [script], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        self.assertEqual(0, proc.returncode)
        self.assertIn('[2, 4, 6]', out.decode('utf-8'))

    def test_script_with_local_functions(self):
        if False:
            print('Hello World!')
        'Submit and test a single script file calling a global function'
        script = self.createTempFile('test.py', '\n            |from pyspark import SparkContext\n            |\n            |def foo(x):\n            |    return x * 3\n            |\n            |sc = SparkContext()\n            |print(sc.parallelize([1, 2, 3]).map(foo).collect())\n            ')
        proc = subprocess.Popen(self.sparkSubmit + [script], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        self.assertEqual(0, proc.returncode)
        self.assertIn('[3, 6, 9]', out.decode('utf-8'))

    def test_module_dependency(self):
        if False:
            i = 10
            return i + 15
        'Submit and test a script with a dependency on another module'
        script = self.createTempFile('test.py', '\n            |from pyspark import SparkContext\n            |from mylib import myfunc\n            |\n            |sc = SparkContext()\n            |print(sc.parallelize([1, 2, 3]).map(myfunc).collect())\n            ')
        zip = self.createFileInZip('mylib.py', '\n            |def myfunc(x):\n            |    return x + 1\n            ')
        proc = subprocess.Popen(self.sparkSubmit + ['--py-files', zip, script], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        self.assertEqual(0, proc.returncode)
        self.assertIn('[2, 3, 4]', out.decode('utf-8'))

    def test_module_dependency_on_cluster(self):
        if False:
            print('Hello World!')
        'Submit and test a script with a dependency on another module on a cluster'
        script = self.createTempFile('test.py', '\n            |from pyspark import SparkContext\n            |from mylib import myfunc\n            |\n            |sc = SparkContext()\n            |print(sc.parallelize([1, 2, 3]).map(myfunc).collect())\n            ')
        zip = self.createFileInZip('mylib.py', '\n            |def myfunc(x):\n            |    return x + 1\n            ')
        proc = subprocess.Popen(self.sparkSubmit + ['--py-files', zip, '--master', 'local-cluster[1,1,1024]', script], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        self.assertEqual(0, proc.returncode)
        self.assertIn('[2, 3, 4]', out.decode('utf-8'))

    def test_package_dependency(self):
        if False:
            print('Hello World!')
        'Submit and test a script with a dependency on a Spark Package'
        script = self.createTempFile('test.py', '\n            |from pyspark import SparkContext\n            |from mylib import myfunc\n            |\n            |sc = SparkContext()\n            |print(sc.parallelize([1, 2, 3]).map(myfunc).collect())\n            ')
        self.create_spark_package('a:mylib:0.1')
        proc = subprocess.Popen(self.sparkSubmit + ['--packages', 'a:mylib:0.1', '--repositories', 'file:' + self.programDir, script], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        self.assertEqual(0, proc.returncode)
        self.assertIn('[2, 3, 4]', out.decode('utf-8'))

    def test_package_dependency_on_cluster(self):
        if False:
            return 10
        'Submit and test a script with a dependency on a Spark Package on a cluster'
        script = self.createTempFile('test.py', '\n            |from pyspark import SparkContext\n            |from mylib import myfunc\n            |\n            |sc = SparkContext()\n            |print(sc.parallelize([1, 2, 3]).map(myfunc).collect())\n            ')
        self.create_spark_package('a:mylib:0.1')
        proc = subprocess.Popen(self.sparkSubmit + ['--packages', 'a:mylib:0.1', '--repositories', 'file:' + self.programDir, '--master', 'local-cluster[1,1,1024]', script], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        self.assertEqual(0, proc.returncode)
        self.assertIn('[2, 3, 4]', out.decode('utf-8'))

    def test_single_script_on_cluster(self):
        if False:
            print('Hello World!')
        'Submit and test a single script on a cluster'
        script = self.createTempFile('test.py', '\n            |from pyspark import SparkContext\n            |\n            |def foo(x):\n            |    return x * 2\n            |\n            |sc = SparkContext()\n            |print(sc.parallelize([1, 2, 3]).map(foo).collect())\n            ')
        proc = subprocess.Popen(self.sparkSubmit + ['--master', 'local-cluster[1,1,1024]', script], stdout=subprocess.PIPE)
        (out, err) = proc.communicate()
        self.assertEqual(0, proc.returncode)
        self.assertIn('[2, 4, 6]', out.decode('utf-8'))

    def test_user_configuration(self):
        if False:
            while True:
                i = 10
        'Make sure user configuration is respected (SPARK-19307)'
        script = self.createTempFile('test.py', '\n            |from pyspark import SparkConf, SparkContext\n            |\n            |conf = SparkConf().set("spark.test_config", "1")\n            |sc = SparkContext(conf = conf)\n            |try:\n            |    if sc._conf.get("spark.test_config") != "1":\n            |        raise RuntimeError("Cannot find spark.test_config in SparkContext\'s conf.")\n            |finally:\n            |    sc.stop()\n            ')
        proc = subprocess.Popen(self.sparkSubmit + ['--master', 'local', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out, err) = proc.communicate()
        self.assertEqual(0, proc.returncode, msg='Process failed with error:\n {0}'.format(out))
if __name__ == '__main__':
    from pyspark.tests.test_appsubmit import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)