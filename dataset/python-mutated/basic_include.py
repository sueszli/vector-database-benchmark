from metaflow_test import MetaflowTest, ExpectationFailed, steps

class BasicIncludeTest(MetaflowTest):
    PRIORITY = 1
    INCLUDE_FILES = {'myfile_txt': {'default': "'./reg.txt'"}, 'myfile_utf8': {'default': "'./utf8.txt'", 'encoding': "'utf8'"}, 'myfile_binary': {'default': "'./utf8.txt'", 'is_text': False}, 'myfile_overriden': {'default': "'./reg.txt'"}, 'absent_file': {'required': False}}
    HEADER = '\nimport codecs\nimport os\nos.environ[\'METAFLOW_RUN_MYFILE_OVERRIDEN\'] = \'./override.txt\'\n\nwith open(\'reg.txt\', mode=\'w\') as f:\n    f.write("Regular Text File")\nwith codecs.open(\'utf8.txt\', mode=\'w\', encoding=\'utf8\') as f:\n    f.write(u"UTF Text File 年")\nwith open(\'override.txt\', mode=\'w\') as f:\n    f.write("Override Text File")\n'

    @steps(0, ['all'])
    def step_all(self):
        if False:
            for i in range(10):
                print('nop')
        assert_equals('Regular Text File', self.myfile_txt)
        assert_equals('UTF Text File 年', self.myfile_utf8)
        assert_equals('UTF Text File 年'.encode(encoding='utf8'), self.myfile_binary)
        assert_equals('Override Text File', self.myfile_overriden)
        assert_equals(None, self.absent_file)
        try:
            self.myfile_txt = 5
            raise ExpectationFailed(AttributeError, 'nothing')
        except AttributeError:
            pass

    def check_results(self, flow, checker):
        if False:
            for i in range(10):
                print('nop')
        for step in flow:
            checker.assert_artifact(step.name, 'myfile_txt', 'Regular Text File')
            checker.assert_artifact(step.name, 'myfile_utf8', 'UTF Text File 年')
            checker.assert_artifact(step.name, 'myfile_binary', 'UTF Text File 年'.encode(encoding='utf8'))
        checker.assert_artifact(step.name, 'myfile_overriden', 'Override Text File')