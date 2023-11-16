from integration.orquesta import base
from st2client import models
from st2common.constants import action as ac_const

class DatastoreFunctionTest(base.TestWorkflowExecution):

    @classmethod
    def set_kvp(cls, name, value, scope='system', secret=False):
        if False:
            while True:
                i = 10
        kvp = models.KeyValuePair(id=name, name=name, value=value, scope=scope, secret=secret)
        cls.st2client.keys.update(kvp)

    @classmethod
    def del_kvp(cls, name, scope='system'):
        if False:
            print('Hello World!')
        kvp = models.KeyValuePair(id=name, name=name, scope=scope)
        cls.st2client.keys.delete(kvp)

    def test_st2kv_system_scope(self):
        if False:
            return 10
        key = 'lakshmi'
        value = 'kanahansnasnasdlsajks'
        self.set_kvp(key, value)
        wf_name = 'examples.orquesta-st2kv'
        wf_input = {'key_name': 'system.%s' % key}
        execution = self._execute_workflow(wf_name, wf_input)
        output = self._wait_for_completion(execution)
        self.assertEqual(output.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertIn('output', output.result)
        self.assertIn('value', output.result['output'])
        self.assertEqual(value, output.result['output']['value'])
        self.del_kvp(key)

    def test_st2kv_user_scope(self):
        if False:
            for i in range(10):
                print('nop')
        key = 'winson'
        value = 'SoDiamondEng'
        self.set_kvp(key, value, 'user')
        wf_name = 'examples.orquesta-st2kv'
        wf_input = {'key_name': key}
        execution = self._execute_workflow(wf_name, wf_input)
        output = self._wait_for_completion(execution)
        self.assertEqual(output.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertIn('output', output.result)
        self.assertIn('value', output.result['output'])
        self.assertEqual(value, output.result['output']['value'])

    def test_st2kv_decrypt(self):
        if False:
            i = 10
            return i + 15
        key = 'kami'
        value = 'eggplant'
        self.set_kvp(key, value, secret=True)
        wf_name = 'examples.orquesta-st2kv'
        wf_input = {'key_name': 'system.%s' % key, 'decrypt': True}
        execution = self._execute_workflow(wf_name, wf_input)
        output = self._wait_for_completion(execution)
        self.assertEqual(output.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertIn('output', output.result)
        self.assertIn('value', output.result['output'])
        self.assertEqual(value, output.result['output']['value'])
        self.del_kvp(key)