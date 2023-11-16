from ut_helpers_ir import exec_instruction

class TestDataCache(object):

    def test_cache(self):
        if False:
            return 10
        'Test CACHE execution'
        exec_instruction('CACHE 0x0, (R0)', [], [])