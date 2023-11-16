import unittest
from Common.Misc import RemoveCComments
from Workspace.BuildClassObject import ArrayIndex

class TestRe(unittest.TestCase):

    def test_ccomments(self):
        if False:
            i = 10
            return i + 15
        TestStr1 = ' {0x01,0x02} '
        self.assertEquals(TestStr1, RemoveCComments(TestStr1))
        TestStr2 = " L'TestString' "
        self.assertEquals(TestStr2, RemoveCComments(TestStr2))
        TestStr3 = " 'TestString' "
        self.assertEquals(TestStr3, RemoveCComments(TestStr3))
        TestStr4 = '\n            {CODE({\n              {0x01, {0x02, 0x03, 0x04 }},// Data comment\n              {0x01, {0x02, 0x03, 0x04 }},// Data comment\n              })\n            }  /*\n               This is multiple line comments\n               The seconde line comment\n               */\n            // This is a comment\n        '
        Expect_TestStr4 = '{CODE({\n              {0x01, {0x02, 0x03, 0x04 }},\n              {0x01, {0x02, 0x03, 0x04 }},\n              })\n            }'
        self.assertEquals(Expect_TestStr4, RemoveCComments(TestStr4).strip())

    def Test_ArrayIndex(self):
        if False:
            return 10
        TestStr1 = '[1]'
        self.assertEquals(['[1]'], ArrayIndex.findall(TestStr1))
        TestStr2 = '[1][2][0x1][0x01][]'
        self.assertEquals(['[1]', '[2]', '[0x1]', '[0x01]', '[]'], ArrayIndex.findall(TestStr2))
if __name__ == '__main__':
    unittest.main()