import unittest
from dist_pass_test_base import PassConflictChecker
from model_zoo import resnet_model
from paddle.distributed.passes import new_pass

class CheckPassConflictTest1(PassConflictChecker):

    def pass_config(self):
        if False:
            while True:
                i = 10
        return [new_pass('fuse_all_reduce', {'max_memory_size': 1024 * 1024}), new_pass('fuse_elewise_add_act')]

    def test_resnet(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_main(resnet_model, batch_size=32)

class CheckPassConflictTest2(PassConflictChecker):

    def pass_config(self):
        if False:
            return 10
        return [new_pass('fuse_elewise_add_act'), new_pass('fuse_all_reduce', {'max_memory_size': 1024 * 1024})]

    def test_resnet(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(Exception):
            self.check_main(resnet_model, batch_size=32)
if __name__ == '__main__':
    unittest.main()