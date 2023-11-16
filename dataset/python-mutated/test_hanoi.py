import unittest

class TestHanoi(unittest.TestCase):

    def test_hanoi(self):
        if False:
            i = 10
            return i + 15
        hanoi = Hanoi()
        num_disks = 3
        src = Stack()
        buff = Stack()
        dest = Stack()
        print('Test: None towers')
        self.assertRaises(TypeError, hanoi.move_disks, num_disks, None, None, None)
        print('Test: 0 disks')
        hanoi.move_disks(num_disks, src, dest, buff)
        self.assertEqual(dest.pop(), None)
        print('Test: 1 disk')
        src.push(5)
        hanoi.move_disks(num_disks, src, dest, buff)
        self.assertEqual(dest.pop(), 5)
        print('Test: 2 or more disks')
        for disk_index in range(num_disks, -1, -1):
            src.push(disk_index)
        hanoi.move_disks(num_disks, src, dest, buff)
        for disk_index in range(0, num_disks):
            self.assertEqual(dest.pop(), disk_index)
        print('Success: test_hanoi')

def main():
    if False:
        for i in range(10):
            print('nop')
    test = TestHanoi()
    test.test_hanoi()
if __name__ == '__main__':
    main()