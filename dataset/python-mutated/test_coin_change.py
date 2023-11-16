import unittest

class Challenge(unittest.TestCase):

    def test_coin_change(self):
        if False:
            return 10
        coin_changer = CoinChanger()
        self.assertEqual(coin_changer.make_change([1, 2], 0), 0)
        self.assertEqual(coin_changer.make_change([1, 2, 3], 5), 5)
        self.assertEqual(coin_changer.make_change([1, 5, 25, 50], 10), 3)
        print('Success: test_coin_change')

def main():
    if False:
        i = 10
        return i + 15
    test = Challenge()
    test.test_coin_change()
if __name__ == '__main__':
    main()