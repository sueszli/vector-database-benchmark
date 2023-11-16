from runner.koan import *

class AboutDeletingObjects(Koan):

    def test_del_can_remove_slices(self):
        if False:
            while True:
                i = 10
        lottery_nums = [4, 8, 15, 16, 23, 42]
        del lottery_nums[1]
        del lottery_nums[2:4]
        self.assertEqual(___, lottery_nums)

    def test_del_can_remove_entire_lists(self):
        if False:
            print('Hello World!')
        lottery_nums = [4, 8, 15, 16, 23, 42]
        del lottery_nums
        with self.assertRaises(___):
            win = lottery_nums

    class ClosingSale:

        def __init__(self):
            if False:
                print('Hello World!')
            self.hamsters = 7
            self.zebras = 84

        def cameras(self):
            if False:
                i = 10
                return i + 15
            return 34

        def toilet_brushes(self):
            if False:
                return 10
            return 48

        def jellies(self):
            if False:
                print('Hello World!')
            return 5

    def test_del_can_remove_attributes(self):
        if False:
            print('Hello World!')
        crazy_discounts = self.ClosingSale()
        del self.ClosingSale.toilet_brushes
        del crazy_discounts.hamsters
        try:
            still_available = crazy_discounts.toilet_brushes()
        except AttributeError as e:
            err_msg1 = e.args[0]
        try:
            still_available = crazy_discounts.hamsters
        except AttributeError as e:
            err_msg2 = e.args[0]
        self.assertRegex(err_msg1, __)
        self.assertRegex(err_msg2, __)

    class ClintEastwood:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self._name = None

        def get_name(self):
            if False:
                print('Hello World!')
            try:
                return self._name
            except:
                return 'The man with no name'

        def set_name(self, name):
            if False:
                while True:
                    i = 10
            self._name = name

        def del_name(self):
            if False:
                while True:
                    i = 10
            del self._name
        name = property(get_name, set_name, del_name, "Mr Eastwood's current alias")

    def test_del_works_with_properties(self):
        if False:
            print('Hello World!')
        cowboy = self.ClintEastwood()
        cowboy.name = 'Senor Ninguno'
        self.assertEqual('Senor Ninguno', cowboy.name)
        del cowboy.name
        self.assertEqual(__, cowboy.name)

    class Prisoner:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self._name = None

        @property
        def name(self):
            if False:
                print('Hello World!')
            return self._name

        @name.setter
        def name(self, name):
            if False:
                for i in range(10):
                    print('nop')
            self._name = name

        @name.deleter
        def name(self):
            if False:
                print('Hello World!')
            self._name = 'Number Six'

    def test_another_way_to_make_a_deletable_property(self):
        if False:
            while True:
                i = 10
        citizen = self.Prisoner()
        citizen.name = 'Patrick'
        self.assertEqual('Patrick', citizen.name)
        del citizen.name
        self.assertEqual(__, citizen.name)

    class MoreOrganisedClosingSale(ClosingSale):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            self.last_deletion = None
            super().__init__()

        def __delattr__(self, attr_name):
            if False:
                while True:
                    i = 10
            self.last_deletion = attr_name

    def tests_del_can_be_overriden(self):
        if False:
            return 10
        sale = self.MoreOrganisedClosingSale()
        self.assertEqual(__, sale.jellies())
        del sale.jellies
        self.assertEqual(__, sale.last_deletion)