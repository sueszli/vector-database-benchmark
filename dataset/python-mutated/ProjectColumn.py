from datetime import datetime, timezone
from . import Framework

class ProjectColumn(Framework.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.get_project_column = self.g.get_project_column(8700460)
        self.move_project_column = self.g.get_project_column(8748065)

    def testGetProjectColumn(self):
        if False:
            return 10
        self.assertEqual(self.get_project_column.id, 8700460)
        self.assertEqual(self.get_project_column.name, 'c1')
        self.assertEqual(self.get_project_column.cards_url, 'https://api.github.com/projects/columns/8700460/cards')
        self.assertEqual(self.get_project_column.node_id, 'MDEzOlByb2plY3RDb2x1bW44NzAwNDYw')
        self.assertEqual(self.get_project_column.project_url, 'https://api.github.com/projects/4294766')
        self.assertEqual(self.get_project_column.url, 'https://api.github.com/projects/columns/8700460')
        self.assertEqual(self.get_project_column.created_at, datetime(2020, 4, 13, 20, 29, 53, tzinfo=timezone.utc))
        self.assertEqual(self.get_project_column.updated_at, datetime(2020, 4, 14, 18, 9, 38, tzinfo=timezone.utc))

    def testGetAllCards(self):
        if False:
            print('Hello World!')
        cards = self.get_project_column.get_cards(archived_state='all')
        self.assertEqual(cards.totalCount, 3)
        self.assertEqual(cards[0].id, 36285184)
        self.assertEqual(cards[0].note, 'Note3')
        self.assertEqual(cards[1].id, 36281526)
        self.assertEqual(cards[1].note, 'Note2')
        self.assertEqual(cards[2].id, 36281516)
        self.assertEqual(cards[2].note, 'Note1')

    def testGetArchivedCards(self):
        if False:
            return 10
        cards = self.get_project_column.get_cards(archived_state='archived')
        self.assertEqual(cards.totalCount, 1)
        self.assertEqual(cards[0].id, 36281516)
        self.assertEqual(cards[0].note, 'Note1')

    def testGetNotArchivedCards(self):
        if False:
            while True:
                i = 10
        cards = self.get_project_column.get_cards(archived_state='not_archived')
        self.assertEqual(cards.totalCount, 2)
        self.assertEqual(cards[0].id, 36285184)
        self.assertEqual(cards[0].note, 'Note3')
        self.assertEqual(cards[1].id, 36281526)
        self.assertEqual(cards[1].note, 'Note2')

    def testGetCards(self):
        if False:
            return 10
        cards = self.get_project_column.get_cards()
        self.assertEqual(cards.totalCount, 2)
        self.assertEqual(cards[0].id, 36285184)
        self.assertEqual(cards[0].note, 'Note3')
        self.assertEqual(cards[1].id, 36281526)
        self.assertEqual(cards[1].note, 'Note2')

    def testCreateCard(self):
        if False:
            i = 10
            return i + 15
        new_card = self.get_project_column.create_card(note='NewCard')
        self.assertEqual(new_card.id, 36290228)
        self.assertEqual(new_card.note, 'NewCard')

    def testDelete(self):
        if False:
            print('Hello World!')
        project_column = self.g.get_project_column(8747987)
        self.assertTrue(project_column.delete())

    def testEdit(self):
        if False:
            return 10
        self.move_project_column.edit('newTestColumn')
        self.assertEqual(self.move_project_column.id, 8748065)
        self.assertEqual(self.move_project_column.name, 'newTestColumn')

    def testMoveFirst(self):
        if False:
            return 10
        self.assertTrue(self.move_project_column.move(position='first'))

    def testMoveLast(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.move_project_column.move(position='last'))

    def testMoveAfter(self):
        if False:
            return 10
        self.assertTrue(self.move_project_column.move(position='after:8700460'))