from . import Framework

class Project1434(Framework.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()

    def testDelete(self):
        if False:
            i = 10
            return i + 15
        project = self.g.get_project(4102095)
        project.delete()

    def testEditWithoutParameters(self):
        if False:
            print('Hello World!')
        project = self.g.get_project(4101939)
        old_name = project.name
        project.edit()
        self.assertEqual(project.name, old_name)

    def testEditWithAllParameters(self):
        if False:
            for i in range(10):
                print('nop')
        project = self.g.get_project(4101939)
        project.edit('New Name', 'New Body', 'open')
        self.assertEqual(project.name, 'New Name')
        self.assertEqual(project.body, 'New Body')
        self.assertEqual(project.state, 'open')