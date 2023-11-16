class Human:

    def __init__(self, n, o):
        if False:
            return 10
        self.name = n
        self.occupation = o

    def do_work(self):
        if False:
            for i in range(10):
                print('nop')
        if self.occupation == 'tennis player':
            print(self.name, 'plays tennis')
        elif self.occupation == 'actor':
            print(self.name, 'shoots film')

    def speaks(self):
        if False:
            print('Hello World!')
        print(self.name, 'says how are you?')
tom = Human('tom cruise', 'actor')
tom.do_work()
tom.speaks()
maria = Human('maria sharapova', 'tennis player')
maria.do_work()
maria.speaks()