def hello(greeting, *args):
    if False:
        for i in range(10):
            print('nop')
    if len(args) == 0:
        print('%s!' % greeting)
    else:
        print('%s, %s!' % (greeting, ', '.join(args)))
hello('Hi')
hello('Hi', 'Sarah')
hello('Hello', 'Michael', 'Bob', 'Adam')
names = ('Bart', 'Lisa')
hello('Hello', *names)