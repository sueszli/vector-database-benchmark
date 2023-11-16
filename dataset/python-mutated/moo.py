import cow

def Moo():
    if False:
        for i in range(10):
            print('nop')
    for i in range(1, 100):
        cow.Say('Moo')
    for i in range(1, 100):
        cow.Say('Ooom')
if __name__ == '__main__':
    Moo()