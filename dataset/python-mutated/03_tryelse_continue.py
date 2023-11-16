def test_specific_values(self):
    if False:
        return 10
    for flags in self:
        if flags:
            try:
                self = 1
            except ValueError:
                continue
            else:
                self = 2
        self = 3

def call(*args):
    if False:
        for i in range(10):
            print('nop')
    try:
        return 5
    except KeyError:
        return 2
    except TypeError:
        return 3

def do_jump(self, arg):
    if False:
        for i in range(10):
            print('nop')
    try:
        arg(1)
    except ValueError:
        arg(2)
    else:
        try:
            arg(3)
        except ValueError:
            arg(4)

def _deliver(self, s, mailfrom, rcpttos):
    if False:
        print('Hello World!')
    try:
        mailfrom(1)
    except RuntimeError:
        mailfrom(2)
    except IndexError:
        for r in s:
            mailfrom()
    return