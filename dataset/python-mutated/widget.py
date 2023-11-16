"""As a simple example of Python Fire, a Widget serves no clear purpose."""
import fire

class Widget(object):

    def whack(self, n=1):
        if False:
            for i in range(10):
                print('nop')
        'Prints "whack!" n times.'
        return ' '.join(('whack!' for _ in range(n)))

    def bang(self, noise='bang'):
        if False:
            while True:
                i = 10
        'Makes a loud noise.'
        return '{noise} bang!'.format(noise=noise)

def main():
    if False:
        i = 10
        return i + 15
    fire.Fire(Widget(), name='widget')
if __name__ == '__main__':
    main()