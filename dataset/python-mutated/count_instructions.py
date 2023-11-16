import sys
from manticore.native import Manticore
"\nCount the number of emulated instructions.\n\nThis example uses the context property of the Manticore object to store data\nthat's updated by the hook function. Manticore.context is needed to properly\nshare data when running with multiple worker processes.\n"
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.stderr.write(f'Usage: {sys.argv[0]} [binary]\n')
        sys.exit(2)
    m = Manticore(sys.argv[1])
    with m.locked_context() as context:
        context['count'] = 0

    @m.hook(None)
    def explore(state):
        if False:
            i = 10
            return i + 15
        with m.locked_context() as context:
            context['count'] += 1
    m.run()
    print(f"Executed {m.context['count']} instructions.")