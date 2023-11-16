try:
    from st2common.runners.base_action import Action
except ImportError:
    Action = object

class PrintCtxAction(Action):

    def run(self, ctx=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        print(ctx)
if __name__ == '__main__':
    pca = PrintCtxAction()
    pca.run({'Hello': 'World', 'Foo': 'Bar'})