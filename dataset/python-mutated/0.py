from lightning.app import LightningWork

class ExampleWork(LightningWork):

    def run(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        print(f'I received the following props: args: {args} kwargs: {kwargs}')
work = ExampleWork()
work.run(value=1)
work.run(value=1)
work.run(value=1)
work.run(value=1)
work.run(value=1)
work.run(value=10)