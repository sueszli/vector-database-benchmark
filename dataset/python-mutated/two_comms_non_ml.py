from lightning.app import LightningWork, LightningFlow, LightningApp
import time

class A(LightningWork):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.msg_changed = False
        self.new_msg = ''

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        for step in range(1000):
            time.sleep(1.0)
            if step % 10 == 0:
                self.msg_changed = True
                self.new_msg = f'A is at step: {step}'
                print(self.new_msg)

class B(LightningWork):

    def run(self, msg):
        if False:
            return 10
        print(f'B: message from A: {msg}')

class Example(LightningFlow):

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.a = A(parallel=True)
        self.b = B(parallel=True)

    def run(self):
        if False:
            return 10
        self.a.run()
        if self.a.msg_changed:
            self.a.msg_changed = False
            self.b.run(self.a.new_msg)
app = LightningApp(Example())