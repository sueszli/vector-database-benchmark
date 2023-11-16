from lightning.app import LightningApp, LightningFlow

class EmptyFlow(LightningFlow):

    def run(self):
        if False:
            i = 10
            return i + 15
        pass
if __name__ == '__main__':
    _ = [1, 2, 3][4]
    app = LightningApp(EmptyFlow())