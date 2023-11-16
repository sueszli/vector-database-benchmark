from lightning.app import LightningWork, LightningApp, CloudCompute

class YourComponent(LightningWork):

    def run(self):
        if False:
            while True:
                i = 10
        print('RUN ANY PYTHON CODE HERE')
compute = CloudCompute('cpu')
worker = YourComponent(cloud_compute=compute)
app = LightningApp(worker)